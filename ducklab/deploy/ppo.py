"""Self-contained PPO for the deployable contract (obs[61] -> actions[14]).

The action space is exactly the robot's: 14 joint offsets from the home pose,
tanh-squashed to [-1, 1] and written straight into the contract. Warm-starting
from the official walk policy (behaviour cloning on expert rollouts) gives the
network a standing/walking prior before task fine-tuning, which is what makes
short training runs viable; the exported artefact is still a single
obs[1,61] -> actions[1,14] ONNX, loadable into any robotd policy slot.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_size=61, action_size=14, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_size)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -1.0))
        for head in (self.actor, self.critic):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs):
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


class DeployPPO:
    def __init__(self, device=None, hidden=256, lr=3e-4, clip_ratio=0.2,
                 update_epochs=10, minibatch_size=512, entropy_coef=0.005,
                 value_coef=0.5, target_kl=0.015, gamma=0.99, gae_lambda=0.95,
                 anchor=None, anchor_coef=0.0):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # anchor: callable(obs_batch_ndarray) -> expert actions, used as an
        # imitation regulariser so task fine-tuning cannot drift away from the
        # stand/walk prior the DAgger warm start installed.
        self.anchor = anchor
        self.anchor_coef = anchor_coef
        self.net = ActorCritic(hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer = []          # list of episodes: dict of arrays
        self.update_count = 0
        self.updates_seconds = 0.0

    # ------------------------------------------------------------- acting

    @torch.no_grad()
    def act_batch(self, observations, deterministic=False):
        array = np.asarray(observations, dtype=np.float32)
        assert array.ndim == 2 and array.shape[1] == 61 and np.isfinite(array).all()
        obs = torch.as_tensor(array, device=self.device)
        mean, value = self.net(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            dist = torch.distributions.Normal(mean, self.net.log_std.exp())
            latent = dist.sample()
            action = torch.tanh(latent)
            correction = torch.log(1.0 - action.square() + 1e-6)
            log_prob = (dist.log_prob(latent) - correction).sum(dim=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy() if log_prob is not None else None, \
            value.cpu().numpy()

    # ------------------------------------------------------------- buffer

    def store_episode(self, obs, actions, log_probs, values, rewards, next_value, done):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nxt = values[t + 1] if t + 1 < len(values) else next_value
            nonterminal = 0.0 if (done and t == len(rewards) - 1) else 1.0
            delta = rewards[t] + self.gamma * nxt * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        self.buffer.append(dict(
            obs=obs, actions=actions,
            log_probs=np.asarray(log_probs, dtype=np.float32),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
        ))

    def clear(self):
        self.buffer = []

    # ------------------------------------------------------------- learning

    def update(self):
        started = time.time()
        obs = np.concatenate([e["obs"] for e in self.buffer])
        actions = np.concatenate([e["actions"] for e in self.buffer])
        old_logp = np.concatenate([e["log_probs"] for e in self.buffer])
        adv = np.concatenate([e["advantages"] for e in self.buffer])
        ret = np.concatenate([e["returns"] for e in self.buffer])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        old_t = torch.as_tensor(old_logp, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        stats = {}
        for _ in range(self.update_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = order[start:start + self.minibatch_size]
                mb = torch.as_tensor(idx, device=self.device)
                mean, value = self.net(obs_t[mb])
                if self.anchor is not None and self.anchor_coef > 0:
                    expert = torch.as_tensor(
                        self.anchor(obs[mb.cpu().numpy()]), dtype=torch.float32,
                        device=self.device)
                    anchor_loss = ((torch.tanh(mean) - expert) ** 2).mean()
                else:
                    anchor_loss = torch.zeros((), device=self.device)
                dist = torch.distributions.Normal(mean, self.net.log_std.exp())
                latent = torch.atanh(torch.clamp(act_t[mb], -0.999, 0.999))
                correction = torch.log(1.0 - act_t[mb].square() + 1e-6)
                log_prob = (dist.log_prob(latent) - correction).sum(dim=-1)
                ratio = (log_prob - old_t[mb]).exp()
                approx_kl = (old_t[mb] - log_prob).mean().item()
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = ((value - ret_t[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy + self.anchor_coef * anchor_loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
            if approx_kl > 1.5 * self.target_kl:
                break
        self.update_count += 1
        stats = dict(
            policy_loss=float(policy_loss.item()), value_loss=float(value_loss.item()),
            entropy=float(entropy.item()), approx_kl=float(approx_kl),
            anchor_loss=float(anchor_loss.item()), transitions=n,
        )
        self.updates_seconds += time.time() - started
        self.clear()
        return stats

    # ------------------------------------------------------------- warm start

    def behaviour_clone(self, observations, actions, epochs=8, batch_size=256, lr=1e-3):
        """Supervised warm start on expert (official policy) rollouts."""
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
        act = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = len(obs)
        for epoch in range(epochs):
            order = np.random.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                mb = torch.as_tensor(order[start:start + batch_size], device=self.device)
                mean, _ = self.net(obs[mb])
                loss = ((torch.tanh(mean) - act[mb]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(mb)
            if epoch == epochs - 1:
                return total / n
        return total / n

    # ------------------------------------------------------------- io

    def state_dict(self):
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def save(self, path):
        torch.save(dict(model=self.state_dict(), update=self.update_count), path)

    def load(self, path):
        blob = torch.load(path, map_location=self.device)
        self.load_state_dict(blob["model"])
        self.update_count = blob.get("update", 0)

    def export_onnx(self, path, opset=18):
        """Export exactly the deployed contract: obs[1,61] -> actions[1,14]."""
        self.net.eval()
        dummy = torch.zeros(1, 61, device=self.device)
        try:
            torch.onnx.export(
                self.net, (dummy,), str(path),
                input_names=["obs"], output_names=["actions"],
                opset_version=opset,
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                dynamo=False,
            )
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(
                self.net, (dummy,), str(path),
                input_names=["obs"], output_names=["actions"],
                opset_version=opset,
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            )
        return path
