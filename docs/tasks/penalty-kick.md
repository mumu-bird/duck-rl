# Fixed-Start Penalty Kick

## Goal

Train a policy that scores from one fixed, reproducible ball/robot/goal arrangement, using the official Wonder Duck kick task as the starting point where the WonderRL release supports it.

## Constraints

- Keep the official policy observation, action, actuator, and control-rate contract unchanged.
- Keep ball and goal positions fixed for the first version. Do not add simulator-only ball state or camera observations to the policy input.
- Derive scene dimensions from the official model/assets; do not guess scale or contact parameters.
- Preserve the official kick policy/config as an untouched baseline.

## Reward structure

Use the WonderRL task API and normalize terms to comparable scales. Begin with the following structure, then tune weights from measured rollout magnitudes:

```text
reward = goal_bonus
       + progress_toward_goal
       - terminal_fall_penalty
       - joint_limit_penalty
       - excessive_action_change_penalty
```

Do not reward ball motion alone if it can move away from the goal. Score only when the ball crosses the goal plane within the goal opening. Treat falls, invalid numeric states, and joint-limit violations as explicit episode outcomes.

## Training sequence

1. Evaluate the official kick baseline on the fixed scene and save 100 evaluation episodes.
2. Warm-start from the official kick checkpoint if supported; otherwise use the official kick task's documented initialization.
3. Train with PPO using the official project and keep evaluation seeds separate from training seeds.
4. After nominal performance stabilizes, introduce bounded friction/actuator randomization supported by the official model. Do not randomize ball/goal placement until the policy has a real observation source for those changes.

## Acceptance

Use `configs/acceptance.toml`: at least 100 holdout episodes, success rate at least 70%, terminal-fall rate strictly below 5%, and zero episodes with joint-limit violations or nonfinite values. Report the official kick baseline and candidate side by side.
