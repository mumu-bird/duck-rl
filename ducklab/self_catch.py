"""Self-toss MDP: a preloaded free ball, a walking duck and two high-level decisions."""

import mujoco
import numpy as np

from .policy import Controller
from .retrieval.scene import make_scene, place

DT = 0.005
RADIUS = 0.010
OPEN = np.deg2rad(39.375)
OBS, ACT = 20, 6
T, X, Y, Z, BASE, MX, MY, MZ, LIP, BOTH, OTHER, FLOOR, UP, BASEZ = range(14)


def make_model():
    item = dict(
        id="self_toss_ball",
        family="ball",
        length=2 * RADIUS,
        half_m=[RADIUS] * 3,
        mass_kg=0.0025,
    )
    m, _ = make_scene(item, bin_xy=(3, 2), moving_jaw=True)
    m.geom("parcel_hull").rgba[:] = (0.99, 0.10, 0.16, 1)
    assert m.opt.timestep == DT and not m.eq_active0.any()
    return m


def judge(tr):
    """World-state referee, blind to policy actions. Never accepts floor bounces."""
    pre, a = tr[(tr[:, T] >= -0.08) & (tr[:, T] < 0)], tr[tr[:, T] >= 0]
    waiting = tr[tr[:, T] < 0]
    armed = bool(
        pre.size
        and np.all(pre[:, BOTH])
        and not waiting[:, OTHER].any()
        and not waiting[:, FLOOR].any()
    )
    upright = bool(
        np.isfinite(tr).all()
        and np.all(tr[:, UP] > 0.60)
        and np.all(tr[:, BASEZ] > 0.065)
    )
    held = bool(
        np.all(a[-50:, BOTH]) and not a[:, OTHER].any() and not a[:, FLOOR].any()
    )
    r = dict(
        success=False,
        armed=armed,
        upright=upright,
        ground=bool(a[:, FLOOR].any()),
        held=held,
        released=False,
        recaught=False,
        air_s=0.0,
        rise_m=0.0,
        forward_m=0.0,
        chase_m=0.0,
        miss_m=0.4,
        separation_m=0.0,
        release_s=None,
        catch_s=None,
    )
    if not armed or not upright:
        return r
    end = int(np.argmax(a[:, OTHER])) if a[:, OTHER].any() else len(a)
    edges = np.diff(np.r_[False, ~a[:end, LIP].astype(bool), False].astype(int))
    for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if j - i < 8 or j >= len(a) or i == 0:
            continue
        air = a[i:j]
        peak = int(np.argmax(air[:, Z]))
        if peak < 1 or air[1, Z] <= air[0, Z]:
            continue
        last = a[j]
        gap = np.linalg.norm(air[:, X : Z + 1] - air[:, MX : MZ + 1], axis=1)
        desc = a[min(j - 1, max(i + peak + 1, i + 14)) : j + 1]
        miss = np.linalg.norm(desc[:, X : Z + 1] - desc[:, MX : MZ + 1], axis=1).min()
        clear_flight = not a[i : j + 1, FLOOR].any()
        recaught = bool(
            j < end
            and last[LIP]
            and last[Z] > 0.075
            and clear_flight
            and j - i > peak + 1
            and (air[-1, Z] - air[-2, Z]) / DT < -0.06
        )
        r.update(
            air_s=float((j - i) * DT),
            rise_m=float(air[peak, Z] - air[0, Z]),
            forward_m=float(last[X] - air[0, X]),
            chase_m=float(last[BASE] - air[0, BASE]),
            separation_m=float(gap.max()),
            miss_m=float(miss),
            released=True,
            recaught=recaught,
            release_s=float(air[0, T]),
            catch_s=float(last[T]) if recaught else None,
        )
        r["success"] = bool(
            recaught
            and held
            and r["air_s"] >= 0.08
            and r["rise_m"] >= 0.012
            and r["forward_m"] >= 0.015
            and r["chase_m"] >= 0.005
            and not r["ground"]
        )
        break
    return r


def reward(r):
    if not r["armed"] or not r["upright"]:
        return -2.0
    if not r["released"]:
        return -1.0
    height = min(1.0, r["rise_m"] / 0.020)
    shape = (
        height
        + 0.35 * np.clip(r["forward_m"] / 0.030, 0, 1)
        + 0.35 * np.clip(r["chase_m"] / 0.012, 0, 1)
    )
    near = np.exp(-((r["miss_m"] / 0.028) ** 2))
    return float(
        shape
        + 1.5 * near
        + 1.2 * r["recaught"]
        + (1.2 + height) * r["held"] * r["recaught"]
        + 6.0 * r["success"]
    )


def scale(u, low, high):
    return low + (high - low) * (0.5 + 0.5 * float(u))


class SelfCatchEnv:
    def __init__(self, fine_jaw=False):
        self.fine_jaw = fine_jaw
        self.m = m = make_model()
        self.upper, self.lower, self.ball, self.floor = (
            m.geom(n).id for n in ("upper_lip", "lower_lip", "parcel_hull", "floor")
        )
        self.head, self.base, self.bid = (
            m.body(n).id for n in ("head_roll", "trunk_base", "parcel")
        )
        self.jaw = m.actuator("bill_servo").id
        self.v = m.joint("parcel_free").dofadr[0]

    def reset(self, seed, record=False):
        self.d = d = mujoco.MjData(self.m)
        self.c = c = Controller(self.m, d)
        rng = np.random.default_rng(seed)
        d.qpos[c.q] += rng.uniform(-0.001, 0.001, len(c.q))
        d.joint("bill_open").qpos[0] = 0.31
        mujoco.mj_forward(self.m, d)
        offset = (rng.uniform(0.048, 0.053), rng.uniform(-0.0025, 0.0025), -0.018)
        place(d, "parcel", d.xpos[self.head] + d.xmat[self.head].reshape(3, 3) @ offset)
        mujoco.mj_forward(self.m, d)
        self.trace, self.frames = [], [] if record else None
        self.cmd = np.zeros(13, np.float32)
        self.phase = 0
        self.a0 = np.zeros(ACT, np.float32)
        self.pos = None
        return self.obs()

    def obs(self):
        d, c = self.d, self.c
        tip = (d.geom_xpos[self.upper] + d.geom_xpos[self.lower]) * 0.5
        # Simulator-state observations (not an RGB policy); no future trajectory or referee signal.
        v = np.r_[
            (d.xpos[self.bid] - tip) * 45,
            d.qvel[self.v : self.v + 3] * 2,
            d.qpos[c.q[5:7]],
            d.qvel[c.v[5:7]] * 0.12,
            d.qvel[0] * 4,
            d.joint("bill_open").qpos[0] * 2,
            1 - self.phase,
            self.phase,
            self.a0,
        ]
        return np.clip(v, -4, 4).astype(np.float32)

    def sample(self):
        d = self.d
        contacts = d.contact.geom
        # Only a pair involving the ball matters; contacts with clutter/skull/floor invalidate a catch.
        pair = (
            contacts[contacts[:, 0] == self.ball, 1].tolist()
            + contacts[contacts[:, 1] == self.ball, 0].tolist()
        )
        hi, lo = self.upper in pair, self.lower in pair
        tip = (d.geom_xpos[self.upper] + d.geom_xpos[self.lower]) * 0.5
        self.trace.append(
            (
                round(d.time - self.start - DT, 5),
                *d.xpos[self.bid],
                d.xpos[self.base, 0],
                *tip,
                hi or lo,
                hi and lo,
                any(g not in (self.upper, self.lower) for g in pair),
                self.floor in pair,
                d.xmat[self.base, 8],
                d.xpos[self.base, 2],
            )
        )
        if self.frames is not None:
            # Cartesian fields are for the start of mj_step. Pair them with the matching qpos.
            self.frames.append(self.pos.copy())
            self.pos[:] = d.qpos

    def tick(self, k, nc=0.0, hc=0.0, recover=99, close=99, catch_speed=None):
        p, d, c = self.p, self.d, self.c
        walk = -p["lead"] <= k < 22
        slot = "walk" if walk else "stand"
        self.cmd[0] = (
            (p["speed"] if catch_speed is None else catch_speed) if walk else 0
        )
        a = c.act(slot, self.cmd)
        a[5:7] = (
            (p["n0"], p["h0"])
            if k < 0
            else (p["n1"], p["h1"])
            if k < recover
            else (nc, hc)
        )
        d.ctrl[self.jaw] = OPEN if p["release"] <= k < close else 0
        if self.fine_jaw:
            sub = 0

            def after_step():
                nonlocal sub
                self.sample()
                sub += 1
                d.ctrl[self.jaw] = OPEN if p["release"] <= k + sub / 4 < close else 0

            c.step(a, slot, after_step)
        else:
            c.step(a, slot, self.sample)

    def step(self, action):
        u = np.clip(action, -1, 1)
        if self.phase == 0:
            self.a0 = np.array(u, np.float32)
            self.p = p = dict(
                n0=scale(u[0], -0.15, 0.94 if self.fine_jaw else 0.75),
                h0=scale(u[1], 0.25, 1.30),
                n1=scale(u[2], -0.48, 0.85),
                h1=scale(u[3], -1.48, -0.05),
                release=round(scale(u[4], 0, 6)),
                lead=round(scale(u[5], 2, 15)),
                speed=0.19,
            )
            if self.fine_jaw:
                p["release"] = round(4 * scale(u[4], 0, 6)) / 4
            for k in range(45):
                a = self.c.act("stand", self.cmd)
                if k >= 10:
                    a[5:7] += min(1.0, (k - 9) / 20) * (
                        np.array((p["n0"], p["h0"])) - a[5:7]
                    )
                self.c.step(a, "stand")
            self.start = self.d.time + 0.4
            if self.frames is not None:
                self.pos = self.d.qpos.copy()
            for k in range(-20, 4):
                self.tick(k)
            self.phase = 1
            return self.obs(), 0.0, False, {}
        if self.phase != 1:
            raise ValueError("reset before starting another episode")
        self.phase = 2
        nc, hc = scale(u[0], -0.95, 0.5), scale(u[1], -0.80, 1.35)
        recover = round(scale(u[2], 4, 12))
        close = round(scale(u[3], 5, 20))
        if self.fine_jaw:
            close = round(4 * scale(u[3], 5, 20)) / 4
        speed = scale(u[4], 0.08, 0.28)
        # Optional yaw correction based on lateral ball motion, not a world-state intervention.
        self.cmd[2] = scale(u[5], -0.24, 0.24)
        for k in range(4, 44):
            if k == 22:
                self.cmd[2] = 0
            self.tick(k, nc, hc, recover, close, speed)
        r = judge(np.asarray(self.trace))
        return np.zeros(OBS, np.float32), reward(r), True, r
