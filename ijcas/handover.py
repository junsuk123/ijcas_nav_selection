"""state_handover (Notion 05, G5): causal alignment of local engines, reset detection, single
output authority with switch-jump logging.

Causal alignment (R08).  A local engine (OpenVINS, FAST-LIO2) reports a metric pose in its own
gravity-aligned odometry frame.  While the global reference (PX4 EKF2 global position) is judged
*Usable* by the selector, pairs (local position, reference position) at common times are kept in
a sliding window and a 4-DOF transform (yaw + translation; no scale, no roll/pitch) is fitted.
It is only accepted with >= ``min_path_m`` of travelled path in the window (yaw observability;
a single GNSS position never defines yaw).  When the reference stops being usable the transform
is frozen and its age grows; the selector treats a frozen alignment as limited.  Ground truth is
never used.

Reset detection.  A jump of the local position inconsistent with the engine's own velocity
(|dp - v dt| > ``reset_jump_m``) marks a reset: the alignment is invalidated and must be
re-established (R08: switching to that engine is held until then).

Handover.  ``HandoverOutput`` turns the selected candidate's global estimate into the single
output.  Two outputs are logged: ``raw`` (jumps at switches) and ``continuous`` (the jump is
absorbed into an offset that decays with ``tau_s``; it inherits the previous engine's global error
for a while, which is why both are evaluated, Notion 03).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


def fit_4dof(src: np.ndarray, dst: np.ndarray):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - ms[:2], dst[:, :2] - md[:2]
    yaw = float(np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum()))
    rot = R.from_euler("z", yaw)
    return yaw, md - rot.apply(ms)


@dataclass
class Alignment:
    window_s: float = 30.0
    min_path_m: float = 10.0
    reset_jump_m: float = 1.0
    pairs: deque = field(default_factory=lambda: deque())
    yaw: float | None = None
    trans: np.ndarray | None = None
    updated_at: float | None = None
    residual_m: float | None = None
    resets: int = 0
    _last: tuple | None = None

    def observe_local(self, t: float, pos: np.ndarray, vel: np.ndarray) -> bool:
        """Feed the engine output; returns True if a reset was detected at this sample."""
        reset = False
        if self._last is not None:
            t0, p0, v0 = self._last
            dt = t - t0
            if 0.0 < dt < 1.0:
                pred = p0 + 0.5 * (v0 + vel) * dt
                if np.linalg.norm(pos - pred) > self.reset_jump_m:
                    reset = True
        if self._last is not None and not reset and self.updated_at is not None and t > self.updated_at:
            self.path_since_update_m += float(np.linalg.norm(pos - self._last[1]))
        self._last = (t, pos.copy(), vel.copy())
        if reset:
            self.resets += 1
            self.pairs.clear()
            self.yaw, self.trans, self.updated_at = None, None, None
            self.established_at = None
        return reset

    def add_pair(self, t: float, local: np.ndarray, ref: np.ndarray):
        self.pairs.append((t, local.copy(), ref.copy()))
        while self.pairs and self.pairs[0][0] < t - self.window_s:
            self.pairs.popleft()
        if len(self.pairs) < 20:
            return
        loc = np.array([p[1] for p in self.pairs])
        ref_ = np.array([p[2] for p in self.pairs])
        path = float(np.linalg.norm(np.diff(ref_[:, :2], axis=0), axis=1).sum())
        if path < self.min_path_m:
            return
        yaw, trans = fit_4dof(loc, ref_)
        rot = R.from_euler("z", yaw)
        self.residual_m = float(np.sqrt(np.mean(np.sum((rot.apply(loc) + trans - ref_) ** 2, 1))))
        self.yaw, self.trans, self.updated_at = yaw, trans, t
        self.path_since_update_m = 0.0
        if self.established_at is None:
            self.established_at = t

    invalidations: list = field(default_factory=list)
    established_at: float | None = None      # first valid fit since the last reset/invalidation
    path_since_update_m: float = 0.0         # local path flown since the alignment was last fitted

    def invalidate(self, t: float, reason: str):
        """R08: the engine lost its essential input; its alignment must be re-established."""
        self.pairs.clear()
        self.yaw, self.trans, self.updated_at = None, None, None
        self.established_at = None
        self.invalidations.append((t, reason))

    def mature(self, t: float, min_age_s: float = 20.0, max_residual_m: float = 1.5) -> bool:
        """Established long enough with a good fit to testify against GNSS (no young witness)."""
        return (self.valid and self.established_at is not None
                and t - self.established_at >= min_age_s
                and (self.residual_m is None or self.residual_m <= max_residual_m))

    def error_bound(self, drift_rate: float, floor_m: float = 0.5) -> float:
        """Expected bound of this engine's global error: fit residual + drift since the fit."""
        return (self.residual_m or 0.0) + drift_rate * self.path_since_update_m + floor_m

    @property
    def valid(self) -> bool:
        return self.yaw is not None

    def age(self, t: float) -> float:
        return float("inf") if self.updated_at is None else t - self.updated_at

    def to_global(self, pos, quat_xyzw, vel):
        rot = R.from_euler("z", self.yaw)
        return (rot.apply(pos) + self.trans, (rot * R.from_quat(quat_xyzw)).as_quat(),
                rot.apply(vel))


@dataclass
class HandoverOutput:
    tau_s: float = 5.0
    current: str | None = None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    offset_t: float = 0.0
    last_raw: np.ndarray | None = None
    last_cont: np.ndarray | None = None
    switches: list = field(default_factory=list)

    def update(self, t: float, selected: str | None, pos_global: np.ndarray | None,
               vel_global: np.ndarray | None, yaw_global: float | None):
        """Returns (raw_pos, continuous_pos) or (None, None) for no-valid-nav."""
        if selected is None or pos_global is None:
            if self.current is not None:
                self.switches.append({"t": t, "from": self.current, "to": None})
            self.current = None
            return None, None
        if selected != self.current:
            jump = None
            if self.last_raw is not None:
                jump = (pos_global - self.last_raw).tolist()
                # continuous output keeps the previous output position at the switch
                self.offset = (self.last_cont if self.last_cont is not None else self.last_raw) \
                    - pos_global
                self.offset_t = t
            self.switches.append({"t": t, "from": self.current, "to": selected,
                                  "position_jump": jump})
            self.current = selected
        decay = np.exp(-(t - self.offset_t) / self.tau_s)
        cont = pos_global + self.offset * decay
        self.last_raw, self.last_cont = pos_global.copy(), cont.copy()
        return pos_global, cont
