"""G8 Stage B: outer-loop position control on the *selected* navigation output (pure numpy).

PX4 keeps its inner attitude/rate loops (on its own EKF2 attitude); the position/velocity loop is
ours and uses only /ijcas/nav/selected (PX4 local ENU frame).  Commands go to PX4 as OFFBOARD
attitude + collective thrust setpoints (MAVLink SET_ATTITUDE_TARGET), so the navigation selection
directly decides where the vehicle flies.  The selected output is never fed back into PX4 as an
aiding source (Notion 03: no circular fusion).

Reference: vertical take-off to the loop's first waypoint, hover, then the scene loop.
No valid navigation: level attitude at hover thrust for ``hold_s`` then a slow descent (abort).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

G = 9.80665
ENU2NED = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
FLU2FRD = R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


@dataclass
class OuterLoop:
    traj: object                       # LoopTrajectory (world ENU)
    takeoff_s: float = 6.0
    ground_z: float = 0.15
    kp: tuple = (1.2, 1.2, 1.6)
    kd: tuple = (1.8, 1.8, 2.2)
    ki_z: float = 0.15
    max_tilt_deg: float = 25.0
    hover_thrust: float = 0.5
    hold_s: float = 3.0
    offset_world_from_px4: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _iz: float = 0.0
    _last_t: float | None = None
    _no_nav_since: float | None = None
    aborted: bool = False

    def reference(self, t):
        s = self.traj.sample(t)
        p, v, a, yaw = s.position.copy(), s.velocity.copy(), s.acceleration.copy(), s.yaw
        if t < self.takeoff_s:                       # smooth vertical climb from the ground
            u = t / self.takeoff_s
            blend = 0.5 * (1 - np.cos(np.pi * u))
            dblend = 0.5 * np.pi / self.takeoff_s * np.sin(np.pi * u)
            ddblend = 0.5 * (np.pi / self.takeoff_s) ** 2 * np.cos(np.pi * u)
            z0, z1 = self.ground_z, p[2]
            p[2] = z0 + (z1 - z0) * blend
            v[2] = (z1 - z0) * dblend
            a[2] = (z1 - z0) * ddblend
        return p, v, a, yaw

    def command(self, t, nav):
        """nav = (pos, quat_xyzw, vel) in PX4 local ENU, or None.  Returns
        (quat_wxyz FRD->NED, thrust 0..1, info)."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        p_r, v_r, a_r, yaw = self.reference(t)
        p_r = p_r - self.offset_world_from_px4             # into the navigation (PX4) frame
        if nav is None:
            self._no_nav_since = t if self._no_nav_since is None else self._no_nav_since
            if t - self._no_nav_since > self.hold_s:
                self.aborted = True
            thrust = self.hover_thrust * (0.92 if self.aborted else 1.0)
            q = (ENU2NED * R.from_euler("z", yaw) * FLU2FRD).as_quat()
            return np.array([q[3], q[0], q[1], q[2]]), float(thrust), \
                {"mode": "abort_descent" if self.aborted else "no_nav_hold"}
        self._no_nav_since = None
        pos, _, vel = nav
        e_p, e_v = p_r - pos, v_r - vel
        self._iz = float(np.clip(self._iz + e_p[2] * dt, -2.0, 2.0))
        a_cmd = a_r + np.asarray(self.kp) * e_p + np.asarray(self.kd) * e_v
        h = np.linalg.norm(a_cmd[:2])
        lim = np.tan(np.radians(self.max_tilt_deg)) * G
        if h > lim:
            a_cmd[:2] *= lim / h
        f = a_cmd + np.array([0.0, 0.0, G])
        z_b = f / np.linalg.norm(f)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        r_enu = R.from_matrix(np.column_stack([x_b, y_b, z_b]))
        thrust = float(np.clip(self.hover_thrust * np.linalg.norm(f) / G + self.ki_z * self._iz,
                               0.05, 0.95))
        q = (ENU2NED * r_enu * FLU2FRD).as_quat()
        return np.array([q[3], q[0], q[1], q[2]]), thrust, \
            {"mode": "track", "pos_err": float(np.linalg.norm(e_p)), "ref": p_r.tolist()}
