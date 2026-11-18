"""Cruise-speed trajectories with curvature-limited speed (outdoor scenarios, 3 / 6 / 9 m/s).

A constant speed along the scene loop would demand up to 40 m/s^2 in the turns at 9 m/s, which a 0.5 m
class quadrotor cannot fly.  ``ProfiledLoopTrajectory`` keeps the same path (LoopTrajectory) and replaces
only the time parametrisation s(t):

    v(s) = min(v_cruise, sqrt(a_lat_max * R(s)))           R = path curvature radius
    forward / backward passes with |dv/dt| <= a_long_max, starting from rest after the hover,
    light smoothing of v(s); t(s) = integral ds / v, inverted by interpolation.

Position, velocity, acceleration, attitude and body rates then follow from the path derivatives exactly
as in LoopTrajectory (sample / table use ``arc``), so the IMU stays consistent with the motion.
"""
from __future__ import annotations

import numpy as np

from .trajectory import LoopTrajectory


class ProfiledLoopTrajectory(LoopTrajectory):
    def __init__(self, waypoints, cruise_mps: float, a_lat_max: float = 6.0, a_long_max: float = 2.0,
                 duration_s: float = 200.0, ds: float = 0.05, smooth_m: float = 1.5, start_ramp_s: float = 4.0,
                 dt: float = 0.005, **kw):
        kw.setdefault("ramp_s", start_ramp_s)
        super().__init__(waypoints, speed=cruise_mps, **kw)
        self.cruise = float(cruise_mps)
        laps = int(np.ceil(cruise_mps * duration_s / self.length)) + 2
        s = np.arange(0.0, laps * self.length + ds, ds)
        sm = s % self.length
        d1, d2 = self._d1(sm), self._d2(sm)
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
        v = np.minimum(self.cruise, np.sqrt(a_lat_max / np.maximum(kappa, 1e-6)))
        n = max(1, int(smooth_m / ds))
        k = np.exp(-0.5 * (np.arange(-3 * n, 3 * n + 1) / n) ** 2)
        k /= k.sum()
        v = np.minimum(v, np.convolve(np.pad(v, 3 * n, mode="edge"), k, mode="valid"))
        for i in range(1, len(v)):                                   # accelerate out of turns
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_long_max * ds))
        for i in range(len(v) - 2, -1, -1):                          # brake before turns
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long_max * ds))
        v_of_s = np.convolve(np.pad(v, 3 * n, mode="edge"), k, mode="valid")
        # time integration with a cosine start ramp (as LoopTrajectory), so s(t), ds/dt, d2s/dt2 are smooth
        tt = np.arange(0.0, duration_s + dt, dt)
        ss = np.zeros_like(tt)
        vv = np.zeros_like(tt)
        r = max(start_ramp_s, 1e-6)
        for i in range(len(tt) - 1):
            w = 0.5 * (1.0 - np.cos(np.pi * min(tt[i], r) / r))
            vv[i] = w * np.interp(ss[i], s, v_of_s)
            ss[i + 1] = ss[i] + vv[i] * dt
        vv[-1] = vv[-2]
        m = max(1, int(0.05 / dt))
        acc = np.gradient(vv, tt)
        acc = np.convolve(np.pad(acc, m, mode="edge"), np.ones(2 * m + 1) / (2 * m + 1), mode="valid")
        self._t, self._s, self._v, self._a = tt, ss, vv, acc
        self.max_speed = float(vv.max())

    def arc(self, t: float):
        t1 = t - self.hover_s
        if t1 <= 0.0:
            return 0.0, 0.0, 0.0
        if t1 >= self._t[-1]:
            raise ValueError("trajectory horizon exceeded; increase duration_s")
        s = float(np.interp(t1, self._t, self._s))
        return s, float(np.interp(t1, self._t, self._v)), float(np.interp(t1, self._t, self._a))

    def max_dynamics(self, dt: float = 0.02, duration_s: float = 180.0) -> dict:
        from scipy.spatial.transform import Rotation
        ts = np.arange(0.0, duration_s, dt)
        acc = np.array([np.linalg.norm(self.sample(t).acceleration) for t in ts])
        rates = np.array([np.linalg.norm(self.body_rates(t)) for t in ts[1:]])
        tilt = np.array([np.degrees(np.arccos(np.clip(
            Rotation.from_quat(self.sample(t).quat_xyzw).apply([0, 0, 1])[2], -1, 1))) for t in ts])
        speed = np.array([self.arc(t)[1] for t in ts])
        return {"max_acc": float(acc.max()), "max_rate": float(rates.max()), "max_tilt_deg": float(tilt.max()),
                "mean_speed": float(speed[ts > self.hover_s + 5].mean()), "max_speed": float(speed.max()),
                "distance_m": float(self.arc(duration_s - 1e-6)[0])}
