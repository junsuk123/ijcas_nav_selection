"""Reference trajectories for the forced-motion Stage A (pure numpy/scipy, no Isaac imports).

A path is a closed loop of ENU waypoints, interpolated by a periodic quintic B-spline and
re-parameterised by arc length (a second periodic quintic through dense arc-length samples), so
position is C4 in arc length.  Arc length follows hover -> cosine speed ramp -> constant cruise
speed.  Velocity and acceleration are analytic (chain rule), attitude follows multirotor
differential flatness (body z = thrust direction a + g e_z, yaw = path heading or fixed), and body
rates are central differences of that smooth attitude.  The accelerometer therefore sees what a
real quadrotor would, without the spikes a piecewise-linear path produces at its vertices
(results/g1/imu_probe, first attempt: 28 rad/s roll-rate spikes).

    traj = LoopTrajectory(waypoints, speed=3.0, hover_s=4.0, ramp_s=4.0)
    s = traj.sample(t)  # position, velocity, acceleration, quaternion xyzw, yaw
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665


@dataclass
class TrajectorySample:
    t: float
    position: np.ndarray        # ENU, m
    velocity: np.ndarray        # ENU, m/s
    acceleration: np.ndarray    # ENU, m/s^2
    quat_xyzw: np.ndarray       # body FLU -> ENU
    yaw: float


def _rounded_polyline(waypoints: np.ndarray, radius: float, spacing: float = 0.25) -> np.ndarray:
    """Closed polyline densified to ``spacing`` and Gaussian-smoothed along arc length.

    A sparse waypoint set interpolated directly by a quintic overshoots between waypoints (it
    left the warehouse aisle); straight legs stay straight after this, corners become ~radius
    arcs pulled slightly inside the polyline corner.
    """
    closed = np.vstack([waypoints, waypoints[:1]])
    pts = []
    for a, b in zip(closed[:-1], closed[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        pts.append(a + (b - a) * np.linspace(0.0, 1.0, n, endpoint=False)[:, None])
    dense = np.vstack(pts)
    sigma = max(1.0, radius / spacing / 2.0)
    half = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.vstack([dense[-half:], dense, dense[:half]])
    smooth = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)])
    return smooth[::4]


def _periodic_spline(x: np.ndarray, y: np.ndarray, k: int = 5):
    return make_interp_spline(x, y, k=k, bc_type="periodic")


class LoopTrajectory:
    def __init__(self, waypoints, speed: float = 3.0, hover_s: float = 4.0, ramp_s: float = 4.0,
                 yaw_mode: str = "heading", fixed_yaw: float = 0.0, start_index: int = 0,
                 arc_sample_m: float = 0.25, corner_radius_m: float = 0.0):
        wp = np.roll(np.asarray(waypoints, float), -start_index, axis=0)
        if corner_radius_m > 0.0:
            wp = _rounded_polyline(wp, corner_radius_m)
        closed = np.vstack([wp, wp[:1]])
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
        by_chord = _periodic_spline(chord, closed)
        u = np.linspace(0.0, chord[-1], max(2000, int(chord[-1] / 0.01)))
        speed_u = np.linalg.norm(by_chord.derivative(1)(u), axis=1)
        s_of_u = np.concatenate([[0.0], np.cumsum(0.5 * (speed_u[1:] + speed_u[:-1]) * np.diff(u))])
        self.length = float(s_of_u[-1])
        n = max(16, int(self.length / arc_sample_m))
        s_knots = np.linspace(0.0, self.length, n + 1)
        u_knots = np.interp(s_knots, s_of_u, u)
        pts = by_chord(u_knots)
        pts[-1] = pts[0]
        self._path = _periodic_spline(s_knots, pts)
        self._d1 = self._path.derivative(1)
        self._d2 = self._path.derivative(2)
        self.speed = float(speed)
        self.hover_s = float(hover_s)
        self.ramp_s = float(ramp_s)
        self.yaw_mode = yaw_mode
        self.fixed_yaw = float(fixed_yaw)

    # --- arc length along time: hover, cosine ramp, cruise -----------------------------------
    def arc(self, t: float) -> tuple[float, float, float]:
        """(s, ds/dt, d2s/dt2) at time t."""
        t1 = t - self.hover_s
        v, r = self.speed, self.ramp_s
        if t1 <= 0.0:
            return 0.0, 0.0, 0.0
        if t1 < r:
            s = v * 0.5 * (t1 - r / np.pi * np.sin(np.pi * t1 / r))
            sd = v * 0.5 * (1.0 - np.cos(np.pi * t1 / r))
            sdd = v * 0.5 * np.pi / r * np.sin(np.pi * t1 / r)
            return s, sd, sdd
        return v * 0.5 * r + v * (t1 - r), v, 0.0

    def sample(self, t: float) -> TrajectorySample:
        s, sd, sdd = self.arc(t)
        s_mod = s % self.length
        p = self._path(s_mod)
        dp = self._d1(s_mod)
        ddp = self._d2(s_mod)
        vel = dp * sd
        acc = ddp * sd * sd + dp * sdd
        yaw = self.fixed_yaw if self.yaw_mode == "fixed" else float(np.arctan2(dp[1], dp[0]))
        return TrajectorySample(t, p, vel, acc, self.attitude(acc, yaw), yaw)

    def table(self, t0: float, n: int, dt: float) -> dict:
        """Vectorised samples at t0 + k dt, k = 0..n-1 (position, velocity, acceleration,
        quaternion xyzw, body rates) for the per-physics-step lookup in ForcedMultirotor."""
        ts = t0 + dt * np.arange(n + 1)
        arcs = np.array([self.arc(t) for t in ts])
        s_mod = arcs[:, 0] % self.length
        p, dp, ddp = self._path(s_mod), self._d1(s_mod), self._d2(s_mod)
        sd, sdd = arcs[:, 1:2], arcs[:, 2:3]
        vel = dp * sd
        acc = ddp * sd * sd + dp * sdd
        yaw = (np.full(len(ts), self.fixed_yaw) if self.yaw_mode == "fixed"
               else np.arctan2(dp[:, 1], dp[:, 0]))
        z_b = acc + np.array([0.0, 0.0, GRAVITY])
        z_b /= np.linalg.norm(z_b, axis=1, keepdims=True)
        x_c = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(ts))])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b, axis=1, keepdims=True)
        x_b = np.cross(y_b, z_b)
        rot = Rotation.from_matrix(np.stack([x_b, y_b, z_b], axis=2))
        rates = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt      # forward difference
        rates = np.vstack([rates[:1], 0.5 * (rates[1:] + rates[:-1])])  # centred on samples
        return {"t": ts[:-1], "position": p[:-1], "velocity": vel[:-1],
                "acceleration": acc[:-1], "quat_xyzw": rot.as_quat()[:-1], "body_rates": rates}

    def body_rates(self, t: float, h: float = 1e-3) -> np.ndarray:
        r0 = Rotation.from_quat(self.sample(t - h).quat_xyzw)
        r1 = Rotation.from_quat(self.sample(t + h).quat_xyzw)
        return (r0.inv() * r1).as_rotvec() / (2 * h)

    @staticmethod
    def attitude(acc_enu: np.ndarray, yaw: float) -> np.ndarray:
        z_b = np.asarray(acc_enu, float) + np.array([0.0, 0.0, GRAVITY])
        z_b /= np.linalg.norm(z_b)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        return Rotation.from_matrix(np.column_stack([x_b, y_b, z_b])).as_quat()

    def max_dynamics(self, dt: float = 0.01) -> dict:
        ts = np.arange(0.0, self.hover_s + self.ramp_s + self.length / max(self.speed, 1e-6), dt)
        acc = np.array([np.linalg.norm(self.sample(t).acceleration) for t in ts])
        rates = np.array([np.linalg.norm(self.body_rates(t)) for t in ts])
        tilt = np.array([np.degrees(np.arccos(np.clip(
            Rotation.from_quat(self.sample(t).quat_xyzw).apply([0, 0, 1])[2], -1, 1))) for t in ts])
        return {"max_acc": float(acc.max()), "max_rate": float(rates.max()),
                "max_tilt_deg": float(tilt.max()), "loop_length_m": self.length}
