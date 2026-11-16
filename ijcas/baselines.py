"""Baseline candidates from the common measurements (Notion 02: GNSS 단독, INS-DR).

GNSS-only   position = the GNSS fix projected to the local tangent plane of the scene origin
            (inverse of the azimuthal-equidistant projection the rig's GNSS model uses),
            velocity = GNSS velocity; no fix -> no output.  No filtering, by design.
INS-DR      explicit strapdown mechanisation of the common IMU (quaternion attitude, gravity
            in ENU, trapezoidal velocity/position) started from an *anchor* state and never
            corrected afterwards.  Used (a) to measure free-inertial drift after aiding is lost
            and (b) as the "limited DR" output when no aided candidate is valid.  The anchor is
            supplied by the caller (e.g. the last valid aided estimate), never ground truth when it
            is used as an online candidate.
Pure numpy; usable online (streaming) and offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

EARTH_RADIUS = 6353000.0          # Pegasus geo_mag_utils.EARTH_RADIUS (same projection)
GRAVITY = np.array([0.0, 0.0, -9.80665])


def lla_to_enu(lat_deg, lon_deg, alt, origin: dict) -> np.ndarray:
    """Inverse of pegasus.simulator.logic.sensors.geo_mag_utils.reprojection."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    lat0, lon0 = np.radians(origin["lat"]), np.radians(origin["lon"])
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    cos_d_lon = np.cos(lon - lon0)
    arg = np.clip(np.sin(lat0) * sin_lat + np.cos(lat0) * cos_lat * cos_d_lon, -1.0, 1.0)
    c = np.arccos(arg)
    k = np.where(np.abs(c) > 1e-12, c / np.maximum(np.sin(c), 1e-15), 1.0)
    north = k * (np.cos(lat0) * sin_lat - np.sin(lat0) * cos_lat * cos_d_lon) * EARTH_RADIUS
    east = k * cos_lat * np.sin(lon - lon0) * EARTH_RADIUS
    return np.column_stack([east, north, np.asarray(alt, float) - origin["alt"]])


def gnss_only(fix: np.ndarray, gvel: np.ndarray, origin: dict) -> dict:
    """fix rows: stamp recv status lat lon alt cov_h cov_v (recorder layout)."""
    valid = (fix[:, 2] >= 0) & np.isfinite(fix[:, 3])
    pos = np.full((len(fix), 3), np.nan)
    if valid.any():
        pos[valid] = lla_to_enu(fix[valid, 3], fix[valid, 4], fix[valid, 5], origin)
    vel = np.full((len(fix), 3), np.nan)
    if len(gvel):
        for i, t in enumerate(fix[:, 0]):
            j = np.searchsorted(gvel[:, 0], t)
            if j < len(gvel) and abs(gvel[j, 0] - t) < 1e-3:
                vel[i] = gvel[j, 2:5]
    return {"stamp": fix[:, 0], "recv": fix[:, 1], "valid": valid, "pos": pos, "vel": vel,
            "sigma_h": np.sqrt(np.maximum(fix[:, 6], 0.0))}


@dataclass
class InsState:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    rot: R                     # body FLU -> ENU
    gyro_bias: np.ndarray
    accel_bias: np.ndarray


class InsDr:
    """Strapdown integration of IMU rows (stamp gx gy gz ax ay az, FLU)."""

    def __init__(self, anchor: InsState):
        self.s = InsState(anchor.t, anchor.pos.copy(), anchor.vel.copy(), anchor.rot,
                          anchor.gyro_bias.copy(), anchor.accel_bias.copy())
        self._prev = None            # previous IMU row (trapezoidal integration)

    def step(self, row: np.ndarray) -> InsState:
        t = float(row[0])
        dt = t - self.s.t
        if dt <= 0.0:
            return self.s
        prev = self._prev if self._prev is not None else row
        self._prev = np.asarray(row, float)
        w0, w1 = prev[1:4] - self.s.gyro_bias, row[1:4] - self.s.gyro_bias
        f0, f1 = prev[4:7] - self.s.accel_bias, row[4:7] - self.s.accel_bias
        rot_new = self.s.rot * R.from_rotvec(0.5 * (w0 + w1) * dt)
        acc_world = 0.5 * (self.s.rot.apply(f0) + rot_new.apply(f1)) + GRAVITY
        vel_new = self.s.vel + acc_world * dt
        self.s.pos = self.s.pos + 0.5 * (self.s.vel + vel_new) * dt
        self.s.vel, self.s.rot, self.s.t = vel_new, rot_new, t
        return self.s

    def run(self, imu: np.ndarray, t_end: float) -> np.ndarray:
        """rows (t, px, py, pz, qx, qy, qz, qw, vx, vy, vz) at the IMU rate until t_end."""
        out = []
        for row in imu[(imu[:, 0] > self.s.t) & (imu[:, 0] <= t_end)]:
            s = self.step(row)
            out.append([s.t, *s.pos, *s.rot.as_quat(), *s.vel])
        return np.asarray(out)
