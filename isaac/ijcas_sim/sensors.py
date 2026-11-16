"""IJCAS sensor models plugged into Pegasus (one measurement feeds PX4 and the ROS engines).

Why not Pegasus' stock sensors as-is (results/g0/px4_probe, results/g1/imu_probe):
* Pegasus' ``update_at_rate`` compares the accumulated float32 physics dt (0.004999999888 s) with
  1/rate; at rate == physics rate it fires every *second* step, so a "200 Hz" IMU ran at 100 Hz and
  PX4's lockstep clock ran at half speed (EKF2 heading drifted 20 deg per turn).  ``RateGate``
  keeps a relative tolerance.
* the stock IMU draws from the global RNG, never adds the accelerometer bias it simulates, and
  reports the true attitude; the stock GNSS has no velocity noise and sends eph/epv in the wrong
  unit (HIL_GPS expects cm).

All models use a per-run ``numpy.random.Generator`` and consult the ``FaultSchedule``.  Every
output also keeps the FLU/ENU values and the state time so the ROS publisher sends exactly what
PX4 received.  Isaac-only module (imports Pegasus).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pegasus.simulator.logic.sensors.barometer import Barometer
from pegasus.simulator.logic.sensors.geo_mag_utils import reprojection
from pegasus.simulator.logic.sensors.magnetometer import Magnetometer
from pegasus.simulator.logic.sensors.sensor import Sensor
from scipy.spatial.transform import Rotation

from .faults import FaultSchedule

GRAVITY = 9.80665
FLU_TO_FRD = np.array([1.0, -1.0, -1.0])


class RateGate:
    """Fires at ``rate_hz`` of simulation time; robust to float32 physics steps."""

    def __init__(self, rate_hz: float, tol: float = 1e-4):
        self.period = 1.0 / rate_hz
        self.tol = tol
        self.acc = 0.0
        self.first = True

    def __call__(self, dt: float) -> tuple[bool, float]:
        self.acc += dt
        if self.first or self.acc >= self.period * (1.0 - self.tol):
            elapsed = self.acc
            self.acc = 0.0 if self.first else max(0.0, self.acc - self.period)
            self.first = False
            return True, elapsed
        return False, 0.0


@dataclass
class ImuNoise:
    """ADIS16448-class MEMS IMU (the Pegasus/RotorS default numbers)."""
    gyro_noise_density: float = 1.8665e-4    # rad/s/sqrt(Hz)
    gyro_random_walk: float = 3.8785e-5      # rad/s^2/sqrt(Hz)
    gyro_bias_tau_s: float = 1000.0
    accel_noise_density: float = 1.86e-3     # m/s^2/sqrt(Hz)
    accel_random_walk: float = 6.0e-3        # m/s^3/sqrt(Hz)
    accel_bias_tau_s: float = 300.0
    gyro_range_rad_s: float = 2000.0 * math.pi / 180.0
    accel_range_mps2: float = 16.0 * GRAVITY


def _gauss_markov(bias, tau, walk, dt, rng):
    phi = math.exp(-dt / tau)
    sigma = math.sqrt(walk * walk * tau / 2.0 * (1.0 - math.exp(-2.0 * dt / tau)))
    return phi * bias + sigma * rng.standard_normal(3)


class IjcasImu(Sensor):
    """Seeded IMU; true specific force from the analytic state acceleration."""

    def __init__(self, rate_hz: float, rng: np.random.Generator, faults: FaultSchedule,
                 clock, noise: ImuNoise | None = None):
        super().__init__(sensor_type="IMU", update_rate=rate_hz)
        self.gate = RateGate(rate_hz)
        self.rng = rng
        self.faults = faults
        self.clock = clock               # callable -> simulation time of the current state
        self.noise = noise or ImuNoise()
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.last = None                 # dict with t, gyro_flu, accel_flu (for ROS)
        self.count = 0

    def update(self, state, dt: float):
        fire, elapsed = self.gate(dt)
        if not fire:
            return None
        n = self.noise
        t = float(self.clock())
        h = max(elapsed, 1e-6)
        self.gyro_bias = _gauss_markov(self.gyro_bias, n.gyro_bias_tau_s, n.gyro_random_walk, h,
                                       self.rng)
        self.accel_bias = _gauss_markov(self.accel_bias, n.accel_bias_tau_s, n.accel_random_walk,
                                        h, self.rng)
        r = Rotation.from_quat(state.attitude)
        specific_force = r.inv().apply(np.asarray(state.linear_acceleration, float)
                                       + np.array([0.0, 0.0, GRAVITY]))
        gyro = (np.asarray(state.angular_velocity, float) + self.gyro_bias
                + n.gyro_noise_density / math.sqrt(h) * self.rng.standard_normal(3))
        accel = (specific_force + self.accel_bias
                 + n.accel_noise_density / math.sqrt(h) * self.rng.standard_normal(3))
        g_range, a_range = n.gyro_range_rad_s, n.accel_range_mps2
        for e in self.faults.active("imu", t):
            if e.kind == "gyro_bias_step":
                gyro = gyro + np.asarray(e.params["rad_s"], float)
            elif e.kind == "accel_bias_step":
                accel = accel + np.asarray(e.params["mps2"], float)
            elif e.kind == "saturation":
                g_range = min(g_range, float(e.params["gyro_rad_s"]))
                a_range = min(a_range, float(e.params["accel_mps2"]))
        gyro = np.clip(gyro, -g_range, g_range)
        accel = np.clip(accel, -a_range, a_range)
        self.count += 1
        self.last = {"t": t, "gyro_flu": gyro, "accel_flu": accel}
        self._state = {"angular_velocity": gyro * FLU_TO_FRD,
                       "linear_acceleration": accel * FLU_TO_FRD}
        return self._state


@dataclass
class GnssNoise:
    horizontal_sigma_m: float = 0.3      # white
    vertical_sigma_m: float = 0.5
    bias_sigma_m: float = 0.4            # Gauss-Markov (multipath-like slow error)
    bias_tau_s: float = 60.0
    velocity_sigma_mps: float = 0.05
    reported_eph_m: float = 0.6
    reported_epv_m: float = 1.0
    satellites: int = 12


class IjcasGnss(Sensor):
    """Seeded GNSS with velocity noise, honest accuracy fields and outage/bias faults."""

    def __init__(self, rate_hz: float, rng: np.random.Generator, faults: FaultSchedule, clock,
                 noise: GnssNoise | None = None):
        super().__init__(sensor_type="GPS", update_rate=rate_hz)
        self.gate = RateGate(rate_hz)
        self.rng = rng
        self.faults = faults
        self.clock = clock
        self.noise = noise or GnssNoise()
        self.bias = np.zeros(3)
        self.last = None
        self.count = 0

    def update(self, state, dt: float):
        fire, elapsed = self.gate(dt)
        if not fire:
            return None
        n = self.noise
        t = float(self.clock())
        h = max(elapsed, 1e-6)
        walk = n.bias_sigma_m * math.sqrt(2.0 / n.bias_tau_s)
        self.bias = _gauss_markov(self.bias, n.bias_tau_s, walk, h, self.rng)
        pos = (np.asarray(state.position, float) + self.bias * np.array([1.0, 1.0, 1.5])
               + self.rng.standard_normal(3) * np.array([n.horizontal_sigma_m,
                                                         n.horizontal_sigma_m,
                                                         n.vertical_sigma_m]))
        vel = np.asarray(state.linear_velocity, float) + n.velocity_sigma_mps * \
            self.rng.standard_normal(3)
        fix_type, sats = 3, n.satellites
        eph, epv = n.reported_eph_m, n.reported_epv_m
        for e in self.faults.active("gnss", t):
            if e.kind == "outage":
                fix_type, sats = 0, 0
            elif e.kind == "position_bias":
                pos = pos + e.ramp(t) * np.asarray(e.params["vector_m"], float)
            elif e.kind == "velocity_invalid":
                vel = (vel + np.asarray(e.params["bias_mps"], float)
                       + float(e.params["sigma_mps"]) * self.rng.standard_normal(3))
        lat, lon = reprojection(pos, math.radians(self._origin_lat),
                                math.radians(self._origin_lon))
        lat_gt, lon_gt = reprojection(np.asarray(state.position, float),
                                      math.radians(self._origin_lat),
                                      math.radians(self._origin_lon))
        speed = float(np.linalg.norm(vel[:2]))
        cog = (math.degrees(math.atan2(vel[0], vel[1])) + 360.0) % 360.0
        self.count += 1
        self.last = {"t": t, "fix_type": fix_type, "lat_deg": math.degrees(lat),
                     "lon_deg": math.degrees(lon), "alt_m": float(pos[2] + self._origin_alt),
                     "vel_enu": vel, "eph_m": eph, "epv_m": epv, "satellites": sats}
        self._state = {
            "latitude": math.degrees(lat), "longitude": math.degrees(lon),
            "altitude": float(pos[2] + self._origin_alt),
            "eph": eph * 100.0, "epv": epv * 100.0,   # HIL_GPS: cm
            "speed": speed, "velocity_north": float(vel[1]), "velocity_east": float(vel[0]),
            "velocity_down": float(-vel[2]), "cog": cog, "fix_type": fix_type,
            "sattelites_visible": sats,
            "latitude_gt": math.degrees(lat_gt), "longitude_gt": math.degrees(lon_gt),
            "altitude_gt": float(state.position[2] + self._origin_alt),
        }
        return self._state


class GatedBarometer(Barometer):
    """Pegasus barometer with a robust rate gate and loss/bias faults."""

    def __init__(self, rate_hz: float, faults: FaultSchedule, clock):
        super().__init__({"update_rate": rate_hz})
        self.gate = RateGate(rate_hz)
        self.faults = faults
        self.clock = clock
        self.last = None

    def update(self, state, dt: float):
        fire, elapsed = self.gate(dt)
        if not fire:
            return None
        t = float(self.clock())
        if self.faults.first("baro", "loss", t):
            return None
        self._first_update = True        # let the stock decorator compute this sample
        self._total_time = 0.0
        out = dict(Barometer.update(self, state, elapsed))
        e = self.faults.first("baro", "bias", t)
        if e is not None:
            out["pressure_altitude"] = out["pressure_altitude"] + float(e.params["bias_m"])
            out["absolute_pressure"] = out["absolute_pressure"] - 0.12 * float(e.params["bias_m"])
        self.last = {"t": t, **out}
        self._state = out
        return out


class GatedMagnetometer(Magnetometer):
    def __init__(self, rate_hz: float, faults: FaultSchedule, clock):
        super().__init__({"update_rate": rate_hz})
        self.gate = RateGate(rate_hz)
        self.faults = faults
        self.clock = clock
        self.last = None

    def update(self, state, dt: float):
        fire, elapsed = self.gate(dt)
        if not fire:
            return None
        t = float(self.clock())
        if self.faults.first("mag", "loss", t):
            return None
        self._first_update = True
        self._total_time = 0.0
        out = dict(Magnetometer.update(self, state, elapsed))
        e = self.faults.first("mag", "bias", t)
        if e is not None:
            out["magnetic_field"] = list(np.asarray(out["magnetic_field"], float)
                                         + np.asarray(e.params["bias_gauss"], float))
        self.last = {"t": t, **out}
        self._state = out
        return out
