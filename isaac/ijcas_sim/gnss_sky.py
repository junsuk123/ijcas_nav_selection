"""Simulated GNSS receiver for the outdoor scenes: satellites, signal paths and the receiver's own solution.

What a real receiver experiences is generated here, in the simulator, from the scene geometry:

Satellites (per run, seeded)
  constellations   GPS, GPS+Galileo or GPS+Galileo+BeiDou-MEO (receiver type);
  orbits           nominal Walker slots (GPS 24/6, Galileo 24/3, BeiDou 24/3) plus extra satellites in
                   random slots (GPS 0-7, the others 0-3), some satellites unhealthy (0-2, never used);
                   Keplerian circular orbits propagated at every epoch with Earth rotation, epoch of day
                   drawn per run -> number of satellites above the mask and their geometry (DOP) vary.
Signal paths (per epoch and satellite, PhysX ray casts against the scene colliders)
  line of sight    ray from the antenna to the satellite;
  reflection       specular single bounce off a nearby plane (facade, ground, panel, tunnel wall): the
                   antenna image in the plane gives the excess path  delta = 2 D (n . u); the specular point
                   must lie on the hit surface and must see the satellite itself;
  signal strength  C/N0 = antenna pattern (47 dB-Hz at zenith, ~38 at 10 deg) + per-run receiver offset,
                   minus the reflection loss (facade 8 dB, ground 12 dB, incl. polarisation reversal);
  tracking         a satellite is tracked while C/N0 >= 30 dB-Hz; after losing it, it must exceed the
                   33 dB-Hz acquisition threshold and is used only after a re-acquisition delay
                   (< 1 s gap: immediate, < 30 s: 1 s, longer, e.g. a tunnel: 4 s);
  LOS + reflection code multipath of a narrow-correlator DLL (coherent envelope, carrier phase from the
                   excess path, amplitude from the reflection loss);
  reflection only  NLOS: the receiver measures the reflected path (error = delta) if it is strong enough.
Range errors       C/N0-dependent code noise, ionospheric residual (activity drawn per run, obliquity
                   mapped, Gauss-Markov), tropospheric residual, broadcast orbit / clock error per satellite.
Receiver solution  weighted least squares (weights from C/N0, as receivers do) linearised at the true
                   state, so DOP, satellite count, multipath and NLOS act through the actual geometry;
                   >= 4 satellites and PDOP <= 20 for a fix.  Reported: satellites used, eph / epv from the
                   formal covariance (the receiver does not know about multipath / NLOS), HDOP / VDOP / PDOP.

Pure numpy except ``physx_raycast``; tests/test_gnss_sky.py exercises it with analytic box worlds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MU = 3.986004418e14            # m^3/s^2
OMEGA_E = 7.2921151467e-5      # rad/s
WGS_A, WGS_F = 6378137.0, 1.0 / 298.257223563
CHIP_CA = 293.052              # m, GPS L1 C/A chip length (also used for E1 / B1C, simplification)
LAMBDA_L1 = 0.190293672798     # m
SIDEREAL_DAY = 86164.0905

# nominal Walker constellations and the number of extra (spare) satellites a run may have
CONSTELLATIONS = {
    "gps": {"planes": 6, "per_plane": 4, "inc_deg": 55.0, "a_m": 26_559_700.0, "phase_deg": 15.0,
            "raan0_deg": 0.0, "extra_max": 7},
    "galileo": {"planes": 3, "per_plane": 8, "inc_deg": 56.0, "a_m": 29_600_000.0, "phase_deg": 15.0,
                "raan0_deg": 20.0, "extra_max": 3},
    "beidou": {"planes": 3, "per_plane": 8, "inc_deg": 55.0, "a_m": 27_906_100.0, "phase_deg": 15.0,
               "raan0_deg": 40.0, "extra_max": 3},
}
CONSTELLATION_SETS = (("gps",), ("gps", "galileo"), ("gps", "galileo", "beidou"))


@dataclass
class GnssErrorModel:
    elevation_mask_deg: float = 10.0
    cn0_zenith_dbhz: float = 47.0         # antenna pattern: cn0 = zenith - drop * (1 - sin el)^1.5
    cn0_drop_db: float = 12.0
    receiver_offset_db: float = 2.0       # per-run receiver / antenna quality, uniform +-
    loss_facade_db: float = 8.0           # one reflection incl. RHCP -> LHCP
    loss_ground_db: float = 12.0          # + antenna gain towards the ground
    track_dbhz: float = 30.0
    acquire_dbhz: float = 33.0
    reacq_short_s: float = 1.0            # gap 1-30 s
    reacq_long_s: float = 4.0             # gap > 30 s (e.g. a tunnel)
    code_sigma_45_m: float = 0.3          # code noise at 45 dB-Hz, scales with 10^((45 - cn0) / 20)
    rate_sigma_45_mps: float = 0.03
    iono_zenith_m: tuple = (0.3, 1.5)     # residual after the broadcast model, drawn per run
    iono_tau_s: float = 600.0
    tropo_zenith_m: float = 0.1
    ephem_sigma_m: float = 0.3            # broadcast orbit + clock, constant per satellite
    unhealthy_max: int = 2
    correlator_spacing_chip: float = 0.1
    reflect_max_m: float = 80.0
    facade_probe_deg: tuple = (-30.0, 0.0, 30.0)   # around the mirrored satellite azimuth
    antenna_up_m: float = 0.3
    ray_skip_m: float = 0.5               # start rays outside the vehicle's own colliders
    ray_max_m: float = 2000.0
    receiver_uere_m: float = 1.0          # what the receiver assumes besides code noise (eph / epv)
    max_pdop: float = 20.0


def geodetic_to_ecef(lat, lon, h=0.0):
    e2 = WGS_F * (2 - WGS_F)
    n = WGS_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])


def ecef_to_enu_matrix(lat, lon):
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def satellite_catalogue(names, rng=None):
    """Orbital elements of the satellites of ``names``: nominal slots, plus random extra satellites when a
    generator is given.  -> dict of arrays a, inc, raan, u0 and ids."""
    a, inc, raan, u0, ids = [], [], [], [], []
    for name in names:
        c = CONSTELLATIONS[name]
        slots = [(p, np.radians(360.0 * k / c["per_plane"] + c["phase_deg"] * p))
                 for p in range(c["planes"]) for k in range(c["per_plane"])]
        if rng is not None:
            for _ in range(int(rng.integers(0, c["extra_max"] + 1))):
                slots.append((int(rng.integers(0, c["planes"])), float(rng.uniform(0, 2 * np.pi))))
        for j, (p, u) in enumerate(slots):
            a.append(c["a_m"])
            inc.append(np.radians(c["inc_deg"]))
            raan.append(np.radians(c["raan0_deg"] + 360.0 * p / c["planes"]))
            u0.append(u)
            ids.append(f"{name[0].upper()}{j:02d}")
    return {"a": np.asarray(a), "inc": np.asarray(inc), "raan": np.asarray(raan), "u0": np.asarray(u0),
            "ids": ids}


def propagate_ecef(cat, t_s):
    """ECEF positions [N,3] of the catalogue at time t_s (s of the sidereal day)."""
    n = np.sqrt(MU / cat["a"] ** 3)
    u = cat["u0"] + n * t_s
    cr, sr, ci, si = np.cos(cat["raan"]), np.sin(cat["raan"]), np.cos(cat["inc"]), np.sin(cat["inc"])
    x = cat["a"][:, None] * np.stack([np.cos(u) * cr - np.sin(u) * ci * sr,
                                      np.cos(u) * sr + np.sin(u) * ci * cr,
                                      np.sin(u) * si], 1)
    th = OMEGA_E * t_s
    rz = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
    return x @ rz.T


def constellation_ecef(names, t_s):
    """Nominal slots only (no extra satellites): ECEF positions [N,3] and ids."""
    cat = satellite_catalogue(names)
    return propagate_ecef(cat, t_s), cat["ids"]


def dll_multipath_error(delta, alpha, phase, d_chip=0.1, chip=CHIP_CA):
    """Code tracking error [m] of a coherent early-late DLL with spacing d for one reflection of relative
    amplitude alpha, excess path delta [m] and relative carrier phase ``phase`` (classic envelope)."""
    half = d_chip * chip / 2.0
    if delta <= half:
        num = alpha * delta * np.cos(phase)
    elif delta <= chip * (1 - d_chip / 2):
        num = alpha * half * np.cos(phase)
    elif delta <= chip * (1 + d_chip / 2):
        num = alpha * half * np.cos(phase) * (chip * (1 + d_chip / 2) - delta) / (d_chip * chip)
    else:
        return 0.0
    return float(num / (1.0 + alpha * np.cos(phase)))


def physx_raycast(origin, direction, max_dist):
    """(hit, distance, point, normal) from the Isaac PhysX scene query (closest hit)."""
    from omni.physx import get_physx_scene_query_interface
    r = get_physx_scene_query_interface().raycast_closest(tuple(float(v) for v in origin),
                                                           tuple(float(v) for v in direction),
                                                           float(max_dist))
    if not r["hit"]:
        return False, np.inf, None, None
    return True, float(r["distance"]), np.asarray(r["position"], float), np.asarray(r["normal"], float)


@dataclass
class SkyModel:
    seed: int
    lat_deg: float
    lon_deg: float
    model: GnssErrorModel = field(default_factory=GnssErrorModel)
    raycast: object = None                       # callable like physx_raycast (None -> PhysX)
    constellations: tuple | None = None          # None -> drawn from the seed (receiver type)
    epoch_s: float | None = None                 # None -> drawn from the seed
    extras: bool = True                          # spare satellites / unhealthy satellites from the seed

    def __post_init__(self):
        m = self.model
        rng = np.random.default_rng(int(self.seed) + 200_000)
        self.rng = rng
        if self.constellations is None:
            self.constellations = CONSTELLATION_SETS[int(rng.integers(0, len(CONSTELLATION_SETS)))]
        if self.epoch_s is None:
            self.epoch_s = float(rng.uniform(0.0, SIDEREAL_DAY))
        self.cat = satellite_catalogue(self.constellations, rng if self.extras else None)
        self.ids = self.cat["ids"]
        n = len(self.ids)
        self.healthy = np.ones(n, bool)
        if self.extras:
            k = int(rng.integers(0, m.unhealthy_max + 1))
            self.healthy[rng.choice(n, size=k, replace=False)] = False
        self.rx_offset_db = float(rng.uniform(-m.receiver_offset_db, m.receiver_offset_db))
        self.iono_zenith = float(rng.uniform(*m.iono_zenith_m))
        self.ephem = rng.standard_normal(n) * m.ephem_sigma_m
        self.iono = rng.standard_normal(n)                  # unit-variance Gauss-Markov state
        self.phase0 = rng.uniform(0, 2 * np.pi, n)
        self.tracking = np.zeros(n, bool)                   # receiver tracking state
        self.lost_at = np.full(n, -np.inf)                  # time the satellite was last lost
        self.ready_at = np.zeros(n)                         # earliest use after (re)acquisition
        self.started = False
        self._lat, self._lon = np.radians(self.lat_deg), np.radians(self.lon_deg)
        self._r0 = geodetic_to_ecef(self._lat, self._lon, 0.0)
        self._R = ecef_to_enu_matrix(self._lat, self._lon)
        self._ray = self.raycast or physx_raycast
        self._t_last = None

    # --- geometry and signal strength ----------------------------------------------------------
    def satellites(self, t):
        """Unit vectors (ENU) and elevations of all satellites at run time t."""
        d = (propagate_ecef(self.cat, self.epoch_s + t) - self._r0) @ self._R.T
        u = d / np.linalg.norm(d, axis=1, keepdims=True)
        return u, np.arcsin(np.clip(u[:, 2], -1, 1))

    def cn0_los(self, el):
        m = self.model
        return m.cn0_zenith_dbhz - m.cn0_drop_db * (1.0 - np.sin(max(el, 0.0))) ** 1.5 + self.rx_offset_db

    def _reflection(self, o, u):
        """Strongest valid specular reflection for unit vector u at antenna o: (delta, loss_db, normal)."""
        m = self.model
        az = np.arctan2(u[1], u[0]) + np.pi
        probes = [np.array([0.0, 0.0, -1.0])]
        probes += [np.array([np.cos(az + np.radians(a)), np.sin(az + np.radians(a)), 0.0]) for a in m.facade_probe_deg]
        best = None
        for d in probes:
            hit, dist, q, nrm = self._ray(o, d, m.reflect_max_m)
            if not hit or nrm is None:
                continue
            nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
            if np.dot(o - q, nrm) < 0:
                nrm = -nrm
            D = float(np.dot(o - q, nrm))
            un = float(np.dot(u, nrm))
            if D <= 0.05 or un <= 0.02:
                continue
            P = (o - 2.0 * D * nrm) + (D / un) * u               # specular point on the plane
            v = P - o
            L = float(np.linalg.norm(v))
            if L > m.reflect_max_m * 1.5:
                continue
            h2, d2, _, n2 = self._ray(o, v / L, L + 1.0)         # the plane really is there
            if not h2 or abs(d2 - L) > 0.3 or n2 is None or abs(np.dot(n2 / np.linalg.norm(n2), nrm)) < 0.95:
                continue
            h3, *_ = self._ray(P + 0.05 * nrm, u, m.ray_max_m)  # specular point sees the satellite
            if h3:
                continue
            loss = m.loss_ground_db if (nrm[2] > 0.7 and q[2] < o[2]) else m.loss_facade_db
            cand = (2.0 * D * un, loss, nrm)
            if best is None or cand[1] < best[1] or (cand[1] == best[1] and cand[0] < best[0]):
                best = cand
        return best

    # --- one epoch ------------------------------------------------------------------------------
    def assess(self, t, pos, vel):
        """Receiver solution errors at run time t for the true vehicle state (antenna on top)."""
        m = self.model
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        if dt > 0:
            phi = np.exp(-dt / m.iono_tau_s)
            self.iono = phi * self.iono + np.sqrt(1 - phi * phi) * self.rng.standard_normal(len(self.iono))
        u_all, el_all = self.satellites(t)
        o = np.asarray(pos, float) + np.array([0.0, 0.0, m.antenna_up_m])
        vel = np.asarray(vel, float)
        rows, err, rate_err, sig, cn0s = [], [], [], [], []
        n_los = n_nlos = n_mp = n_wait = 0
        mask = np.radians(m.elevation_mask_deg)
        for i, (u, el) in enumerate(zip(u_all, el_all)):
            if el < mask or not self.healthy[i]:
                self._lose(i, t)
                continue
            blocked, *_ = self._ray(o + m.ray_skip_m * u, u, m.ray_max_m)
            refl = self._reflection(o, u)
            c_los = self.cn0_los(el)
            if not blocked:
                cn0, kind = c_los, "los"
            elif refl is not None:
                cn0, kind = c_los - refl[1], "nlos"
            else:
                self._lose(i, t)
                continue
            if not self._track(i, t, cn0):
                continue
            if t < self.ready_at[i]:
                n_wait += 1
                continue
            if kind == "los":
                n_los += 1
                e, r_e = 0.0, 0.0
                if refl is not None:
                    n_mp += 1
                    delta, loss, _ = refl
                    alpha = 10.0 ** (-loss / 20.0)
                    phase = 2 * np.pi * delta / LAMBDA_L1 + self.phase0[i] + np.pi
                    e = dll_multipath_error(delta, alpha, phase, m.correlator_spacing_chip)
            else:
                n_nlos += 1
                delta, _, nrm = refl
                e = delta
                r_e = 2.0 * float(np.dot(vel, nrm)) * float(np.dot(nrm, u))     # d(delta)/dt
            s_el = max(np.sin(el), 0.1)
            scale = 10.0 ** ((45.0 - cn0) / 20.0)
            code_sigma = m.code_sigma_45_m * scale
            iono = self.iono_zenith / np.sqrt(1.0 - (0.94 * np.cos(el)) ** 2) * self.iono[i]
            tropo = m.tropo_zenith_m / s_el * 0.3 * self.iono[i]
            err.append(e + iono + tropo + self.ephem[i] + code_sigma * self.rng.standard_normal())
            rate_err.append(r_e + m.rate_sigma_45_mps * scale * self.rng.standard_normal())
            rows.append(np.r_[-u, 1.0])
            sig.append(np.hypot(code_sigma, m.receiver_uere_m))
            cn0s.append(cn0)
        self.started = True
        n_used = len(rows)
        out = {"t": float(t), "n_above_mask": int(((el_all >= mask) & self.healthy).sum()), "n_los": n_los,
               "n_nlos": n_nlos, "n_multipath": n_mp, "n_reacquiring": n_wait, "used": n_used, "fix": False,
               "pos_err": np.zeros(3), "vel_err": np.zeros(3), "pdop": np.inf, "hdop": np.inf, "vdop": np.inf,
               "eph": np.inf, "epv": np.inf, "cn0_mean": float(np.mean(cn0s)) if cn0s else np.nan}
        if n_used < 4:
            return out
        H = np.asarray(rows)
        W = np.diag(1.0 / np.asarray(sig) ** 2)
        try:
            G = np.linalg.inv(H.T @ W @ H)
            Gu = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            return out
        dx = G @ H.T @ W @ np.asarray(err)
        dv = G @ H.T @ W @ np.asarray(rate_err)
        pdop = float(np.sqrt(np.trace(Gu[:3, :3])))
        out.update(pos_err=dx[:3], vel_err=dv[:3], pdop=pdop, hdop=float(np.sqrt(Gu[0, 0] + Gu[1, 1])),
                   vdop=float(np.sqrt(Gu[2, 2])), eph=float(np.sqrt(G[0, 0] + G[1, 1])),
                   epv=float(np.sqrt(G[2, 2])), fix=bool(np.isfinite(pdop) and pdop <= m.max_pdop))
        return out

    # --- receiver tracking state -------------------------------------------------------------------
    def _lose(self, i, t):
        if self.tracking[i]:
            self.tracking[i] = False
            self.lost_at[i] = t

    def _track(self, i, t, cn0):
        """Tracking with hysteresis and re-acquisition delay; True while the satellite is tracked."""
        m = self.model
        if self.tracking[i]:
            if cn0 >= m.track_dbhz:
                return True
            self._lose(i, t)
            return False
        if cn0 < m.acquire_dbhz:
            return False
        self.tracking[i] = True
        if self.started:                         # satellites present at power-up are tracked already
            gap = t - self.lost_at[i]
            self.ready_at[i] = t + (0.0 if gap < 1.0 else m.reacq_short_s if gap < 30.0 else m.reacq_long_s)
        return True
