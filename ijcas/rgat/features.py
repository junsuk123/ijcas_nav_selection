"""Facts for the graph selector: node features, ontology fact states and selector-independent
candidate states (design 2.1, 3.1, 5.1; ontology doc 'fact generation').

Every feature is an observation available online (FactStream refuses ground-truth topics) and
carries its source stamp, so the model gets value y, presence m and age dt = t - source stamp.
A missing diagnostic is marked missing (m = 0), never filled with a 'normal' value, and a node
only gets the diagnostics its source really provides (PX4 odometry carries no covariance,
so none is invented; OpenVINS and FAST-LIO2 do).  Each sensor / estimator node also gets the 4-level fact state
(Usable / Limited / Unusable / Unknown) of its ontology capability, evaluated by the frozen
first-implementation indicators (config/selector/thresholds_validated.yaml).

``CandidateBank`` keeps every candidate's global output independent of any selection:
  gnss_ins  PX4 EKF2 output as is
  vio, lio  OpenVINS / FAST-LIO2 causally aligned (4-DOF) to PX4's frame while GNSS position is
            Usable (same alignment as the fixed comparators, ijcas.handover)
  ins       free-inertial strapdown anchored on PX4's state while GNSS position is Usable and
            propagated from the IMU otherwise; valid for max_dr_s after the last anchor
Eligibility (policy, not learned): output age <= 0.3 s, alignment established (vio, lio), and the
last external aiding input not older than AID_LIMIT_S.  Since 2026-11-18 (user decision) every
candidate also requires IMU reception (>= 150 Hz over the last 1 s, ijcas.selector.eligibility):
the IMU is the essential input of all four (ontology :requiresEssential :Inertial).  On the existing
recordings this changed no eligibility in the evaluation windows (validation: no tick at all).
Output age and the time since the last external aiding input (GNSS fix / tracked visual features /
usable LiDAR scan) are kept apart: an IMU-propagated output can be fresh while aiding stopped.
Aiding *reception* is a proxy, not proof that the engine fused it (ontology doc).
"""
from __future__ import annotations

import numpy as np

from ..selector.common import (LIMITED, UNKNOWN, UNUSABLE, USABLE, indicator_state, signals,
                               worst)
from ..selector.engine import DrAnchor, SelectionEngine, _NullReasoner
from .graph import CANDIDATES, NODES

STATES = (USABLE, LIMITED, UNUSABLE, UNKNOWN)
MAX_OUTPUT_AGE_S = 0.3            # ontology :maxOutputAgeS of the global waypoint task
# Eligibility policy "time since the last valid external aiding input" (ontology doc: output age
# and last correction are managed separately).  GNSS/INS: :px4_global :deadReckonLimitS 5.0;
# VIO: tracked visual features lost >= 1 s (the G5 confirmed-loss window); LIO: no usable scan for
# longer than the LiDAR-age invalid limit (1.5 s).  Reception is a proxy for fusion (stated).
AID_LIMIT_S = {"gnss_ins": 5.0, "vio": 1.0, "lio": 1.5}
NAV_SRC = {"gnss_ins": "px4", "vio": "openvins", "lio": "fastlio"}


def _disagreement_names():
    out = []
    for c in CANDIDATES:
        out.append([f"dis.{o}.{q}" for o in CANDIDATES if o != c for q in ("pos", "vel")])
    return dict(zip(CANDIDATES, out))


DIS = _disagreement_names()
ALIGN = ["align.valid", "align.age_log", "align.residual_m", "align.path_since_m",
         "align.vel_disagree_mps", "reset.age_log"]


def _states(cap):
    return [f"state.{cap}.{s}" for s in STATES]


# feature names per node (order = array layout)
SPEC = {
    "E_gnss": ["fix.status_ok", "fix.sigma_h_log", "fix.sigma_v_log"],
    "E_vis": ["vis.brightness", "vis.contrast"],
    "E_geo": ["lidar.lambda_ratio", "lidar.lambda_min", "lidar.range_median_log",
              "lidar.range_p90_log", "lidar.vertical_frac", "lidar.ground_frac"],
    "E_mot": ["imu.gyro_mean", "imu.gyro_max", "imu.spec_force_dev", "imu.acc_std"],
    "S_gnss": ["gnss.fix", "gnss.pos_vel_residual_m", "gnss.sigma_h"]
              + _states("GnssPosition") + _states("GnssVelocity"),
    "S_imu": ["imu.rate_hz", "imu.non_monotonic", "imu.stamp_lead_s", "imu.saturation_frac",
              "imu.max_abs_gyro", "imu.max_abs_accel", "aux.mag_rate_hz", "aux.baro_rate_hz"]
             + _states("Inertial") + _states("MagHeading") + _states("BaroHeight"),
    "S_cam": ["vision.image_age_s", "vision.features_log", "vision.grid_cells_ge10",
              "vision.stereo_matches_log"] + _states("VisualFeatures"),
    "S_lidar": ["lidar.scan_age_s", "lidar.points_log", "lidar.voxels_log"]
               + _states("LidarStructure"),
    "N_gi": ["out.valid", "out.age_s", "aid.age_log", "px4.pos_horiz_ratio", "px4.vel_ratio",
             "px4.pos_vert_ratio", "px4.mag_ratio", "px4.flag_pos_horiz_abs",
             "px4.pos_horiz_acc_log", "px4.pos_vert_acc_log", "consistency.gnss_local_z"]
            + DIS["gnss_ins"] + _states("PX4_EKF2"),
    "N_vio": ["out.valid", "out.age_s", "aid.age_log", "openvins.tracked_log", "ov.pos_var_log",
              "ov.ori_var_log"] + ALIGN + DIS["vio"] + _states("OpenVINS"),
    "N_lio": ["out.valid", "out.age_s", "aid.age_log", "lio.rot_var_log", "lio.pos_var_log"]
             + ALIGN + DIS["lio"],
    "N_ins": ["out.valid", "anchor.age_log"] + DIS["ins"],
}
assert set(SPEC) == set(NODES)
CAPABILITIES = ("GnssPosition", "GnssVelocity", "Inertial", "MagHeading", "BaroHeight",
                "VisualFeatures", "LidarStructure", "PX4_EKF2", "OpenVINS")


def _log(x, floor=1e-6):
    return float(np.log10(max(float(x), floor)))


class CandidateBank:
    """Selector-independent candidate outputs in PX4's local ENU frame + fact states."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.eng = SelectionEngine(_NullReasoner(), cfg, name="candidate_bank")
        self.dr = DrAnchor(cfg["selection"]["max_dr_s"])
        self.inds = {m: [i for i in cfg["indicators"] if m in i["assesses"]] for m in CAPABILITIES}
        self.last_reset = {c: -np.inf for c in ("openvins", "fastlio")}
        self._resets = {c: 0 for c in ("openvins", "fastlio")}
        ind = {i["id"]: i for i in cfg["indicators"]}
        self.ov_min = float(ind["ov_tracking"].get("invalid_below", 4))
        self.lidar_min = float(ind["lidar_points"].get("invalid_below", 1500))
        self.last_aid = {"gnss_ins": -np.inf, "vio": -np.inf, "lio": -np.inf}

    def _track_aiding(self, snap, sig):
        f = snap.fix
        if f is not None and f[2] >= 0 and np.isfinite(f[3]):
            self.last_aid["gnss_ins"] = max(self.last_aid["gnss_ins"], float(f[0]))
        if "openvins.tracked_features" in sig:
            n, st = sig["openvins.tracked_features"]
            if n >= self.ov_min:
                self.last_aid["vio"] = max(self.last_aid["vio"], float(st))
        li = snap.diag.get("lidar")
        if li and li[0].get("points") is not None and li[0]["points"] >= self.lidar_min:
            self.last_aid["lio"] = max(self.last_aid["lio"], float(li[0]["t"]))

    def aid_age(self, k, t):
        return t - self.last_aid[k]

    def step(self, snap):
        sig = signals(snap)
        sig.update(self.eng.derived_signals(snap))      # GNSS-local consistency, alignment stats
        self._track_aiding(snap, sig)
        imu_ok = self.eng.imu_received(snap)            # essential input of every candidate
        states = {m: worst(indicator_state(i, sig, snap.t)[0] for i in self.inds[m])
                  for m in CAPABILITIES}
        self.eng._update_alignment(snap, states)
        for c, al in self.eng.align.items():
            if al.resets != self._resets[c] or (al.invalidations and al.invalidations[-1][0] == snap.t):
                self.last_reset[c] = snap.t
            self._resets[c] = al.resets
        px4 = snap.outputs.get("px4")
        ref_ok = (px4 is not None and states["GnssPosition"] == USABLE
                  and states["PX4_EKF2"] == USABLE and snap.t - px4.stamp < MAX_OUTPUT_AGE_S)
        if ref_ok:
            self.dr.anchor(snap.t, px4.pos, px4.quat, px4.vel)
            ins = (px4.pos.copy(), px4.quat.copy(), px4.vel.copy())
        else:
            ins = self.dr.propagate(snap.imu_window if snap.imu_window is not None else [], snap.t)
        out = {}
        for k in CANDIDATES:
            if k == "ins":
                ok = imu_ok and ins is not None and self.dr.anchor_t is not None and \
                    self.dr.age(snap.t) <= self.dr.max_dr_s
                out[k] = ins if ok else None
                continue
            src = NAV_SRC[k]
            o = snap.outputs.get(src)
            g = self.eng.global_state(src, snap)
            out[k] = g if (imu_ok and o is not None and g is not None
                           and snap.t - o.stamp <= MAX_OUTPUT_AGE_S
                           and self.aid_age(k, snap.t) <= AID_LIMIT_S[k]) else None
        return out, states, sig


class FeatureExtractor:
    """One tick -> {node: (values, source stamps)} following SPEC (NaN = missing)."""

    def __init__(self, ov_cov: np.ndarray, cfg: dict, lio_odom: np.ndarray | None = None):
        self.ov_cov = ov_cov[np.argsort(ov_cov[:, 1])] if ov_cov.size else np.zeros((0, 8))
        self.ov_i = 0
        self.ov_last = None
        # FAST-LIO2 odometry covariance diagonal (recorder columns 12-17).  FAST-LIO2 writes its
        # rotation block first: diag[0:3] = rotation variance, diag[3:6] = position variance.
        lo = lio_odom if lio_odom is not None and lio_odom.size else np.zeros((0, 18))
        self.lio = lo[np.argsort(lo[:, 1])]
        self.lio_i = 0
        self.lio_last = None

    def __call__(self, snap, bank: CandidateBank, cands: dict, states: dict, sig: dict):
        t = snap.t
        f: dict[str, tuple[float, float]] = {}

        def put(name, value, stamp=t):
            if value is None:
                return
            v = float(value)
            if np.isfinite(v):
                f[name] = (v, float(stamp))

        # --- environment / motion -------------------------------------------------------
        if snap.fix is not None:
            st = snap.fix[0]
            ok = snap.fix[2] >= 0 and np.isfinite(snap.fix[3])
            put("fix.status_ok", 1.0 if ok else 0.0, st)
            if ok:
                put("fix.sigma_h_log", _log(np.sqrt(max(snap.fix[6], 0.0))), st)
                put("fix.sigma_v_log", _log(np.sqrt(max(snap.fix[7], 0.0))), st)
        v = snap.diag.get("vision")
        if v:
            p = v[0]
            put("vis.brightness", p.get("mean"), p["t"])
            put("vis.contrast", p.get("std"), p["t"])
            put("vision.image_age_s", t - p["t"], t)
            if p.get("features") is not None:
                put("vision.features_log", np.log1p(p["features"]), p["t"])
            put("vision.grid_cells_ge10", p.get("grid_cells_ge10"), p["t"])
            if p.get("stereo_matches") is not None:
                put("vision.stereo_matches_log", np.log1p(p["stereo_matches"]), p["t"])
        if "openvins.tracked_features" in sig:
            n, st = sig["openvins.tracked_features"]
            put("openvins.tracked_log", np.log1p(n), st)
        li = snap.diag.get("lidar")
        if li:
            p = li[0]
            put("lidar.scan_age_s", t - p["t"], t)
            if p.get("points") is not None:
                put("lidar.points_log", np.log1p(p["points"]), p["t"])
            if p.get("voxels") is not None:
                put("lidar.voxels_log", np.log1p(p["voxels"]), p["t"])
            put("lidar.lambda_ratio", p.get("lambda_ratio"), p["t"])
            if p.get("lambda"):
                put("lidar.lambda_min", min(p["lambda"]), p["t"])
            for k, n in (("range_median_m", "lidar.range_median_log"),
                         ("range_p90_m", "lidar.range_p90_log")):
                if p.get(k) is not None:
                    put(n, np.log1p(p[k]), p["t"])
            put("lidar.vertical_frac", p.get("vertical_structure_frac"), p["t"])
            put("lidar.ground_frac", p.get("ground_frac"), p["t"])
        w = snap.imu_window
        if w is not None and len(w):
            st = w[-1, 0]
            g = np.linalg.norm(w[:, 1:4], axis=1)
            a = np.linalg.norm(w[:, 4:7], axis=1)
            put("imu.gyro_mean", g.mean(), st)
            put("imu.gyro_max", g.max(), st)
            put("imu.spec_force_dev", abs(a.mean() - 9.80665), st)
            put("imu.acc_std", a.std(), st)
        # --- sensor quality ---------------------------------------------------------------
        for k in ("gnss.fix", "gnss.pos_vel_residual_m", "imu.rate_hz", "imu.non_monotonic",
                  "imu.stamp_lead_s", "imu.saturation_frac", "aux.mag_rate_hz",
                  "aux.baro_rate_hz", "px4.pos_horiz_ratio", "px4.vel_ratio",
                  "px4.flag_pos_horiz_abs", "consistency.gnss_local_z"):
            if k in sig:
                put(k, sig[k][0], sig[k][1])
        g = snap.diag.get("gnss")
        if g and g[0].get("sigma_h") is not None:
            put("gnss.sigma_h", g[0]["sigma_h"], g[0]["t"])
        im = snap.diag.get("imu")
        if im:
            p = im[0]
            put("imu.max_abs_gyro", p.get("max_abs_gyro"), p["t"])
            put("imu.max_abs_accel", p.get("max_abs_accel"), p["t"])
        if snap.px4_status:
            p = snap.px4_status[0]
            put("px4.pos_vert_ratio", p.get("pos_vert_ratio"), p["t"])
            put("px4.mag_ratio", p.get("mag_ratio"), p["t"])
            if p.get("pos_horiz_accuracy") is not None:
                put("px4.pos_horiz_acc_log", _log(p["pos_horiz_accuracy"]), p["t"])
            if p.get("pos_vert_accuracy") is not None:
                put("px4.pos_vert_acc_log", _log(p["pos_vert_accuracy"]), p["t"])
        # OpenVINS pose covariance (the only engine that publishes one)
        while self.ov_i < len(self.ov_cov) and self.ov_cov[self.ov_i, 1] <= t:
            self.ov_last = self.ov_cov[self.ov_i]
            self.ov_i += 1
        if self.ov_last is not None:
            put("ov.pos_var_log", _log(self.ov_last[2:5].sum(), 1e-9), self.ov_last[0])
            put("ov.ori_var_log", _log(self.ov_last[5:8].sum(), 1e-12), self.ov_last[0])
        while self.lio_i < len(self.lio) and self.lio[self.lio_i, 1] <= t:
            self.lio_last = self.lio[self.lio_i]
            self.lio_i += 1
        if self.lio_last is not None and np.any(self.lio_last[12:18] != 0):
            f["N_lio|lio.rot_var_log"] = (_log(self.lio_last[12:15].sum(), 1e-12), self.lio_last[0])
            f["N_lio|lio.pos_var_log"] = (_log(self.lio_last[15:18].sum(), 1e-12), self.lio_last[0])
        # --- fact states (ontology capabilities, frozen indicators) --------------------
        for cap, st in states.items():
            for s in STATES:
                put(f"state.{cap}.{s}", 1.0 if st == s else 0.0, t)
        # --- navigation algorithms -------------------------------------------------------
        for k, node in (("gnss_ins", "N_gi"), ("vio", "N_vio"), ("lio", "N_lio"), ("ins", "N_ins")):
            ok = cands[k] is not None
            f[f"{node}|out.valid"] = (1.0 if ok else 0.0, t)
            if k != "ins":
                o = snap.outputs.get(NAV_SRC[k])
                if o is not None:
                    f[f"{node}|out.age_s"] = (min(t - o.stamp, 5.0), t)
        aid = {"N_gi": bank.last_aid["gnss_ins"], "N_vio": bank.last_aid["vio"],
               "N_lio": bank.last_aid["lio"]}
        for node, la in aid.items():
            if np.isfinite(la):
                f[f"{node}|aid.age_log"] = (float(np.log1p(min(t - la, 600.0))), t)
        if bank.dr.anchor_t is not None:
            f["N_ins|anchor.age_log"] = (float(np.log1p(min(bank.dr.age(t), 600.0))), t)
        for k, node in (("vio", "N_vio"), ("lio", "N_lio")):
            src = NAV_SRC[k]
            al = bank.eng.align[src]
            f[f"{node}|align.valid"] = (1.0 if al.valid else 0.0, t)
            if al.valid:
                f[f"{node}|align.age_log"] = (float(np.log1p(min(al.age(t), 600.0))), t)
                f[f"{node}|align.path_since_m"] = (float(al.path_since_update_m), t)
            for key, name in ((f"align.{src}.residual_m", "align.residual_m"),
                              (f"align.{src}.vel_disagree_mps", "align.vel_disagree_mps")):
                if key in sig:
                    f[f"{node}|{name}"] = (float(sig[key][0]), sig[key][1])
            if np.isfinite(bank.last_reset[src]):
                f[f"{node}|reset.age_log"] = (float(np.log1p(min(t - bank.last_reset[src], 600.0))), t)
        for k in CANDIDATES:
            node = {"gnss_ins": "N_gi", "vio": "N_vio", "lio": "N_lio", "ins": "N_ins"}[k]
            for o in CANDIDATES:
                if o == k or cands[k] is None or cands[o] is None:
                    continue
                dp = float(np.linalg.norm((cands[k][0] - cands[o][0])[:2]))
                dv = float(np.linalg.norm(cands[k][2] - cands[o][2]))
                f[f"{node}|dis.{o}.pos"] = (float(np.log1p(dp)), t)
                f[f"{node}|dis.{o}.vel"] = (float(np.log1p(dv)), t)
        return self._layout(f, t)

    @staticmethod
    def _layout(f, t):
        """-> {node: (values[F], stamps[F])}; node-qualified keys win over shared ones."""
        out = {}
        for node, names in SPEC.items():
            vals = np.full(len(names), np.nan)
            stamps = np.full(len(names), np.nan)
            for j, n in enumerate(names):
                hit = f.get(f"{node}|{n}") or f.get(n)
                if hit is not None:
                    vals[j], stamps[j] = hit
            out[node] = (vals, stamps)
        return out
