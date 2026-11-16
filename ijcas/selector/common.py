"""Shared selector mechanics (identical for Flat-K and Ontology-K; Notion 03 선택 절차 4-6).

* ``signals``      observation signals with their source stamps from a Snapshot
* ``indicator_state``  one indicator -> Usable / Limited / Unusable / Unknown
* ``CostModel``    J_i = w_q q_i + w_a min(1, a_i / L_max) + w_s [i != current]
* ``SwitchManager`` R07 hysteresis: minimum dwell, recovery confirmation, margin; immediate
                    exclusion of a current candidate that lost eligibility
* ``Decision``     what every selector returns each tick (states, reasons, selected, costs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

USABLE, LIMITED, UNKNOWN, UNUSABLE = "Usable", "Limited", "Unknown", "Unusable"
RANK = {USABLE: 3, LIMITED: 2, UNKNOWN: 1, UNUSABLE: 0}
ESTIMATOR_FLAG_POS_HORIZ_ABS = 16
CONFIG = Path(__file__).resolve().parents[2] / "config/selector/thresholds.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG).read_text())


def worst(states):
    states = list(states)
    return min(states, key=lambda s: RANK[s]) if states else UNKNOWN


def best(states):
    states = list(states)
    return max(states, key=lambda s: RANK[s]) if states else UNUSABLE


def signals(snap) -> dict:
    """{signal name: (value, source stamp)} - observation-only, no ground truth."""
    s = {}
    t = snap.t
    g = snap.diag.get("gnss")
    if g:
        p = g[0]
        s["gnss.fix"] = (1.0 if p.get("fix") else 0.0, p["t"])
        if p.get("pos_vel_residual_m") is not None:
            s["gnss.pos_vel_residual_m"] = (float(p["pos_vel_residual_m"]), p["t"])
    if snap.px4_status:
        p = snap.px4_status[0]
        s["px4.pos_horiz_ratio"] = (float(p["pos_horiz_ratio"]), p["t"])
        s["px4.vel_ratio"] = (float(p["vel_ratio"]), p["t"])
        s["px4.flag_pos_horiz_abs"] = (1.0 if int(p["flags"]) & ESTIMATOR_FLAG_POS_HORIZ_ABS
                                       else 0.0, p["t"])
    v = snap.diag.get("vision")
    if v:
        p = v[0]
        s["vision.image_age_s"] = (t - p["t"], t)
        for k in ("features", "grid_cells_ge10", "mean", "stereo_matches"):
            if p.get(k) is not None:
                s[f"vision.{k}"] = (float(p[k]), p["t"])
        if "ov_points_msckf" in p or "ov_points_slam" in p:
            n = float(p.get("ov_points_msckf", 0)) + float(p.get("ov_points_slam", 0))
            st = max(p.get("ov_points_msckf_t", p["t"]), p.get("ov_points_slam_t", p["t"]))
            s["openvins.tracked_features"] = (n, st)
    lidar = snap.diag.get("lidar")
    if lidar:
        p = lidar[0]
        s["lidar.scan_age_s"] = (t - p["t"], t)
        for k in ("points", "lambda_ratio"):
            if p.get(k) is not None:
                s[f"lidar.{k}"] = (float(p[k]), p["t"])
    imu = snap.diag.get("imu")
    if imu:
        p = imu[0]
        s["imu.rate_hz"] = (float(p["rate_hz"]), p["t"])
        s["imu.non_monotonic"] = (float(p["non_monotonic"]), p["t"])
        # signed: a causal sensor is never stamped ahead of the simulation clock; the negative
        # side is ordinary delivery latency (up to one render frame, results/g3).  The earlier
        # |stamp - clock| folded that latency into the fault side (false alarms, G3 finding).
        s["imu.stamp_lead_s"] = (float(p["stamp_minus_clock_s"]), p["t"])
        if p.get("saturation_frac") is not None:
            s["imu.saturation_frac"] = (float(p["saturation_frac"]), p["t"])
        for k in ("mag", "baro"):
            if p.get(f"{k}_rate_hz") is not None:
                s[f"aux.{k}_rate_hz"] = (float(p[f"{k}_rate_hz"]), p["t"])
    return s


def indicator_state(ind: dict, sig: dict, t: float) -> tuple[str, str]:
    """(state, reason code) of one indicator definition."""
    if ind["signal"] not in sig:
        if ind.get("missing") == "skip":        # cross-checks: no evidence is not a fault
            return USABLE, f"{ind['id']}:not_available"
        st = UNUSABLE if ind.get("missing") == "invalid" else UNKNOWN
        return st, f"{ind['id']}:missing"
    value, src_t = sig[ind["signal"]]
    if t - src_t > float(ind.get("valid_for_s", 1.0)):
        st = UNUSABLE if ind.get("missing") == "invalid" else UNKNOWN
        return st, f"{ind['id']}:stale({t - src_t:.2f}s)"
    if "invalid_below" in ind and value < ind["invalid_below"]:
        return UNUSABLE, f"{ind['id']}:{value:.3g}<{ind['invalid_below']}"
    if "invalid_above" in ind and value > ind["invalid_above"]:
        return UNUSABLE, f"{ind['id']}:{value:.3g}>{ind['invalid_above']}"
    if "degraded_below" in ind and value < ind["degraded_below"]:
        return LIMITED, f"{ind['id']}:{value:.3g}<{ind['degraded_below']}"
    if "degraded_above" in ind and value > ind["degraded_above"]:
        return LIMITED, f"{ind['id']}:{value:.3g}>{ind['degraded_above']}"
    return USABLE, f"{ind['id']}:ok"


@dataclass
class Decision:
    t: float
    task: str
    states: dict                       # candidate -> state for the task
    reasons: dict                      # candidate -> list of reason codes (justification)
    measurement_states: dict
    costs: dict = field(default_factory=dict)
    selected: str | None = None
    output_mode: str = "normal"        # normal | limited_dr | no_valid_nav
    switched: bool = False


class CostModel:
    def __init__(self, cfg: dict, max_age_s: float):
        w = cfg["selection"]["weights"]
        self.wq, self.wa, self.ws = w["q"], w["age"], w["switch"]
        self.lmax = max_age_s
        self.limited_penalty = cfg["selection"]["limited_penalty"]

    def cost(self, q: float, age: float, state: str, is_current: bool) -> float:
        q_eff = min(1.0, q + (self.limited_penalty if state == LIMITED else 0.0))
        return (self.wq * q_eff + self.wa * min(1.0, age / self.lmax)
                + (0.0 if is_current else self.ws))


class SwitchManager:
    """R07: hysteresis and recovery confirmation over the eligible set."""

    def __init__(self, cfg: dict):
        s = cfg["selection"]
        self.margin, self.dwell, self.confirm = s["margin"], s["min_dwell_s"], s["recovery_confirm_s"]
        self.current: str | None = None
        self.since: float = -1e9
        self.eligible_since: dict = {}

    def update(self, t: float, eligible: dict) -> tuple[str | None, bool]:
        """eligible: candidate -> cost (only eligible ones).  Returns (selected, switched)."""
        for c in list(self.eligible_since):
            if c not in eligible:
                del self.eligible_since[c]
        for c in eligible:
            self.eligible_since.setdefault(c, t)
        confirmed = {c: j for c, j in eligible.items()
                     if t - self.eligible_since[c] >= self.confirm - 1e-9 or c == self.current}
        prev = self.current
        if not eligible:
            self.current = None
        elif self.current not in eligible:                       # lost eligibility: immediate
            pool = confirmed or eligible
            self.current = min(pool, key=pool.get)
            self.since = t
        elif t - self.since >= self.dwell - 1e-9 and confirmed:
            cand = min(confirmed, key=confirmed.get)
            if cand != self.current and confirmed[cand] < eligible[self.current] - self.margin:
                self.current = cand
                self.since = t
        return self.current, self.current != prev
