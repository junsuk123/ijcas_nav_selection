"""navigation_selector: one engine, interchangeable reasoners, the Notion 04 comparators.

``SelectionEngine(reasoner)`` runs every tick (10 Hz):
  1. causal alignment of the local engines to PX4's global frame while GnssPosition is Usable (G5)
  2. INS-DR anchor = last selected global state; DR is only a limited fallback (R09)
  3. reasoner -> candidate states + justification (Ontology-K or Flat-K: same inputs/knowledge)
  4. quality penalty q_i (QualityModel; calibrated in G3), cost J_i (R06), R07 hysteresis
  5. output: selected candidate in the global frame, or limited DR, or no-valid-nav (R09)
Comparators with the same output format: Fixed (F-G / F-V / F-L), HealthFsm (H-FSM), Oracle
(ground-truth best candidate, offline reference only).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..baselines import InsDr, InsState
from ..handover import Alignment
from .common import (LIMITED, UNKNOWN, UNUSABLE, USABLE, CostModel, Decision, SwitchManager,
                     load_config, signals)
from .eligibility import IMU_SAT_REALIGN_MAX_RESIDUAL_M, ImuReception, ImuSaturation

GLOBAL_TASK = "GlobalWaypoint"
# R08: the essential exteroceptive input of each local engine (ontology :requiresEssential).
# Losing it invalidates the engine's alignment: what it outputs after the loss must be re-aligned
# before it can be used globally or testify against GNSS (S08 recovery deadlock, results/g5).
LOCAL_INPUT = {"openvins": "VisualFeatures", "fastlio": "LidarStructure"}
# The engine's own estimator-level indicator (e.g. OpenVINS tracked features) reports the same loss
# from the inside; G3 found it the informative one (the image-level cues were not), so a confirmed
# 'Unusable' of either invalidates the alignment.
LOCAL_ESTIMATOR = {"openvins": "OpenVINS", "fastlio": "FAST_LIO"}


class DrAnchor:
    """INS-DR fallback anchored on the last selected global state; valid for max_dr_s."""

    def __init__(self, max_dr_s: float):
        self.max_dr_s = max_dr_s
        self.ins: InsDr | None = None
        self.anchor_t: float | None = None
        self.valid = False

    def anchor(self, t, pos, quat, vel):
        self.ins = InsDr(InsState(t, np.asarray(pos, float), np.asarray(vel, float),
                                  R.from_quat(quat), np.zeros(3), np.zeros(3)))
        self.anchor_t = t
        self.valid = True

    def age(self, t):
        return float("inf") if self.anchor_t is None else t - self.anchor_t

    def propagate(self, imu_rows, t):
        if self.ins is None:
            return None
        if self.age(t) > self.max_dr_s:
            self.valid = False
        for row in imu_rows:
            self.ins.step(row)
        s = self.ins.s
        return s.pos.copy(), s.rot.as_quat(), s.vel.copy()


class QualityModel:
    """q_i in [0, 1].  Initial form (G4); G3 replaces the coefficients by the calibrated ones.

    Uses only what the reasoner already established: measurement states of the candidate's
    inputs and, for local engines in the global task, the alignment age (drift since the last
    causal alignment)."""

    INPUTS = {"px4": ["GnssPosition", "GnssVelocity"], "openvins": ["VisualFeatures"],
              "fastlio": ["LidarStructure"], "ins_dr": []}

    def __init__(self, cfg: dict, coef: dict | None = None):
        self.coef = coef or {"limited": 0.5, "unknown": 0.7, "align_age_s": 60.0, "dr": 0.8}
        self.models = cfg.get("quality_model", {}) or {}

    def _logistic(self, m, sig, align_age):
        x = []
        for f in m["features"]:
            if f.startswith("align_age_"):
                v = align_age if align_age is not None and np.isfinite(align_age) else np.nan
            else:
                v = sig[f][0] if f in sig else np.nan
            miss = not np.isfinite(v)
            x += [m["median"][f] if miss else float(v), 1.0 if miss else 0.0]
        z = (np.asarray(x) - np.asarray(m["mu"])) / np.asarray(m["sd"])
        return float(1.0 / (1.0 + np.exp(-(z @ np.asarray(m["coef"]) + m["intercept"]))))

    def q(self, cid, ms, align_age, sig=None):
        if cid == "ins_dr":
            return self.coef["dr"]
        if cid in self.models and sig is not None:        # G3-calibrated P(global error > 1 m)
            return self._logistic(self.models[cid], sig, align_age)
        st = [ms.get(m, UNKNOWN) for m in self.INPUTS[cid]]
        q = max([0.0] + [self.coef["limited"] if s == LIMITED else
                         self.coef["unknown"] if s in (UNKNOWN, UNUSABLE) else 0.0 for s in st])
        if cid in ("openvins", "fastlio") and align_age is not None and np.isfinite(align_age):
            q = min(1.0, q + align_age / self.coef["align_age_s"])
        return q


class SelectionEngine:
    def __init__(self, reasoner, cfg: dict | None = None, task: str = GLOBAL_TASK,
                 quality: QualityModel | None = None, name: str = "selector"):
        self.cfg = cfg or load_config()
        self.reasoner = reasoner
        self.task = task
        self.name = name
        s = self.cfg["selection"]
        self.align = {c: Alignment(window_s=s["alignment_window_s"],
                                   min_path_m=s["alignment_min_path_m"])
                      for c in ("openvins", "fastlio")}
        self.dr = DrAnchor(s["max_dr_s"])
        self.stale = s["alignment_stale_s"]
        self.cost = CostModel(self.cfg, 0.3)
        self.switch = SwitchManager(self.cfg)
        self.quality = quality or QualityModel(self.cfg)
        self.last_ms = {}
        self.imu_rx = ImuReception.from_cfg(self.cfg)
        self.imu_sat = ImuSaturation.from_cfg(self.cfg)

    def imu_received(self, snap) -> bool:
        """IMU reception criterion shared by every selector (2026-11-18): the IMU is the essential
        input of all candidates, so none is eligible (and no INS fallback) while it is not
        received.  Call once per tick."""
        self.imu_rx.update(snap.t, snap.imu_window)
        return self.imu_rx.ok(snap.t)

    # -- G5 helpers -------------------------------------------------------------------------
    def _update_alignment(self, snap, ms):
        lost_since = getattr(self, "_lost_since", {})
        self._lost_since = lost_since
        for c, m in LOCAL_INPUT.items():
            lost = [k for k in (m, LOCAL_ESTIMATOR[c]) if ms.get(k) == UNUSABLE]
            if lost:
                lost_since.setdefault(c, snap.t)
                if snap.t - lost_since[c] >= 1.0 and self.align[c].valid:   # confirmed loss
                    self.align[c].invalidate(snap.t, f"{'+'.join(lost)}=Unusable>=1s")
            else:
                lost_since.pop(c, None)
        ref = snap.outputs.get("px4")
        ref_ok = ref is not None and ms.get("GnssPosition") == USABLE and \
            ms.get("PX4_EKF2", USABLE) == USABLE and snap.t - ref.stamp < 0.3
        for c, al in self.align.items():
            for o in snap.output_history.get(c, []):
                al.observe_local(o.stamp, o.pos, o.vel)
            out = snap.outputs.get(c)
            if ref_ok and out is not None and abs(out.stamp - ref.stamp) < 0.1:
                al.add_pair(snap.t, out.pos, ref.pos)
        # IMU quality (2026-11-18): a saturation report means the local engines' states may be
        # corrupted -> their alignment is invalidated (re-established only after the saturation ends,
        # and no witness role until mature again).  Called once per tick by every selector.
        self.imu_sat.update(snap.t, snap.diag.get("imu"))
        if self.imu_sat.active(snap.t):
            for c, al in self.align.items():
                if al.valid or al.pairs or al.yaw is not None:
                    al.invalidate(snap.t, "imu_saturation")
                al.require_residual_m = IMU_SAT_REALIGN_MAX_RESIDUAL_M

    def derived_signals(self, snap) -> dict:
        """Cross-source consistency (R02).  Each causally aligned local engine is a witness for
        PX4's GNSS-based global position if its alignment is mature (>= 20 s, fit residual
        <= 1.5 m) and its expected error bound (residual + drift rate x path since the last fit)
        is still below 5 m.  The signal is the disagreement normalised by that bound, for the
        most trustworthy witness (smallest bound), averaged over 1 s; > 1 means the GNSS-based
        position disagrees with an independent source by more than that source's own uncertainty.
        A witness that just recovered from an input loss or has drifted too far cannot overrule
        GNSS (S08 recovery), and a long-lived accurate one keeps a GNSS bias exposed (S02)."""
        px4 = snap.outputs.get("px4")
        if px4 is None or snap.t - px4.stamp > 0.3:
            return {}
        out = self._agreement_signals(snap, px4)
        rates = self.cfg["selection"].get("local_drift_rate", {"openvins": 0.02, "fastlio": 0.004})
        best = None
        for c, al in self.align.items():
            o = snap.outputs.get(c)
            if o is None or not al.mature(snap.t) or snap.t - o.stamp > 0.3:
                continue
            bound = al.error_bound(rates.get(c, 0.02))
            if bound > 5.0:
                continue
            g = al.to_global(o.pos, o.quat, o.vel)[0]
            z = float(np.linalg.norm((g - px4.pos)[:2])) / bound
            if best is None or bound < best[0]:
                best = (bound, z)
        if best is None:
            return out
        self._cons = [x for x in getattr(self, "_cons", []) if x[0] > snap.t - 1.0] + [(snap.t, best[1])]
        out["consistency.gnss_local_z"] = (float(np.mean([x[1] for x in self._cons])), snap.t)
        return out

    def _agreement_signals(self, snap, px4) -> dict:
        """Quality-model inputs of the local engines (G3): the residual of the current causal
        alignment fit and the disagreement of the aligned engine velocity with PX4's (1 s mean).
        Velocity is used because a slow GNSS position bias barely moves it, while an engine that
        diverges (e.g. VIO under IMU saturation, results/g3) shows it at once.  Which of the two
        is wrong is left to the calibrated quality model; the states are not changed here."""
        out = {}
        hist = getattr(self, "_vdis", None)
        if hist is None:
            hist = self._vdis = {c: [] for c in self.align}
        for c, al in self.align.items():
            if not al.valid:
                hist[c] = []
                continue
            if al.residual_m is not None:
                out[f"align.{c}.residual_m"] = (float(al.residual_m), snap.t)
            o = snap.outputs.get(c)
            if o is None or snap.t - o.stamp > 0.3:
                continue
            v = R.from_euler("z", al.yaw).apply(o.vel)
            hist[c] = [x for x in hist[c] if x[0] > snap.t - 1.0] + \
                [(snap.t, float(np.linalg.norm(v - px4.vel)))]
            out[f"align.{c}.vel_disagree_mps"] = (float(np.mean([x[1] for x in hist[c]])), snap.t)
        return out

    def global_state(self, cid, snap):
        """Candidate output in the global (PX4 local ENU) frame, or None."""
        if cid == "px4":
            o = snap.outputs.get("px4")
            return None if o is None else (o.pos, o.quat, o.vel)
        if cid in self.align:
            o, al = snap.outputs.get(cid), self.align[cid]
            if o is None or not al.valid:
                return None
            return al.to_global(o.pos, o.quat, o.vel)
        if cid == "ins_dr":
            return self._dr_state
        return None

    def step(self, snap) -> tuple[Decision, tuple | None]:
        sig = {**signals(snap), **self.derived_signals(snap)}
        self.last_sig = sig                    # what this tick decided on (G3 dataset, logging)
        # measurement states first (for alignment gating), then the full evaluation
        alignment = {**self.align, "ins_dr": self.dr}
        states, reasons, ms = self.reasoner.evaluate(self.task, sig, snap.t, snap.outputs,
                                                     alignment, self.stale)
        self._update_alignment(snap, ms)
        self._dr_state = self.dr.propagate(snap.imu_window if snap.imu_window is not None else [],
                                           snap.t)
        imu_ok = self.imu_received(snap)
        if not imu_ok:
            for cid in states:
                reasons.setdefault(cid, []).append(self.imu_rx.reason(snap.t))
        ins_clean = self.imu_sat.clean_since(self.dr.anchor_t)
        if not ins_clean and "ins_dr" in states:
            reasons.setdefault("ins_dr", []).append("imu_saturation:anchor_before_saturation")
        costs, eligible = {}, {}
        for cid, st in states.items():
            if not imu_ok or st not in (USABLE, LIMITED):
                continue
            if cid == "ins_dr" and not ins_clean:
                continue
            if self.global_state(cid, snap) is None and self.task == GLOBAL_TASK:
                continue
            out = snap.outputs.get(cid)
            age = 0.0 if cid == "ins_dr" or out is None else snap.t - out.stamp
            al_age = self.align[cid].age(snap.t) if cid in self.align else None
            q = self.quality.q(cid, ms, al_age, sig)
            costs[cid] = self.cost.cost(q, age, st, cid == self.switch.current)
            if cid != "ins_dr":
                eligible[cid] = costs[cid]
        selected, switched = self.switch.update(snap.t, eligible)
        mode = "normal"
        if selected is None and "ins_dr" in costs and self.dr.valid:
            selected, mode = "ins_dr", "limited_dr"
        elif selected is None:
            mode = "no_valid_nav"
        out = self.global_state(selected, snap) if selected else None
        if selected and selected != "ins_dr" and out is not None:
            self.dr.anchor(snap.t, *out)
        d = Decision(snap.t, self.task, states, reasons, ms, costs, selected, mode, switched)
        return d, out


class Fixed:
    """F-G / F-V / F-L: always the same engine (causally aligned for the global task)."""

    def __init__(self, cid: str, cfg: dict | None = None):
        self.cid = cid
        self.inner = SelectionEngine(_NullReasoner(), cfg, name=f"fixed_{cid}")

    def step(self, snap):
        self.inner._update_alignment(snap, _ms_for_alignment(snap, self.inner.cfg, self.inner))
        imu_ok = self.inner.imu_received(snap)
        out = self.inner.global_state(self.cid, snap)
        o = snap.outputs.get(self.cid)
        if not imu_ok or (o is not None and snap.t - o.stamp > 0.3):
            out = None
        d = Decision(snap.t, GLOBAL_TASK, {}, {}, {}, {}, self.cid if out is not None else None,
                     "normal" if out is not None else "no_valid_nav")
        return d, out


class HealthFsm:
    """H-FSM: reactive switching on sensor health / message freshness only (no environment
    diagnostics, no capability reasoning): GNSS fix -> PX4, else VIO, else LIO, else DR."""

    def __init__(self, cfg: dict | None = None):
        self.inner = SelectionEngine(_NullReasoner(), cfg, name="h_fsm")

    def step(self, snap):
        e = self.inner
        ms = _ms_for_alignment(snap, e.cfg, e)
        e._update_alignment(snap, ms)
        e._dr_state = e.dr.propagate(snap.imu_window if snap.imu_window is not None else [],
                                     snap.t)
        imu_ok = e.imu_received(snap)
        fresh = lambda c: snap.outputs.get(c) is not None and snap.t - snap.outputs[c].stamp < 0.3  # noqa
        g = snap.diag.get("gnss")
        fix = bool(g and g[0].get("fix") and snap.t - g[0]["t"] < 1.0)
        sel = None
        for cid, ok in (("px4", fix and fresh("px4")), ("openvins", fresh("openvins")),
                        ("fastlio", fresh("fastlio"))):
            if imu_ok and ok and e.global_state(cid, snap) is not None:
                sel = cid
                break
        mode = "normal"
        if sel is None and e.dr.valid and imu_ok and e.imu_sat.clean_since(e.dr.anchor_t):
            sel, mode = "ins_dr", "limited_dr"
        elif sel is None:
            mode = "no_valid_nav"
        out = e.global_state(sel, snap) if sel else None
        if sel and sel != "ins_dr" and out is not None:
            e.dr.anchor(snap.t, *out)
        return Decision(snap.t, GLOBAL_TASK, {}, {}, {}, {}, sel, mode), out


class Oracle:
    """Offline reference: the candidate with the smallest ground-truth error at each tick
    (aligned causally like the others).  Never an online input."""

    def __init__(self, gt: np.ndarray, cfg: dict | None = None, offset=None):
        self.gt = gt
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, float)
        self.inner = SelectionEngine(_NullReasoner(), cfg, name="oracle")

    def step(self, snap):
        from ..evaluation import gt_at
        e = self.inner
        e._update_alignment(snap, _ms_for_alignment(snap, e.cfg, e))
        imu_ok = e.imu_received(snap)
        gp, _, _ = gt_at(self.gt, np.array([snap.t]))
        best, err = None, np.inf
        for cid in ("px4", "openvins", "fastlio"):
            o = snap.outputs.get(cid)
            g = e.global_state(cid, snap)
            if not imu_ok or o is None or g is None or snap.t - o.stamp > 0.3:
                continue
            d = float(np.linalg.norm(g[0] + self.offset - gp[0]))
            if d < err:
                best, err = cid, d
        out = e.global_state(best, snap) if best else None
        return Decision(snap.t, GLOBAL_TASK, {}, {}, {}, {}, best,
                        "normal" if best else "no_valid_nav"), out


class _NullReasoner:
    def evaluate(self, *a, **k):
        return {}, {}, {}


def _ms_for_alignment(snap, cfg, engine=None):
    """GNSS position usability for the comparators' alignment: same indicators (incl. the
    cross-source consistency) as the selectors, so every comparator gets the same alignment."""
    from .common import indicator_state, worst
    sig = signals(snap)
    if engine is not None:
        sig.update(engine.derived_signals(snap))
    out = {}
    for m in ("GnssPosition", "PX4_EKF2", "VisualFeatures", "LidarStructure", "OpenVINS", "FAST_LIO"):
        inds = [i for i in cfg["indicators"] if m in i["assesses"]]
        out[m] = worst(indicator_state(i, sig, snap.t)[0] for i in inds)
    return out
