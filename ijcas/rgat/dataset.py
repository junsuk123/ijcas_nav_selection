"""Per-run tensors for the graph selector (design 5.1-5.2).

    build_run(run_dir, cfg) -> dict of arrays (saved as one npz per run)

Inputs (observation-only, 20 Hz):  y_<node> raw feature values (NaN = missing), m_<node> presence,
  dt_<node> age of the source value [s]; valid[T,K] = policy eligibility of each candidate.
Labels / evaluation (ground truth, offline only):  cand_* candidate states in the GT world frame,
  err_pos/vel/att per candidate, gt_*, regime per tick (from the injected fault schedule).
Rule-based comparators (frozen first implementation, at its own 10 Hz):  Ontology-K and H-FSM
  outputs and errors (rule_*).

Global frame: candidates are expressed in PX4's local ENU frame; the constant PX4->world offset
(GPS_GLOBAL_ORIGIN) is added for evaluation only, never for the inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..evaluation import gt_at
from ..replay import px4_to_world
from ..selector.engine import HealthFsm, SelectionEngine
from ..selector.facts import FactStream
from ..selector.ontology import OntologyReasoner
from .features import SPEC, CandidateBank, FeatureExtractor
from .graph import CANDIDATES

RATE_HZ = 20.0
RULE_RATE_HZ = 10.0
REGIMES = ("normal", "single", "compound", "recovery", "warmup")
RECOVERY_S = 15.0
TARGET_GROUP = {"gnss": "gnss", "gnss_ros": "gnss", "camera": "camera", "lidar": "lidar",
                "imu": "imu", "mag": "aux", "baro": "aux"}
RULE_LABEL = {"px4": 0, "openvins": 1, "fastlio": 2, "ins_dr": 3, None: -1}


def regime_of(t: np.ndarray, faults: list, normal_s: float) -> np.ndarray:
    """Offline label per tick: warmup (< normal_s), single / compound degradation (number of
    distinct degraded sensor groups), recovery (<= 15 s after the last fault ended), normal."""
    out = np.zeros(len(t), int)
    ends = sorted(f["end_s"] for f in faults)
    for i, ti in enumerate(t):
        if ti < normal_s:
            out[i] = 4
            continue
        groups = {TARGET_GROUP.get(f["target"], f["target"]) for f in faults
                  if f["start_s"] <= ti < f["end_s"]}
        if len(groups) == 1:
            out[i] = 1
        elif len(groups) > 1:
            out[i] = 2
        elif any(0.0 <= ti - e <= RECOVERY_S for e in ends):
            out[i] = 3
    return out


def _errors(gt, t, pos, quat, vel):
    """pos/vel/attitude errors of states already in the world frame (NaN where missing)."""
    gp, gr, gv = gt_at(gt, t)
    ep = np.linalg.norm(pos - gp, axis=-1)
    ev = np.linalg.norm(vel - gv, axis=-1)
    ea = np.full(len(t), np.nan)
    ok = np.isfinite(quat).all(-1)
    if ok.any():
        ea[ok] = np.degrees((gr[ok].inv() * R.from_quat(quat[ok])).magnitude())
    return ep, ev, ea, gp, gr, gv


def build_run(run_dir, cfg: dict) -> dict:
    run = Path(run_dir)
    man = json.loads((run / "manifest.json").read_text())
    gt = np.load(run / "gt.npz")["gt"]
    rec = np.load(run / "record.npz")
    offset, how = px4_to_world(run, gt, man)
    bank = CandidateBank(cfg)
    fx = FeatureExtractor(rec["ov_pose_cov"], cfg,
                          rec["odom_fastlio"] if "odom_fastlio" in rec.files else None)
    ts, ys, ss, cp, cq, cv = [], {n: [] for n in SPEC}, {n: [] for n in SPEC}, [], [], []
    for snap in FactStream(run, RATE_HZ):
        cands, states, sig = bank.step(snap)
        feats = fx(snap, bank, cands, states, sig)
        ts.append(snap.t)
        for n, (v, s) in feats.items():
            ys[n].append(v)
            ss[n].append(s)
        P, Q, V = (np.full((len(CANDIDATES), d), np.nan) for d in (3, 4, 3))
        for k, c in enumerate(CANDIDATES):
            if cands[c] is not None:
                P[k], Q[k], V[k] = cands[c][0] + offset, cands[c][1], cands[c][2]
        cp.append(P), cq.append(Q), cv.append(V)
    t = np.asarray(ts)
    out = {"t": t}
    for n in SPEC:
        y, s = np.asarray(ys[n]), np.asarray(ss[n])
        out[f"y_{n}"] = y.astype(np.float32)
        out[f"m_{n}"] = np.isfinite(y).astype(np.float32)
        out[f"dt_{n}"] = np.where(np.isfinite(s), np.clip(t[:, None] - s, 0.0, 600.0),
                                  600.0).astype(np.float32)
    cp, cq, cv = np.asarray(cp), np.asarray(cq), np.asarray(cv)
    out["valid"] = np.isfinite(cp).all(-1)
    out.update(cand_pos=cp, cand_quat=cq, cand_vel=cv)
    K = len(CANDIDATES)
    ep, ev, ea = (np.full((len(t), K), np.nan) for _ in range(3))
    for k in range(K):
        ep[:, k], ev[:, k], ea[:, k], gp, gr, gv = _errors(gt, t, cp[:, k], cq[:, k], cv[:, k])
    out.update(err_pos=ep, err_vel=ev, err_att=ea, gt_pos=gp, gt_vel=gv, gt_quat=gr.as_quat())
    sc = man["scenario"]
    normal_s = float(sc.get("normal_s", 20.0))
    out["regime"] = regime_of(t, sc.get("faults", []), normal_s)
    out.update(_rule_comparators(run, cfg, gt, offset, normal_s, sc.get("faults", [])))
    meta = {"run": str(run), "set": run.parent.name, "id": run.name, "scenario": sc.get("id"),
            "seed": man["seed"], "scene": man["scene"], "variant": sc.get("variant"),
            "faults": sc.get("faults", []), "normal_s": normal_s, "offset": offset.tolist(),
            "origin_method": how, "rate_hz": RATE_HZ}
    out["meta"] = np.array(json.dumps(meta))
    return out


def _rule_comparators(run, cfg, gt, offset, normal_s, faults) -> dict:
    """Ontology-K and H-FSM exactly as frozen (10 Hz, own alignment and limited-DR anchor)."""
    sels = {"ont": SelectionEngine(OntologyReasoner(cfg), cfg, name="ontology"),
            "hfsm": HealthFsm(cfg)}
    rows = {k: {"sel": [], "mode": [], "pos": [], "quat": [], "vel": []} for k in sels}
    ts = []
    for snap in FactStream(run, RULE_RATE_HZ):
        ts.append(snap.t)
        for k, s in sels.items():
            d, o = s.step(snap)
            r = rows[k]
            r["sel"].append(RULE_LABEL.get(d.selected, -1))
            r["mode"].append({"normal": 0, "limited_dr": 1, "no_valid_nav": 2}[d.output_mode])
            if o is None:
                r["pos"].append([np.nan] * 3), r["quat"].append([np.nan] * 4), r["vel"].append([np.nan] * 3)
            else:
                r["pos"].append(np.asarray(o[0]) + offset), r["quat"].append(o[1]), r["vel"].append(o[2])
    t = np.asarray(ts)
    out = {"rule_t": t, "rule_regime": regime_of(t, faults, normal_s)}
    for k, r in rows.items():
        P, Q, V = np.asarray(r["pos"]), np.asarray(r["quat"]), np.asarray(r["vel"])
        ep, ev, ea, *_ = _errors(gt, t, P, Q, V)
        out.update({f"rule_{k}_sel": np.asarray(r["sel"]), f"rule_{k}_mode": np.asarray(r["mode"]),
                    f"rule_{k}_pos": P, f"rule_{k}_vel": V, f"rule_{k}_quat": Q,
                    f"rule_{k}_err_pos": ep, f"rule_{k}_err_vel": ev, f"rule_{k}_err_att": ea})
    return out
