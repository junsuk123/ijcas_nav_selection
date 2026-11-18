"""Per-run tensors for the navigation-solution fusion (20 Hz), built from one recorded run.

    build_fusion_run(run_dir, cfg) -> dict (saved as one npz per run, tools/fusion_build_dataset.py)

Online inputs (what the runtime sees; FactStream refuses ground-truth topics):
  y_/m_/dt_<node>   the v1 node features (ijcas/rgat/features.py), unchanged, for all 12 v1 nodes
  y_cov             [T,3,3] log10 trace of each solution's reported covariance block (pos, vel, att);
                    NaN where the estimator reported nothing (never substituted)
  y_F_nav           [T,3] fusion-node features: number of eligible solutions, number reporting a
                    position covariance, largest pairwise position disagreement of eligible solutions
  valid3            [T,3] eligibility of (gnss_ins, vio, lio) - the v1 policy of CandidateBank
                    (output age, alignment, aiding age, IMU reception)
  pos3/vel3/quat3   [T,3,...] solution states in the common frame (PX4 local ENU; the constant
                    PX4->world offset of the run is added for evaluation only, it does not change the
                    fusion, which is translation equivariant), position propagated causally to the tick
                    time with the solution's own velocity over its output age (age3)
  cov3, avail3      [T,3,3,3,3] / [T,3,3] reported covariance blocks (ijcas/fusion/covariance.py)
  valid4, ins_*     v1 policy mask incl. INS and the INS state (selection baselines / limited fallback)
Offline (evaluation and training labels only): gt_pos, gt_vel, gt_quat, err3_*, regime, meta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..evaluation import gt_at
from ..replay import px4_to_world
from ..rgat.dataset import RATE_HZ, _rule_comparators, regime_of
from ..rgat.features import NAV_SRC, SPEC, CandidateBank, FeatureExtractor
from ..selector.facts import FactStream
from .covariance import solution_blocks

SOLUTIONS = ("gnss_ins", "vio", "lio")


def _sorted_odom(rec, name):
    a = rec[f"odom_{name}"] if f"odom_{name}" in rec.files else np.zeros((0, 18))
    return a[np.argsort(a[:, 1])] if a.size else a


def build_fusion_run(run_dir, cfg: dict, rules: bool = True) -> dict:
    run = Path(run_dir)
    man = json.loads((run / "manifest.json").read_text())
    gt = np.load(run / "gt.npz")["gt"]
    rec = np.load(run / "record.npz")
    offset, how = px4_to_world(run, gt, man)
    odom = {n: _sorted_odom(rec, n) for n in ("px4", "openvins", "fastlio")}
    bank = CandidateBank(cfg)
    fx = FeatureExtractor(rec["ov_pose_cov"], cfg, odom["fastlio"] if odom["fastlio"].size else None)
    rows = {k: [] for k in ("t", "valid3", "pos3", "vel3", "quat3", "age3", "cov3", "avail3", "y_cov",
                            "y_F_nav", "valid4", "ins_pos", "ins_vel", "ins_quat")}
    ys, ss = {n: [] for n in SPEC}, {n: [] for n in SPEC}
    nan3 = np.full(3, np.nan)
    for snap in FactStream(run, RATE_HZ):
        t = snap.t
        cands, states, sig = bank.step(snap)
        feats = fx(snap, bank, cands, states, sig)
        for n, (v, s) in feats.items():
            ys[n].append(v)
            ss[n].append(s)
        px4s = snap.px4_status[0] if snap.px4_status else None
        v3 = np.zeros(3, bool)
        P3, Q3, V3 = np.full((3, 3), np.nan), np.full((3, 4), np.nan), np.full((3, 3), np.nan)
        age, cov, av, ycov = np.full(3, np.nan), np.full((3, 3, 3, 3), np.nan), np.zeros((3, 3), bool), \
            np.full((3, 3), np.nan)
        for k, sol in enumerate(SOLUTIONS):
            src = NAV_SRC[sol]
            o = snap.outputs.get(src)
            if cands[sol] is None or o is None:
                continue
            p, q, v = (np.asarray(x, float) for x in cands[sol])
            a = t - o.stamp
            v3[k] = True
            P3[k], Q3[k], V3[k], age[k] = p + v * a + offset, q, v, a
            od = odom[src]
            j = np.searchsorted(od[:, 1], t, side="right") - 1          # the row FactStream used
            row = od[j] if j >= 0 else None
            yaw = bank.eng.align[src].yaw if src in bank.eng.align else None
            cov[k], av[k] = solution_blocks(src, row, yaw, o.quat, px4s)
            for b in range(3):
                if av[k, b]:
                    ycov[k, b] = np.log10(max(np.trace(cov[k, b]), 1e-15))
        dis = [np.linalg.norm((P3[i] - P3[j])[:2]) for i in range(3) for j in range(i + 1, 3) if v3[i] and v3[j]]
        rows["y_F_nav"].append([float(v3.sum()), float(av[:, 0].sum()),
                                float(np.log1p(max(dis))) if dis else np.nan])
        ins = cands["ins"]
        rows["ins_pos"].append(np.asarray(ins[0]) + offset if ins is not None else nan3)
        rows["ins_vel"].append(np.asarray(ins[2]) if ins is not None else nan3)
        rows["ins_quat"].append(np.asarray(ins[1]) if ins is not None else np.full(4, np.nan))
        rows["valid4"].append([cands[c] is not None for c in ("gnss_ins", "vio", "lio", "ins")])
        for k, x in (("t", t), ("valid3", v3), ("pos3", P3), ("vel3", V3), ("quat3", Q3), ("age3", age),
                     ("cov3", cov), ("avail3", av), ("y_cov", ycov)):
            rows[k].append(x)
    out = {k: np.asarray(v) for k, v in rows.items()}
    t = out["t"]
    for n in SPEC:
        y, s = np.asarray(ys[n]), np.asarray(ss[n])
        out[f"y_{n}"] = y.astype(np.float32)
        out[f"m_{n}"] = np.isfinite(y).astype(np.float32)
        out[f"dt_{n}"] = np.where(np.isfinite(s), np.clip(t[:, None] - s, 0.0, 600.0), 600.0).astype(np.float32)
    gp, gr, gv = gt_at(gt, t)
    out.update(gt_pos=gp, gt_vel=gv, gt_quat=gr.as_quat())
    ep = np.linalg.norm(out["pos3"] - gp[:, None], axis=-1)
    ev = np.linalg.norm(out["vel3"] - gv[:, None], axis=-1)
    ea = np.full(ep.shape, np.nan)
    for k in range(3):
        ok = out["valid3"][:, k]
        if ok.any():
            ea[ok, k] = np.degrees((gr[ok].inv() * R.from_quat(out["quat3"][ok, k])).magnitude())
    out.update(err3_pos=ep, err3_vel=ev, err3_att=ea)
    sc = man["scenario"]
    normal_s = float(sc.get("normal_s", 20.0))
    out["regime"] = regime_of(t, sc.get("faults", []), normal_s)
    if rules:
        out.update(_rule_comparators(run, cfg, gt, offset, normal_s, sc.get("faults", [])))
    meta = {"run": str(run), "set": run.parent.name, "id": run.name, "scenario": sc.get("id"),
            "seed": man["seed"], "scene": man["scene"], "variant": sc.get("variant"),
            "faults": sc.get("faults", []), "normal_s": normal_s, "offset": np.asarray(offset).tolist(),
            "origin_method": how, "rate_hz": RATE_HZ,
            "full_covariance_record": bool(odom["px4"].size and odom["px4"].shape[1] >= 90)}
    out["meta"] = np.array(json.dumps(meta))
    return out
