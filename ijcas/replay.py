"""Offline replay of every selector on one recorded run (Stage A: identical inputs) + metrics.

    python3 -m ijcas.replay $IJCAS_DATA/runs/validation/S01_001 [--out result.json]

Selectors: F-G, F-V, F-L, H-FSM, Flat-K, Ontology-K (and Oracle as the reference).  All see the
same causal fact stream.  Global-frame errors use PX4's local-frame origin (GPS_GLOBAL_ORIGIN,
or, for older runs without it, the mean offset over the initial hover).

Metrics (Notion 04 평가 지표, global waypoint task, alarm limit 2 m):
  availability        share of ticks with an output whose error <= alarm
  output_fraction     share of ticks with any output (normal or limited DR)
  false_normal_s      time an output is given while its error > alarm  (H2)
  no_valid_nav_s      time the selector declares no valid navigation
  pos_rmse / p95      over ticks with an output
  switches, max_jump  handover count and largest position jump at a switch (raw output)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .baselines import lla_to_enu
from .evaluation import gt_at
from .selector.common import load_config
from .selector.engine import Fixed, HealthFsm, Oracle, SelectionEngine
from .selector.facts import FactStream
from .selector.flat import FlatRules
from .selector.ontology import OntologyReasoner


def px4_to_world(run_dir: Path, gt: np.ndarray, manifest: dict):
    """Offset to add to PX4-local-ENU positions to get the GT world frame (evaluation only)."""
    rec = json.loads((run_dir / "px4_mavlink.json").read_text()) if \
        (run_dir / "px4_mavlink.json").exists() else {}
    origin = rec.get("origin") or []
    if origin:
        lat, lon, alt = origin[-1][:3]
        return lla_to_enu(np.array([lat]), np.array([lon]), np.array([alt]),
                          manifest["origin"])[0], "gps_global_origin"
    # older runs: constant origin from PX4's output over the hover before motion
    px4 = np.load(run_dir / "record.npz")["odom_px4"]
    hover_end = float(manifest["scenario"].get("hover_s", 8.0))
    m = (px4[:, 0] > 3.0) & (px4[:, 0] < hover_end)
    gp, _, _ = gt_at(gt, px4[m, 0])
    return (gp - px4[m, 2:5]).mean(0), "hover_mean"


def make_selectors(cfg, gt, offset):
    return {"F-G": Fixed("px4", cfg), "F-V": Fixed("openvins", cfg), "F-L": Fixed("glim", cfg),
            "H-FSM": HealthFsm(cfg),
            "Flat-K": SelectionEngine(FlatRules(cfg), cfg, name="flat"),
            "Ontology-K": SelectionEngine(OntologyReasoner(cfg), cfg, name="ontology"),
            "Oracle": Oracle(gt, cfg, offset)}


def replay(run_dir: str | Path, cfg: dict | None = None, alarm: float = 2.0,
           keep_trace: bool = False, only: tuple | None = None) -> dict:
    run_dir = Path(run_dir)
    cfg = cfg or load_config()
    gt = np.load(run_dir / "gt.npz")["gt"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stream = FactStream(run_dir, cfg["selection"]["rate_hz"])
    offset, how = px4_to_world(run_dir, gt, manifest)
    sels = make_selectors(cfg, gt, offset)
    if only:
        sels = {k: v for k, v in sels.items() if k in only}
    traces = {k: [] for k in sels}
    timing = {k: 0.0 for k in sels}
    for snap in stream:
        for name, sel in sels.items():
            t0 = time.perf_counter()
            d, out = sel.step(snap)
            timing[name] += time.perf_counter() - t0
            traces[name].append((snap.t, d.selected, d.output_mode,
                                 None if out is None else np.asarray(out[0], float),
                                 d.states, d.switched,
                                 None if out is None else np.asarray(out[2], float),
                                 None if out is None else float(R.from_quat(out[1]).as_euler("ZYX")[0])))
    res = {"run": str(run_dir), "scene": manifest["scene"], "seed": manifest["seed"],
           "scenario": manifest["scenario"].get("id"), "variant": manifest["scenario"].get("variant"),
           "origin_method": how, "selectors": {}}
    dt = 1.0 / cfg["selection"]["rate_hz"]
    start = float(manifest["scenario"].get("normal_s", 20.0))
    for name, tr in traces.items():
        ts = np.array([r[0] for r in tr])
        m = ts >= start                          # evaluate after the initial normal period
        gp, _, _ = gt_at(gt, ts)
        err = np.full(len(tr), np.nan)
        for i, r in enumerate(tr):
            if r[3] is not None:
                err[i] = np.linalg.norm(r[3] + offset - gp[i])
        has = ~np.isnan(err)
        ok = has & (err <= alarm)
        sel = [r[1] for r in tr]
        switches = [i for i in range(1, len(sel)) if sel[i] != sel[i - 1] and m[i]]
        jumps, vjumps, yjumps = [], [], []
        for i in switches:
            if tr[i][3] is not None and tr[i - 1][3] is not None:
                # remove the motion of one tick so only the handover discontinuity remains
                expect = tr[i - 1][3] + (tr[i - 1][6] if tr[i - 1][6] is not None else 0) * dt
                jumps.append(float(np.linalg.norm(tr[i][3] - expect)))
                vjumps.append(float(np.linalg.norm(tr[i][6] - tr[i - 1][6])))
                yjumps.append(float(abs(np.degrees(np.angle(np.exp(1j * (tr[i][7] - tr[i - 1][7])))))))
        e = err[m & has]
        res["selectors"][name] = {
            "availability": float(ok[m].mean()), "output_fraction": float(has[m].mean()),
            "false_normal_s": float((has & ~ok)[m].sum() * dt),
            "no_valid_nav_s": float(np.array([r[2] == "no_valid_nav" for r in tr])[m].sum() * dt),
            "limited_dr_s": float(np.array([r[2] == "limited_dr" for r in tr])[m].sum() * dt),
            "pos_rmse_m": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
            "pos_p95_m": float(np.percentile(e, 95)) if len(e) else None,
            "switches": len(switches), "max_jump_m": max(jumps) if jumps else 0.0,
            "median_jump_m": float(np.median(jumps)) if jumps else 0.0,
            "max_vel_jump_mps": max(vjumps) if vjumps else 0.0,
            "max_yaw_jump_deg": max(yjumps) if yjumps else 0.0,
            "selected_share": {c: float(np.mean([s == c for s, mm in zip(sel, m) if mm]))
                               for c in ("px4", "openvins", "glim", "ins_dr", None)},
            "selector_ms_per_tick": 1000 * timing[name] / max(len(tr), 1)}
        if keep_trace:
            res["selectors"][name]["trace"] = [(r[0], r[1], r[2], None if np.isnan(x) else x)
                                               for r, x in zip(tr, err)]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    allres = []
    for rd in a.runs:
        r = replay(rd)
        allres.append(r)
        print(f"== {Path(rd).name} {r['scenario']} {r['scene']} origin={r['origin_method']}")
        for n, s in r["selectors"].items():
            print(f"  {n:11s} avail {s['availability']:.3f} out {s['output_fraction']:.3f} "
                  f"false-normal {s['false_normal_s']:6.1f}s noNav {s['no_valid_nav_s']:5.1f}s "
                  f"rmse {s['pos_rmse_m'] if s['pos_rmse_m'] is None else round(s['pos_rmse_m'], 2)} "
                  f"sw {s['switches']:3d} jump {s['max_jump_m']:.2f} "
                  f"share {{{', '.join(f'{k}:{v:.2f}' for k, v in s['selected_share'].items() if v)}}} "
                  f"{s['selector_ms_per_tick']:.2f}ms")
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1, default=str))
