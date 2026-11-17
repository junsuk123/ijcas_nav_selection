"""G2: nominal behaviour of every candidate (S00 runs) -> results/g2/*.json

    python3 tools/g2_report.py $IJCAS_DATA/g2/S00_warehouse_1 [...more runs]

Per candidate: initialisation (first output), output rate / gaps, latency (reception - stamp, in
simulation time), error with the alignment conventions of ijcas.evaluation, 10 m drift.
Baselines: GNSS-only error; INS-DR free-inertial drift after 10/30/60 s from anchors at several
times (anchored on ground truth here because this characterises the baseline itself; online the
anchor is the last valid aided estimate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.baselines import InsDr, InsState, gnss_only  # noqa: E402
from ijcas.evaluation import errors, gt_at, load_run, relative_drift, timing  # noqa: E402


def ins_drift(imu, gt, anchors=(30.0, 70.0, 110.0), horizons=(10.0, 30.0, 60.0)):
    out = {f"{int(h)}s": [] for h in horizons}
    for a in anchors:
        gp, gr, gv = gt_at(gt, np.array([a]))
        ins = InsDr(InsState(a, gp[0], gv[0], gr[0], np.zeros(3), np.zeros(3)))
        traj = ins.run(imu, a + max(horizons))
        for h in horizons:
            if len(traj) == 0 or traj[-1, 0] < a + h - 0.01:
                continue
            i = np.searchsorted(traj[:, 0], a + h)
            i = min(i, len(traj) - 1)
            g, _, _ = gt_at(gt, traj[i:i + 1, 0])
            out[f"{int(h)}s"].append(float(np.linalg.norm(traj[i, 1:4] - g[0])))
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}


def report(run_dir: str) -> dict:
    run = load_run(run_dir)
    gt, man = run["gt"], run["manifest"]
    res = {"run": run_dir, "scene": man["scene"], "seed": man["seed"], "sim_s": man["sim_s"],
           "rtf": man["rtf"], "candidates": {}}
    motion_start = float(man["scenario"].get("hover_s", 8.0))
    for name, tr in run["tracks"].items():
        if tr is None:
            res["candidates"][name] = {"outputs": 0}
            continue
        c = {"outputs": len(tr), **timing(tr)}
        if name == "px4":
            c["error_origin_initial"] = errors(tr, gt, "origin", 10.0, t_from=motion_start)
        else:
            c["error_initial_4dof"] = errors(tr, gt, "initial", 10.0)
            c["error_full_4dof"] = errors(tr, gt, "full")
        c["drift"] = relative_drift(tr, gt)
        res["candidates"][name] = c
    rec = run["record"]
    g = gnss_only(rec["fix"], rec["gvel"], man["origin"])
    gp, _, gv = gt_at(gt, g["stamp"])
    ok = g["valid"] & (g["stamp"] >= gt[0, 0])
    e = np.linalg.norm(g["pos"][ok] - gp[ok], axis=1)
    ev = np.linalg.norm(g["vel"][ok] - gv[ok], axis=1)
    res["candidates"]["gnss_only"] = {
        "outputs": int(ok.sum()), "rate_hz": float(1 / np.median(np.diff(g["stamp"]))),
        "pos_rmse_m": float(np.sqrt(np.mean(e ** 2))), "pos_p95_m": float(np.percentile(e, 95)),
        "horiz_rmse_m": float(np.sqrt(np.mean(np.sum((g["pos"][ok] - gp[ok])[:, :2] ** 2, 1)))),
        "vel_rmse_mps": float(np.sqrt(np.nanmean(ev ** 2))),
        "reported_sigma_h_m": float(np.median(g["sigma_h"][ok]))}
    res["candidates"]["ins_dr"] = {"free_inertial_drift_m": ins_drift(rec["imu"], gt),
                                   "note": "GT-anchored characterisation of the baseline"}
    return res


if __name__ == "__main__":
    out_dir = ROOT / "results/g2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rd in sys.argv[1:]:
        r = report(rd)
        name = Path(rd).name
        (out_dir / f"{name}.json").write_text(json.dumps(r, indent=1))
        print(f"== {name}  rtf {r['rtf']:.2f}")
        for k, c in r["candidates"].items():
            if k in ("openvins", "fastlio", "px4") and c.get("outputs"):
                e = c.get("error_initial_4dof") or c.get("error_origin_initial")
                f = c.get("error_full_4dof", {})
                print(f"  {k:8s} n={c['outputs']:5d} first={c['first_output_s']:6.1f}s "
                      f"rate={c['rate_hz']:5.1f} lat50={1000 * c['latency_median_s']:5.0f}ms "
                      f"lat95={1000 * c['latency_p95_s']:5.0f}ms | init-align RMSE "
                      f"{e.get('pos_rmse_m', float('nan')):6.2f} m final {e.get('pos_final_m', float('nan')):6.2f} "
                      f"yaw {e.get('yaw_rmse_deg', float('nan')):5.1f} deg | full-align "
                      f"{f.get('pos_rmse_m', float('nan')):6.2f} m | drift/10m "
                      f"{c['drift']['drift_per_10m_mean_m']}")
            elif k in ("openvins", "fastlio", "px4"):
                print(f"  {k:8s} NO OUTPUT")
        g = r["candidates"]["gnss_only"]
        print(f"  gnss     pos RMSE {g['pos_rmse_m']:.2f} m (horiz {g['horiz_rmse_m']:.2f}) "
              f"vel {g['vel_rmse_mps']:.2f} m/s; INS-DR drift {r['candidates']['ins_dr']['free_inertial_drift_m']}")
