"""G3 step 1: tick-level calibration dataset from validation runs (ground truth used here only).

    python3 tools/g3_dataset.py $IJCAS_DATA/runs/validation/*/  -> results/g3/dataset.npz

Per tick (10 Hz, after the first 20 s): every observation signal the selectors use, alignment
ages, and outcome labels computed from ground truth:
  gnss_pos_err   |GNSS-only fix - truth|            (NaN without fix)
  gnss_vel_err   |GNSS velocity - truth|
  rel_err_<c>    2 s relative position error of engine c (yaw from the run-level 4-DOF fit)
  glob_err_<c>   global error of candidate c in PX4's frame after the selector's causal alignment
  imu_fault      1 inside an injected IMU fault window (scenario label; calibration only)
  imu_fault_kind the injected IMU fault kind ('' outside), for kind-specific indicator labels
Runs are processed in parallel (one process per run).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.baselines import lla_to_enu  # noqa: E402
from ijcas.evaluation import fit_4dof, gt_at  # noqa: E402
from ijcas.replay import px4_to_world  # noqa: E402
from ijcas.selector.common import load_config, signals  # noqa: E402
from ijcas.selector.engine import SelectionEngine  # noqa: E402
from ijcas.selector.facts import FactStream  # noqa: E402
from ijcas.selector.ontology import OntologyReasoner  # noqa: E402

SIGNALS = ["gnss.fix", "consistency.gnss_local_z", "gnss.pos_vel_residual_m", "px4.pos_horiz_ratio", "px4.vel_ratio",
           "px4.flag_pos_horiz_abs", "vision.image_age_s", "vision.features",
           "vision.grid_cells_ge10", "vision.mean", "vision.stereo_matches",
           "openvins.tracked_features", "lidar.scan_age_s", "lidar.points", "lidar.lambda_ratio",
           "imu.rate_hz", "imu.non_monotonic", "imu.stamp_lead_s",
           "imu.saturation_frac", "aux.mag_rate_hz", "aux.baro_rate_hz",
           "align.openvins.residual_m", "align.glim.residual_m",
           "align.openvins.vel_disagree_mps", "align.glim.vel_disagree_mps"]
CANDS = ("px4", "openvins", "glim")


def run_rows(run: Path, cfg):
    man = json.loads((run / "manifest.json").read_text())
    gt = np.load(run / "gt.npz")["gt"]
    rec = np.load(run / "record.npz")
    offset, _ = px4_to_world(run, gt, man)
    # run-level yaw of each local engine (evaluation of relative error only)
    yaw = {}
    for c in ("openvins", "glim"):
        a = rec[f"odom_{c}"]
        if len(a) > 50:
            gp, _, _ = gt_at(gt, a[:, 0])
            rot, _ = fit_4dof(a[:, 2:5], gp)
            yaw[c] = rot
    eng = SelectionEngine(OntologyReasoner(cfg), cfg)
    imu_windows = [(f["start_s"], f.get("end_s") or 1e9, f["kind"]) for f in man["scenario"]["faults"]
                   if f["target"] == "imu"]
    rows = []
    hist = {c: [] for c in CANDS}
    fix = rec["fix"]
    for snap in FactStream(run, cfg["selection"]["rate_hz"]):
        d, _ = eng.step(snap)
        if offset is None:                     # older runs: PX4 origin from the hover
            o = snap.outputs.get("px4")
            if o is not None and 3.0 < snap.t < 8.0:
                hist.setdefault("_off", []).append(gt_at(gt, np.array([snap.t]))[0][0] - o.pos)
            if snap.t >= 8.0 and hist.get("_off"):
                offset = np.mean(hist["_off"], axis=0)
        if snap.t < 20.0 or offset is None:
            continue
        sig = eng.last_sig                     # exactly the signals the selector used this tick
        row = {s: (sig[s][0] if s in sig else np.nan) for s in SIGNALS}
        gp, _, gv = gt_at(gt, np.array([snap.t]))
        # GNSS outcome
        if snap.fix is not None and snap.fix[2] >= 0 and np.isfinite(snap.fix[3]):
            p = lla_to_enu(snap.fix[3:4], snap.fix[4:5], snap.fix[5:6], man["origin"])[0]
            gpf, _, _ = gt_at(gt, np.array([snap.fix[0]]))
            row["gnss_pos_err"] = float(np.linalg.norm(p - gpf[0]))
        else:
            row["gnss_pos_err"] = np.nan
        for c in CANDS:
            o = snap.outputs.get(c)
            g = eng.global_state(c, snap)
            row[f"glob_err_{c}"] = np.nan if g is None or o is None or snap.t - o.stamp > 0.3 \
                else float(np.linalg.norm(g[0] + offset - gp[0]))
            if c in eng.align:
                row[f"align_age_{c}"] = eng.align[c].age(snap.t)
            if o is not None and snap.t - o.stamp < 0.3:
                hist[c].append((snap.t, o.pos.copy()))
            row[f"rel_err_{c}"] = np.nan
            if c in yaw:
                past = [h for h in hist[c] if snap.t - 2.05 <= h[0] <= snap.t - 1.95]
                if past and o is not None and snap.t - o.stamp < 0.3:
                    g0, _, _ = gt_at(gt, np.array([past[-1][0]]))
                    d_est = yaw[c].apply(o.pos - past[-1][1])
                    row[f"rel_err_{c}"] = float(np.linalg.norm(d_est - (gp[0] - g0[0])))
            hist[c] = [h for h in hist[c] if h[0] > snap.t - 3.0]
        kinds = [k for a, b, k in imu_windows if a <= snap.t < b]
        row["imu_fault"] = float(bool(kinds))
        row["imu_fault_kind"] = kinds[0] if kinds else ""
        row["t"] = snap.t
        row["scenario"] = man["scenario"]["id"]
        row["scene"] = man["scene"]
        row["seed"] = man["seed"]
        rows.append(row)
    return rows


def _one(r):
    r = Path(r)
    if not (r / "DONE").exists() and not (r / "record.npz").exists():
        return r, None, "not finished"
    try:
        return r, run_rows(r, load_config()), None
    except Exception as e:  # recorded, not silently dropped
        return r, None, f"{type(e).__name__}: {e}"


def main(runs):
    from concurrent.futures import ProcessPoolExecutor
    allrows = []
    with ProcessPoolExecutor(int(os.environ.get("G3_WORKERS", "5"))) as ex:
        for r, rows, err in ex.map(_one, runs):
            if err:
                print(f"FAILED {r}: {err}", flush=True)
                continue
            allrows += rows
            print(f"{r.name}: {len(rows)} ticks", flush=True)
    keys = sorted({k for row in allrows for k in row})
    out = ROOT / "results/g3"
    out.mkdir(parents=True, exist_ok=True)
    data = {k: np.array([row.get(k, np.nan) for row in allrows],
                        dtype=object if k in ("scenario", "scene", "imu_fault_kind") else float) for k in keys}
    np.savez_compressed(out / "dataset.npz", **data)
    print("rows", len(allrows), "->", out / "dataset.npz")


if __name__ == "__main__":
    main(sys.argv[1:])
