"""Independent check of the OpenVINS simulator example (G0).

Reads the official run's saved files (``timestamp q(xyzw) p v ...``, global frame shared with
the simulator ground truth, filter initialised from truth so no alignment) and recomputes the
position / orientation RMSE that ``run_simulation`` prints at the end.

    python tools/g0_check_openvins.py results/g0/openvins_sim
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", usecols=range(8))


def main(out: Path) -> dict:
    est, gt = load(out / "ov_estimate.txt"), load(out / "ov_groundtruth.txt")
    t = np.intersect1d(np.round(est[:, 0], 6), np.round(gt[:, 0], 6))
    e = est[np.isin(np.round(est[:, 0], 6), t)]
    g = gt[np.isin(np.round(gt[:, 0], 6), t)]
    dp = e[:, 5:8] - g[:, 5:8]
    rot_err = (Rotation.from_quat(e[:, 1:5]) * Rotation.from_quat(g[:, 1:5]).inv()).magnitude()
    steps = np.linalg.norm(np.diff(g[:, 5:8], axis=0), axis=1)
    result = {
        "matched_states": int(len(t)),
        "duration_s": float(g[-1, 0] - g[0, 0]),
        "path_length_m": float(steps.sum()),
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(dp ** 2, axis=1)))),
        "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rot_err ** 2)))),
        "final_position_error_m": float(np.linalg.norm(dp[-1])),
        "official_log": (out / "final_metrics.txt").read_text(encoding="utf-8").strip().splitlines(),
    }
    (out / "check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
