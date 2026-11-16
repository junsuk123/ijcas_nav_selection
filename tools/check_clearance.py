"""Minimum clearance of a scene trajectory against a free-space grid (results/g1/scenes/*.npz).

    python3 tools/check_clearance.py warehouse
Grid cells are 'occupied' when a sphere of the probe's clearance radius (0.8 m) overlapped a
collider; the reported margin is the extra distance to the nearest occupied cell centre.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac"))
from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402


def check(scene: str) -> dict:
    scenes = yaml.safe_load((ROOT / "config/scenes.yaml").read_text())["scenes"]
    sc = scenes[scene]
    tcfg = sc["trajectory"]
    traj = LoopTrajectory(tcfg["waypoints"], speed=tcfg["speed_mps"], hover_s=0.0, ramp_s=4.0,
                          corner_radius_m=tcfg.get("corner_radius_m", 0.0))
    out = {"scene": scene, "length_m": round(traj.length, 1), **{k: round(v, 3) for k, v in
                                                                  traj.max_dynamics().items()}}
    grid = sc.get("clearance_grid")
    if grid:
        d = np.load(ROOT / grid)
        xs, ys = d["xs"], d["ys"]
        cell = float(xs[1] - xs[0])
        alts = sorted(float(k[4:]) for k in d.files if k.startswith("alt_"))
        # flight band only (e.g. office: furniture below 1.5 m and the ceiling above 2.5 m)
        alts = [a for a in alts if a in sc.get("clearance_alts", alts)]
        occ = np.zeros_like(d[f"alt_{alts[0]}"])
        for a in alts:
            occ |= d[f"alt_{a}"]
        dist = distance_transform_edt(~occ) * cell
        s = np.linspace(0, traj.length, int(traj.length / 0.1))
        pts = np.array([traj._path(v) for v in s])
        i = np.clip(np.round((pts[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1)
        j = np.clip(np.round((pts[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
        margin = dist[j, i]
        worst = int(np.argmin(margin))
        out.update({"min_margin_m": float(margin.min()), "worst_xy": pts[worst, :2].round(2).tolist(),
                    "inside_occupied": int((margin <= 0).sum()),
                    "z_range": [float(pts[:, 2].min()), float(pts[:, 2].max())]})
    return out


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1]), indent=2))
