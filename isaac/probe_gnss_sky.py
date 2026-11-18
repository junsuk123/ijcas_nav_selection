"""GNSS sky model along a scene's flight loop (no PX4, no sensors): satellites, DOP, LOS/NLOS, errors.

    isaac/run_isaac.sh isaac/probe_gnss_sky.py --scene urban --out <dir> [--seeds 1,2,3,4,5,6]

Builds the scene exactly as isaac/ijcas_world.py does, then evaluates ijcas_sim.gnss_sky.SkyModel (PhysX
ray casts) every ``--step`` metres of the loop for several seeds (each seed = other constellations / epoch).
Writes <out>/<scene>_gnss_sky.npz.  Used to check the model in the real geometry before pilot runs.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seeds", default="1,2,3,4,5,6")
parser.add_argument("--step", type=float, default=2.0)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import yaml  # noqa: E402

cfg = yaml.safe_load((ROOT / "config/scenes.yaml").read_text())["scenes"][args.scene]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402

from ijcas_sim.gnss_sky import SkyModel  # noqa: E402
from ijcas_sim.scenes import build_scene  # noqa: E402
from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402

pg = PegasusInterface()
pg._world_settings = {"physics_dt": 1.0 / 200.0, "stage_units_in_meters": 1.0, "rendering_dt": 0.05}
pg._world = World(**pg._world_settings)
tc = cfg["trajectory"]
traj = LoopTrajectory(tc["waypoints"], speed=float(tc["speed_mps"]),
                      corner_radius_m=float(tc.get("corner_radius_m", 0.0)))
s = np.arange(0.0, traj.length, args.step)
path = np.array([traj._path(x)[:3] for x in s])
seed0 = int(args.seeds.split(",")[0])
build_scene(pg, app, args.scene, cfg, seed0, path_xy=path[:, :2])
pg.world.reset()
omni.timeline.get_timeline_interface().play()
for _ in range(10):
    app.update()
res = {"s": s, "xyz": path, "seeds": np.array([int(x) for x in args.seeds.split(",")])}
keys = ("used", "n_los", "n_nlos", "n_multipath", "n_above_mask", "fix", "pdop", "hdop", "eph", "epv")
for k in keys:
    res[k] = np.zeros((len(res["seeds"]), len(s)))
res["pos_err"] = np.zeros((len(res["seeds"]), len(s), 3))
res["constellations"] = []
for i, sd in enumerate(res["seeds"]):
    sky = SkyModel(seed=int(sd), lat_deg=pg.latitude, lon_deg=pg.longitude)
    res["constellations"].append("+".join(sky.constellations))
    for j, p in enumerate(path):
        a = sky.assess(j * args.step / 6.0, p, np.zeros(3))        # ~6 m/s along the loop
        for k in keys:
            res[k][i, j] = float(a[k])
        res["pos_err"][i, j] = a["pos_err"]
    print(f"[probe] seed {sd} {res['constellations'][-1]}: used {res['used'][i].mean():.1f} "
          f"fix {res['fix'][i].mean():.2f} nlos {res['n_nlos'][i].mean():.2f} "
          f"herr med {np.median(np.linalg.norm(res['pos_err'][i, :, :2], axis=1)):.2f} m", flush=True)
res["constellations"] = np.array(res["constellations"])
np.savez_compressed(out / f"{args.scene}_gnss_sky.npz", **res)
print(f"[probe] wrote {out / f'{args.scene}_gnss_sky.npz'}", flush=True)
omni.timeline.get_timeline_interface().stop()
app.close()
