"""Scene inspection: extents and a free-space grid at flight altitude for trajectory design.

    isaac/run_isaac.sh isaac/probe_scene.py --env "Full Warehouse" --out results/g1/scenes

Occupancy comes from PhysX overlap queries (spheres of radius --clearance on a grid at each
altitude), so it reflects colliders, not just render meshes.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--clearance", type=float, default=0.8)
parser.add_argument("--cell", type=float, default=0.5)
parser.add_argument("--alts", default="1.5,2.5,3.5,5.0")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

pg = PegasusInterface()
pg._world_settings = {"physics_dt": 1.0 / 200.0, "stage_units_in_meters": 1.0,
                      "rendering_dt": 0.05}
pg._world = World(**pg._world_settings)
pg.load_environment(SIMULATION_ENVIRONMENTS[args.env])
while is_stage_loading():
    app.update()
world = pg.world
world.reset()
omni.timeline.get_timeline_interface().play()
for _ in range(5):
    app.update()
stage = omni.usd.get_context().get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
query = omni.physx.get_physx_scene_query_interface()
xs = np.arange(max(lo[0], -60), min(hi[0], 60), args.cell)
ys = np.arange(max(lo[1], -60), min(hi[1], 60), args.cell)
grids = {}
for alt in [float(a) for a in args.alts.split(",")]:
    occ = np.zeros((len(ys), len(xs)), bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            hits = []
            query.overlap_sphere(args.clearance, (float(x), float(y), alt),
                                 lambda h: hits.append(1) or True, False)
            occ[j, i] = bool(hits)
    grids[alt] = occ
np.savez_compressed(out / f"{args.env.replace(' ', '_')}.npz", xs=xs, ys=ys,
                    **{f"alt_{a}": g for a, g in grids.items()})
summary = {"env": args.env, "bbox_min": lo.round(2).tolist(), "bbox_max": hi.round(2).tolist(),
           "grid_x": [float(xs[0]), float(xs[-1])], "grid_y": [float(ys[0]), float(ys[-1])],
           "free_fraction": {str(a): float(1 - g.mean()) for a, g in grids.items()}}
(out / f"{args.env.replace(' ', '_')}.json").write_text(json.dumps(summary, indent=2))
print("PROBE " + json.dumps(summary), flush=True)
for alt, g in grids.items():
    print(f"--- alt {alt} m  (# = occupied, x right, y up, cell {args.cell} m)")
    step = max(1, int(round(2.0 / args.cell)))
    for row in g[::-step]:
        print("".join("#" if v else "." for v in row[::step]))
omni.timeline.get_timeline_interface().stop()
app.close()
