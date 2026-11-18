"""Third-person documentation renders of the IJCAS scenes (not used by any experiment).

    for s in warehouse open_field corridor; do isaac/run_isaac.sh isaac/render_overview.py --scene $s --out <dir>; done

Builds each scene exactly as isaac/ijcas_world.py does (same build_scene, same seed, same lights), puts
the Iris model on the reference trajectory, marks the reference loop with small spheres and renders
1600x900 RGB images from a chase camera and an overview camera (isaacsim.sensors.camera.Camera, as the
rig does).  The corridor overview hides the ceiling (cut-away); the file name says so.  No physics,
PX4, ROS or sensors are started.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--scene", required=True,
                    choices=("warehouse", "open_field", "corridor", "urban", "solar_farm", "tunnel_road"))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--warmup", type=int, default=40, help="render updates before each capture")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import yaml  # noqa: E402

SCENES = yaml.safe_load((ROOT / "config/scenes.yaml").read_text())["scenes"]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
import carb  # noqa: E402

_s = carb.settings.get_settings()
for key in ("/app/asyncRendering", "/app/asyncRenderingLowLatency", "/omni/replicator/asyncRendering"):
    _s.set(key, False)

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ijcas_sim.scenes import build_scene  # noqa: E402
from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402


def look_quat_wxyz(eye, target):
    """Orientation (w, x, y, z) of a 'world'-axes camera (x forward, z up) at eye looking at target."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.hypot(d[0], d[1]))
    x, y, z, w = Rotation.from_euler("ZY", [yaw, pitch]).as_quat()
    return np.array([w, x, y, z])


def mark_path(stage, pts, color=(0.95, 0.45, 0.05), r=0.07):
    """Reference loop as small spheres (rendered by RTX without a material)."""
    for i, p in enumerate(pts):
        sp = UsdGeom.Sphere.Define(stage, f"/World/DocPath/p{i}")
        sp.CreateRadiusAttr(r)
        sp.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))


def render(name, eye, target, cams):
    cam = cams.get("cam")
    if cam is None:
        cam = Camera(prim_path="/World/DocCam", resolution=(args.width, args.height))
        cam.initialize()
        usd = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/World/DocCam"))
        usd.GetFocalLengthAttr().Set(14.0)                           # ~74 deg horizontal FOV
        usd.GetHorizontalApertureAttr().Set(20.955)
        usd.GetVerticalApertureAttr().Set(20.955 * args.height / args.width)
        usd.GetClippingRangeAttr().Set((0.05, 1000.0))
        cams["cam"] = cam
    cam.set_world_pose(position=np.asarray(eye, float), orientation=look_quat_wxyz(eye, target),
                       camera_axes="world")
    for _ in range(args.warmup):
        app.update()
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        print(f"[render] {name}: empty frame", flush=True)
        return
    img = np.asarray(rgba)[:, :, :3].astype(np.uint8)
    Image.fromarray(img).save(out / f"{name}.png")
    print(f"[render] {name}.png mean={img.mean():.1f}", flush=True)


VIEWS = {   # scene -> chase sample time t [s after hover + ramp], fixed views (name suffix, eye, target)
    "warehouse": {"t": 14.0, "views": [("overview", (4.0, -38.0, 7.5), (-12.0, 5.0, 0.5))]},  # roof ~9.3 m
    "open_field": {"t": 18.0, "views": [("overview", (-40.0, -45.0, 32.0), (15.0, 12.0, 0.0))]},
    "corridor": {"t": 10.0, "views": [("overview_ceiling_hidden", (0.0, -38.0, 42.0), (0.0, 0.0, 0.0))],
                 "hide": "/World/Scene/ceiling"},
    "urban": {"t": 20.0, "views": [("overview", (-150.0, -170.0, 120.0), (0.0, 0.0, 10.0)),
                                   ("street", (56.0, -85.0, 9.0), (56.0, 40.0, 14.0))]},
    "solar_farm": {"t": 10.0, "views": [("overview", (-110.0, -80.0, 40.0), (20.0, 10.0, 0.0))]},
    "tunnel_road": {"t": 40.0, "views": [("portal", (-330.0, 60.0, 45.0), (-200.0, 0.0, 5.0)),
                                         ("overview", (-520.0, -260.0, 220.0), (0.0, 0.0, 0.0))]},
}

if True:
    scene = args.scene
    cfg = SCENES[scene]
    pg = PegasusInterface()
    pg._world_settings = {"physics_dt": 1.0 / 200.0, "stage_units_in_meters": 1.0, "rendering_dt": 0.05}
    pg._world = World(**pg._world_settings)
    tc = cfg["trajectory"]
    traj = LoopTrajectory(tc["waypoints"], speed=float(tc["speed_mps"]), hover_s=8.0, ramp_s=4.0,
                          yaw_mode=tc.get("yaw", "heading"), corner_radius_m=float(tc.get("corner_radius_m", 0.0)))
    path = np.array([traj._path(s)[:3] for s in np.linspace(0, traj.length, 600)])
    build_scene(pg, app, scene, cfg, args.seed, path_xy=path[:, :2])
    stage = omni.usd.get_context().get_stage()
    v = VIEWS[scene]
    st = traj.sample(8.0 + 4.0 + v["t"])
    p = np.asarray(st.position, float)
    add_reference_to_stage(ROBOTS["Iris"], "/World/doc_iris")
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/doc_iris"))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*p))
    qx, qy, qz, qw = st.quat_xyzw
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))
    step = max(1, int(len(path) / max(1.0, traj.length / 1.2)))
    mark_path(stage, path[::step])
    pg.world.reset()
    pg.world.pause()                                   # render only: no physics, the model stays put
    cams = {}
    heading = Rotation.from_quat(st.quat_xyzw).apply([1.0, 0.0, 0.0])
    heading[2] = 0.0
    heading /= np.linalg.norm(heading) + 1e-9
    render(f"{scene}_chase", p - 4.5 * heading + np.array([0, 0, 1.4]), p + 3.0 * heading, cams)
    if "hide" in v:
        UsdGeom.Imageable(stage.GetPrimAtPath(v["hide"])).MakeInvisible()
    for suffix, eye, target in v["views"]:
        render(f"{scene}_{suffix}", eye, target, cams)

app.close()
