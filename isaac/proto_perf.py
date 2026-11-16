"""G1 feasibility probe: can this PC render the IJCAS sensor rig fast enough?

Pegasus Iris (no backends, gravity off, velocity-driven circle), physics 200 Hz, one render per
50 ms (stereo 640x480 @ 20 Hz), RTX LiDAR Ouster OS1 32ch 10 Hz with a scan buffer.  Prints the
real-time factor and what the sensors returned.

    $ISAACSIM_PATH/python.sh isaac/proto_perf.py --env "Full Warehouse" --seconds 20
"""
import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Full Warehouse")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--no-lidar", action="store_true")
parser.add_argument("--no-cameras", action="store_true")
parser.add_argument("--out", default="")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.sensors.rtx import LidarRtx  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

pg = PegasusInterface()
pg._world_settings = {"physics_dt": 1.0 / 200.0, "stage_units_in_meters": 1.0,
                      "rendering_dt": 1.0 / 20.0}
pg._world = World(**pg._world_settings)
world = pg.world
t0 = time.time()
pg.load_environment(SIMULATION_ENVIRONMENTS[args.env])
while is_stage_loading():
    app.update()
load_s = time.time() - t0

cfg = MultirotorConfig()
cfg.backends = []
cfg.sensors = []
vehicle = Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0.0, 0.0, 2.0],
                     [0.0, 0.0, 0.0, 1.0], config=cfg)
body_path = "/World/quadrotor/body"
stage = world.stage
PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(body_path)).CreateDisableGravityAttr(True)

cams = []
if not args.no_cameras:
    for name, y in (("cam0", 0.06), ("cam1", -0.06)):
        cam = Camera(prim_path=f"{body_path}/{name}", resolution=(640, 480), frequency=20)
        cam.set_local_pose(translation=np.array([0.15, y, 0.0]),
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="world")
        cams.append(cam)
lidar = None
if not args.no_lidar:
    lidar = LidarRtx(prim_path=f"{body_path}/lidar", translation=np.array([0.0, 0.0, 0.12]),
                     orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                     config_file_name="OS1_REV6_32ch10hz1024res")

world.reset()
for cam in cams:
    cam.initialize()
if lidar is not None:
    lidar.initialize()
    lidar.attach_annotator("IsaacCreateRTXLidarScanBuffer", outputTimestamp=True,
                           outputIntensity=True)

from isaacsim.core.prims import RigidPrim  # noqa: E402
body = RigidPrim(body_path)
omni.timeline.get_timeline_interface().play()
for _ in range(10):
    app.update()

steps = int(args.seconds / 0.05)
cam_frames, scans, points = 0, 0, []
ts_info = None
w0 = time.time()
for k in range(steps):
    t = k * 0.05
    w = 2 * np.pi / 20.0
    vel = np.array([[-3.0 * np.sin(w * t), 3.0 * np.cos(w * t), 0.0]])
    body.set_velocities(np.hstack([vel, [[0.0, 0.0, w]]]))
    world.step(render=True)
    for cam in cams:
        rgba = cam.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            cam_frames += 1
    if lidar is not None:
        frame = lidar.get_current_frame()
        buf = frame.get("IsaacCreateRTXLidarScanBuffer") if frame else None
        if buf is not None and len(buf.get("data", [])):
            scans += 1
            points.append(len(buf["data"]))
            if ts_info is None and "timestamp" in buf and len(buf["timestamp"]):
                ts = np.asarray(buf["timestamp"])
                ts_info = {"keys": sorted(buf.keys()), "ts_min": float(ts.min()),
                           "ts_max": float(ts.max()), "ts_dtype": str(ts.dtype)}
wall = time.time() - w0
pos, _ = body.get_world_poses()
result = {"env": args.env, "load_s": round(load_s, 1), "sim_s": steps * 0.05,
          "wall_s": round(wall, 2), "rtf": round(steps * 0.05 / wall, 3),
          "camera_frames": cam_frames, "lidar_scans_seen": scans,
          "points_per_scan_median": int(np.median(points)) if points else 0,
          "lidar_buffer": ts_info, "final_position": [round(float(v), 2) for v in pos[0]]}
print("PROBE " + json.dumps(result), flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
omni.timeline.get_timeline_interface().stop()
app.close()
