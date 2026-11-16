"""IJCAS rig: one run of one scenario on Isaac Sim 5.1 + Pegasus 5.1.0 + PX4 v1.14.3 (Stage A).

    isaac/run_isaac.sh isaac/ijcas_world.py --scenario config/scenarios/S00.yaml --scene warehouse \
        --seed 1 --out $IJCAS_DATA/runs/S00_warehouse_1

Physics 200 Hz, one render per 50 ms (stereo 20 Hz, LiDAR sectors 20 Hz -> scans 10 Hz).  The
vehicle follows the scene's reference loop (forced motion), PX4 runs as an estimator, every sensor
sample goes to PX4 and to ROS 2 unchanged except by the scenario's fault schedule.  Writes
manifest.json, gt.npz, px4_mavlink.json, faults.json into --out.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--duration", type=float, default=None, help="override scenario duration")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--no-px4", action="store_true")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import yaml  # noqa: E402

scenario = yaml.safe_load(Path(args.scenario).read_text())
scenes = yaml.safe_load((ROOT / "config/scenes.yaml").read_text())["scenes"]
scene_cfg = scenes[args.scene]
duration = float(args.duration or scenario["duration_s"])

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui})
import carb  # noqa: E402

_settings = carb.settings.get_settings()
for key in ("/app/asyncRendering", "/app/asyncRenderingLowLatency",
            "/omni/replicator/asyncRendering"):
    _settings.set(key, False)
_settings.set("/app/hydraEngine/waitIdle", True)
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackendConfig  # noqa
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ijcas_sim.degrade import DelayQueue, degrade_image, degrade_scan  # noqa: E402
from ijcas_sim.faults import FaultSchedule  # noqa: E402
from ijcas_sim.px4 import Px4Keeper, Px4Params, exact_time_backend_class, make_launch_tool  # noqa
from ijcas_sim.rig import Rig  # noqa: E402
from ijcas_sim.rosio import RosIO  # noqa: E402
from ijcas_sim.scenes import build_scene  # noqa: E402
from ijcas_sim.sensors import (GatedBarometer, GatedMagnetometer, IjcasGnss,  # noqa: E402
                               IjcasImu)
from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402

PHYS_DT = 1.0 / 200.0
RENDER_DT = 0.05
NED2ENU = Rotation.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
FRD2FLU = Rotation.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# --- world, scene, trajectory -------------------------------------------------------------
pg = PegasusInterface()
pg._world_settings = {"physics_dt": PHYS_DT, "stage_units_in_meters": 1.0,
                      "rendering_dt": RENDER_DT}
pg._world = World(**pg._world_settings)
world = pg.world
tcfg = scene_cfg["trajectory"]
traj = LoopTrajectory(tcfg["waypoints"], speed=float(tcfg["speed_mps"]),
                      hover_s=float(scenario.get("hover_s", 8.0)),
                      ramp_s=float(scenario.get("ramp_s", 4.0)),
                      yaw_mode=tcfg.get("yaw", "heading"),
                      corner_radius_m=float(tcfg.get("corner_radius_m", 0.0)))
path_xy = np.array([traj._path(s)[:2] for s in np.linspace(0, traj.length, 600)])
scene_info = build_scene(pg, app, args.scene, scene_cfg, args.seed, path_xy=path_xy)

# --- sensors, PX4, vehicle -------------------------------------------------------------------
np.random.seed(args.seed)                     # Pegasus baro/mag noise (global RNG)
rng = np.random.default_rng(args.seed)
rng_deg = np.random.default_rng(args.seed + 100_000)
faults = FaultSchedule.from_config(scenario.get("faults", []))
holder = {}
clock = lambda: holder["vehicle"].state_time  # noqa: E731
imu = IjcasImu(200.0, rng, faults, clock)
gnss = IjcasGnss(float(scenario.get("gnss_rate_hz", 5.0)), rng, faults, clock)
baro = GatedBarometer(50.0, faults, clock)
mag = GatedMagnetometer(50.0, faults, clock)
backends = []
backend = None
params = Px4Params()
if not args.no_px4:
    backend = exact_time_backend_class()(PX4MavlinkBackendConfig({
        "vehicle_id": 0, "px4_autolaunch": True, "px4_dir": os.environ["PX4_DIR"],
        "px4_vehicle_model": "gazebo-classic_iris", "update_rate": 200.0}), imu)
    backend.px4_tool = make_launch_tool(os.environ["PX4_DIR"], params)
    backends.append(backend)
mcfg = MultirotorConfig()
mcfg.sensors = [baro, imu, mag, gnss]
mcfg.backends = backends
mcfg.graphical_sensors = []
s0 = traj.sample(0.0)
from ijcas_sim.vehicle import ForcedMultirotor  # noqa: E402

vehicle = ForcedMultirotor("/World/quadrotor", ROBOTS["Iris"], 0, list(s0.position),
                           list(s0.quat_xyzw), config=mcfg, trajectory=traj)
holder["vehicle"] = vehicle

rig = Rig(body_path="/World/quadrotor/body")
rig.build()
world.reset()
rig.initialize()
for _ in range(3):
    app.update()
calibration = rig.calibrate()

# --- ROS ----------------------------------------------------------------------------------------
ros = RosIO()
static = [{"parent": "base_link", "child": "imu_link", "xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]}]
for name, c in calibration["cameras"].items():
    q = Rotation.from_matrix(np.asarray(c["R_body_optical"])).as_quat()
    static.append({"parent": "base_link", "child": f"{name}_optical",
                   "xyz": c["p_body_optical"], "quat_xyzw": q.tolist()})
ql = Rotation.from_matrix(np.asarray(calibration["lidar"]["R_body_lidar"])).as_quat()
static.append({"parent": "base_link", "child": "lidar", "xyz": calibration["lidar"]["p_body_lidar"],
               "quat_xyzw": ql.tolist()})
ros.static_transforms(static)
meta = {"scenario": scenario, "scene": args.scene, "scene_info": scene_info, "seed": args.seed,
        "duration_s": duration, "calibration": calibration, "trajectory": tcfg,
        "trajectory_length_m": traj.length, "px4_params": params.values,
        "origin": {"lat": pg.latitude, "lon": pg.longitude, "alt": pg.altitude}}
ros.json("meta", meta)

gt_rows = []
state = {"last_imu": 0, "last_gnss": 0, "last_baro": None, "last_mag": None, "step": 0,
         "faults_key": None}


PROF = {"physics_cb": 0.0, "world_step": 0.0, "images": 0.0, "scan": 0.0, "px4": 0.0}


def on_physics(dt):
    _p0 = time.perf_counter()
    t = vehicle.state_time
    ros.clock(t)
    if imu.count != state["last_imu"] and imu.last is not None:
        state["last_imu"] = imu.count
        stamp_t = imu.last["t"]
        e = faults.first("imu", "time_offset", stamp_t)
        if e is not None:
            stamp_t += float(e.params["offset_s"])
        ros.imu(stamp_t, imu.last["gyro_flu"], imu.last["accel_flu"], imu.noise)
    if gnss.count != state["last_gnss"] and gnss.last is not None:
        state["last_gnss"] = gnss.count
        g = gnss.last
        d = sum(max(0.0, float(e.params["mean_s"]) + float(e.params["jitter_s"])
                    * rng_deg.standard_normal()) for e in faults.active("gnss_ros", g["t"]))
        gnss_queue.push(g["t"] + d, g)
    for g in gnss_queue.pop_due(t):
        ros.gnss(g["t"], g)
    if baro.last is not None and baro.last is not state["last_baro"]:
        state["last_baro"] = baro.last
        ros.baro(baro.last["t"], baro.last)
    if mag.last is not None and mag.last is not state["last_mag"]:
        state["last_mag"] = mag.last
        ros.mag(mag.last["t"], mag.last)
    state["step"] += 1
    if state["step"] % 4 == 0:
        st = vehicle.state
        ros.odometry("gt_odom", t, "map", "base_link_gt", st.position, st.attitude,
                     st.linear_velocity, st.angular_velocity)
        gt_rows.append([t, *st.position, *st.attitude, *st.linear_velocity,
                        *st.angular_velocity])
    PROF["physics_cb"] += time.perf_counter() - _p0


gnss_queue = DelayQueue()
cam_queue = DelayQueue()
lidar_queue = DelayQueue()
world.add_physics_callback("ijcas_publish", on_physics)

keeper = None
if backend is not None:
    backend.px4_tool.launch_px4()
    keeper = Px4Keeper(instance=0, hover_thrust=0.6, start_after_s=3.0)
px4_published = {"lpos": 0, "est": 0}


def publish_px4():
    rec = keeper.records
    att = rec["att"]
    for row in rec["lpos"][px4_published["lpos"]:]:
        tb = row[0]
        if not att:
            break
        a = min(att[-50:], key=lambda r: abs(r[0] - tb))
        q_ned = Rotation.from_quat([a[2], a[3], a[4], a[1]])
        q_enu = (NED2ENU * q_ned * FRD2FLU).as_quat()
        ros.odometry("px4_odom", tb, "px4_local_enu", "base_link_px4",
                     [row[2], row[1], -row[3]], q_enu, [row[5], row[4], -row[6]])
    px4_published["lpos"] = len(rec["lpos"])
    for row in rec["est"][px4_published["est"]:]:
        ros.json("px4_status", {"t": row[0], "flags": row[1], "vel_ratio": row[2],
                                "pos_horiz_ratio": row[3], "pos_vert_ratio": row[4],
                                "mag_ratio": row[5], "hagl_ratio": row[6],
                                "pos_horiz_accuracy": row[8], "pos_vert_accuracy": row[9]})
    px4_published["est"] = len(rec["est"])


omni.timeline.get_timeline_interface().play()
steps = int(round(duration / RENDER_DT))
w0 = time.time()
error = None
try:
    for k in range(steps):
        _p = time.perf_counter()
        world.step(render=True)
        PROF["world_step"] += time.perf_counter() - _p
        now = vehicle.state_time
        _p = time.perf_counter()
        if keeper is not None:
            keeper.tick(now)
            publish_px4()
        PROF["px4"] += time.perf_counter() - _p
        _p = time.perf_counter()
        for name, rt, gray in rig.new_images():
            img, delay = degrade_image(gray, name, rt, faults, rng_deg)
            if img is not None:
                cam_queue.push(rt + delay, (name, rt, img))
        for name, rt, img in cam_queue.pop_due(now):
            c = calibration["cameras"][name]
            ros.image(name, rt, img, f"{name}_optical")
            ros.camera_info(f"{name}_info", rt, f"{name}_optical", np.asarray(c["K"]),
                            c["width"], c["height"])
        PROF["images"] += time.perf_counter() - _p
        _p = time.perf_counter()
        scan = rig.new_scan()
        if scan is not None:
            st, xyz, inten, rel = scan
            res, delay = degrade_scan(xyz, inten, rel, st, faults, rng_deg)
            if res is not None:
                lidar_queue.push(st + delay, (st, *res))
        for st, xyz, inten, rel in lidar_queue.pop_due(now):
            ros.points(st, xyz, inten, rel)
        PROF["scan"] += time.perf_counter() - _p
        labels = faults.labels(now)
        key = json.dumps(labels, sort_keys=True)
        if key != state["faults_key"]:
            state["faults_key"] = key
            ros.json("gt_faults", {"t": now, "active": labels})
except Exception as exc:  # recorded, then the stack is torn down
    error = f"{type(exc).__name__}: {exc}"
    raise
finally:
    wall = time.time() - w0
    sim_s = vehicle.state_time
    manifest = {**meta, "sim_s": sim_s, "wall_s": wall, "rtf": sim_s / max(wall, 1e-6),
                "error": error, "ros_counts": ros.counts,
                "profile_s": {**PROF, "vehicle_state": vehicle.prof_s},
                "sensor_counts": {"imu": imu.count, "gnss": gnss.count},
                "versions": {"isaac_sim": "5.1.0 (47d886f2)", "pegasus": "v5.1.0 (644da37e)",
                             "px4": "v1.14.3 (1dacb4cd)", "ros": "humble",
                             "python": sys.version.split()[0]}}
    if keeper is not None:
        manifest["px4"] = keeper.summary()
        manifest["px4_param_readback"] = {n: keeper.read_param(n, 1.0) for n in
                                          ("COM_RC_IN_MODE", "EKF2_EV_CTRL", "EKF2_GPS_CTRL",
                                           "EKF2_HGT_REF", "UXRCE_DDS_DOM_ID")}
        (out / "px4_mavlink.json").write_text(json.dumps(keeper.records))
    np.savez_compressed(out / "gt.npz", gt=np.asarray(gt_rows),
                        columns=np.array("t px py pz qx qy qz qw vx vy vz wx wy wz".split()))
    (out / "faults.json").write_text(json.dumps(faults.describe(), indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("IJCAS_RUN " + json.dumps({"sim_s": round(sim_s, 2), "rtf": round(sim_s / max(wall, 1e-6), 3),
                                     "error": error, "counts": ros.counts}), flush=True)
    omni.timeline.get_timeline_interface().stop()
    if backend is not None:
        backend.stop()
        if backend.px4_tool is not None:
            backend.px4_tool.kill_px4()
    ros.close()
    app.close()
