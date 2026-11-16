"""G1 probe: RTX LiDAR (Ouster OS1 32ch 10 Hz) frame, timing and coverage.

A 1 m thick wall is placed 6 m in front (+x) of a static LiDAR at 2 m height; the probe dumps the
scan-buffer fields so we know (1) the point frame (expect sensor frame, x forward, y left, z up),
(2) what ``timestamp`` means (absolute sim ns? span per buffer?), (3) azimuth coverage per buffer.
Then the LiDAR rotates in yaw at 1 rad/s to see whether points within one buffer share a pose.

    isaac/run_isaac.sh isaac/probe_lidar.py --out results/g1/lidar_probe
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--sync", action="store_true", help="disable asynchronous rendering")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
if args.sync:
    import carb
    _s = carb.settings.get_settings()
    for key in ("/app/asyncRendering", "/app/asyncRenderingLowLatency",
                "/omni/replicator/asyncRendering"):
        _s.set(key, False)
    _s.set("/app/hydraEngine/waitIdle", True)

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
from isaacsim.core.prims import SingleXFormPrim  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.sensors.rtx import LidarRtx  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

world = World(physics_dt=1.0 / 200.0, rendering_dt=0.05, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
FixedCuboid("/World/wall", position=np.array([6.5, 0.0, 2.0]), scale=np.array([1.0, 8.0, 4.0]))
mount = SingleXFormPrim("/World/mount", position=np.array([0.0, 0.0, 2.0]))
lidar = LidarRtx(prim_path="/World/mount/lidar", config_file_name="OS1_REV6_32ch10hz1024res")
cam = Camera(prim_path="/World/mount/cam", resolution=(320, 240), frequency=20)
world.reset()
cam.initialize()
lidar.initialize()
lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
omni.timeline.get_timeline_interface().play()
scans = []
yaw = 0.0
for k in range(80):
    rotating = k >= 40
    if rotating:
        yaw += 1.0 * 0.05
        q = Rotation.from_euler("z", yaw).as_quat()
        mount.set_world_pose(np.array([0.0, 0.0, 2.0]), np.array([q[3], q[0], q[1], q[2]]))
    world.step(render=True)
    sim_t = world.current_time
    frame = lidar.get_current_frame()
    buf = frame.get("IsaacExtractRTXSensorPointCloudNoAccumulator") if frame else None
    if buf is None or not len(buf.get("data", [])):
        continue
    pts = np.asarray(buf["data"], float)
    ts = np.asarray(buf.get("timestamp", []), np.float64)
    if ts.size == 0:
        ts = np.zeros(1)
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    em = np.asarray(buf.get("emitterId", []))
    front = (np.abs(np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))) < 10) & (np.abs(pts[:, 2]) < 0.5)
    scans.append({
        "k": k, "rotating": rotating, "sim_time": sim_t, "yaw": yaw, "n": int(len(pts)),
        "ts_min_ns": float(ts.min()), "ts_max_ns": float(ts.max()),
        "ts_span_ms": float((ts.max() - ts.min()) * 1e-6),
        "az_min_deg": float(az.min()), "az_max_deg": float(az.max()),
        "az_hist": np.histogram(az, bins=8, range=(-180, 180))[0].tolist(),
        "rendering_time": float(frame.get("rendering_time", -1)),
        "camera_rendering_time": float(cam.get_current_frame().get("rendering_time", -1)),
        "emitters": int(len(np.unique(em))) if len(em) else None,
        "front_wall_x_median": float(np.median(pts[front, 0])) if front.any() else None,
        "front_count": int(front.sum()),
        "z_range": [float(pts[:, 2].min()), float(pts[:, 2].max())],
        "frame_keys": sorted(k2 for k2 in frame.keys()),
        "buffer_keys": sorted(buf.keys()),
        "info_keys": sorted((buf.get("info") or {}).keys()) if isinstance(buf.get("info"), dict) else None,
    })
    if len(scans) in (3, 30):
        np.savez_compressed(out / f"scan_{len(scans)}.npz", points=pts, timestamp=ts, azimuth=az,
                            emitter=em)
(out / "scans.json").write_text(json.dumps(scans, indent=1))
print("PROBE scans", len(scans), json.dumps(scans[2], indent=None)[:800], flush=True)
omni.timeline.get_timeline_interface().stop()
app.close()
