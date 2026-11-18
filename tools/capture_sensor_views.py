"""Grab single sensor frames from a running IJCAS simulation for documentation figures.

    source env/ijcas_env.sh
    python3 tools/capture_sensor_views.py --out <dir> [--minutes 60] [--max-runs 12]

For every run that starts while this tool is up it reads /ijcas/run/meta and, at fixed simulation times
(normal flight at 30 s, and 6 s into each camera / LiDAR fault), subscribes once to the camera,
LiDAR and ground-truth topics, keeps the first message of each and drops the subscriptions again, so
the running estimators see no extra load between captures.  Only OUTPUT topics of the simulator are
read (no estimator debug topics: OpenVINS draws /ov_msckf/trackhist only when someone subscribes).
Each capture is one <run>_<tag>.npz (cam0, cam1 mono8, LiDAR x y z intensity in the sensor frame,
FAST-LIO registered scan in its map frame, ground-truth pose) plus run meta; nothing is recorded
continuously.  The figures are drawn by tools/make_doc_figures.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

_PF = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
GRAB = {"cam0": ("/cam0/image_raw", Image), "cam1": ("/cam1/image_raw", Image),
        "lidar": ("/ijcas/lidar/points", PointCloud2), "lio_scan": ("/cloud_registered", PointCloud2),
        "gt": ("/ijcas/gt/odom", Odometry)}


def image_array(m: Image) -> np.ndarray:
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)
    if m.encoding == "mono8":
        return a[:, :m.width].copy()
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[m.encoding]
    a = a[:, :m.width * ch].reshape(m.height, m.width, ch)[..., :3]
    return (a[..., ::-1] if m.encoding.startswith("bgr") else a).copy()


def cloud_array(m: PointCloud2) -> np.ndarray:
    f = {x.name: x for x in m.fields}
    names = [n for n in ("x", "y", "z", "intensity") if n in f]
    dt = np.dtype({"names": names, "formats": [_PF[f[n].datatype] for n in names],
                   "offsets": [f[n].offset for n in names], "itemsize": m.point_step})
    a = np.frombuffer(bytes(m.data), dt, count=m.width * m.height)
    out = np.stack([a[n].astype(np.float32) for n in names], 1)
    return out[np.isfinite(out[:, :3]).all(1)]


class Capture(Node):
    def __init__(self, out: Path, max_runs: int):
        super().__init__("ijcas_doc_capture")
        self.out, self.max_runs = out, max_runs
        self.meta, self.run, self.plan, self.done_runs = None, None, [], []
        self.pending: dict = {}
        self.got: dict = {}
        self.tag = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/ijcas/run/meta", self.on_meta, latched)
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
        self.now = 0.0

    def on_meta(self, m: String):
        meta = json.loads(m.data)
        sc = meta.get("scenario", meta)
        run = f"{sc['id']}_{int(sc['seed']):03d}"
        if run == self.run:
            return
        self.run, self.meta = run, meta
        plan = [("normal", 30.0)]
        for f in sc.get("faults", []):
            if f.get("target") in ("camera", "lidar"):
                plan.append((f"{f['target']}_{f['kind']}", float(f["start_s"]) + 6.0))
        self.plan = sorted(set(plan), key=lambda x: x[1])
        self.get_logger().info(f"run {run} scene={sc.get('scene')} plan={self.plan}")

    def on_clock(self, m: Clock):
        self.now = m.clock.sec + m.clock.nanosec * 1e-9
        if self.pending or not self.plan or self.run is None:
            return
        tag, ts = self.plan[0]
        if self.now >= ts:
            self.plan.pop(0)
            if self.now > ts + 5.0:                              # joined mid-run: skip stale targets
                return
            self.tag, self.got = tag, {"t_req": self.now}
            for key, (topic, typ) in GRAB.items():
                self.pending[key] = self.create_subscription(
                    typ, topic, lambda msg, k=key: self.on_msg(k, msg), qos_profile_sensor_data)

    def on_msg(self, key, msg):
        sub = self.pending.pop(key, None)
        if sub is None:
            return
        self.destroy_subscription(sub)
        if isinstance(msg, Image):
            self.got[key] = image_array(msg)
        elif isinstance(msg, PointCloud2):
            self.got[key] = cloud_array(msg)
            self.got[key + "_frame"] = msg.header.frame_id
        else:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            self.got[key] = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        self.got[key + "_stamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.pending:
            self.finish()

    def finish(self):
        path = self.out / f"{self.run}_{self.tag}.npz"
        np.savez_compressed(path, meta=json.dumps(self.meta), **self.got)
        self.get_logger().info(f"saved {path.name} ({', '.join(k for k in GRAB if k in self.got)})")
        if not self.plan and self.run not in self.done_runs:
            self.done_runs.append(self.run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-runs", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Capture(out, a.max_runs)
    t_end = time.time() + 60 * a.minutes
    try:
        while rclpy.ok() and time.time() < t_end and len(node.done_runs) < a.max_runs:
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.pending and node.now - node.got.get("t_req", node.now) > 10.0:   # topic silent
                for k, s in list(node.pending.items()):
                    node.destroy_subscription(s)
                    node.get_logger().warn(f"{node.run}_{node.tag}: no {k}")
                node.pending.clear()
                node.finish()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
