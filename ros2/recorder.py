"""Run recorder (ROS 2, system Python, use_sim_time): every candidate output and diagnostic with
its header stamp *and* the simulation time at reception (for output age / latency).

    python3 ros2/recorder.py --out $RUN/record.npz --seconds 600

Saved arrays (one row per message):
  odom/<name>   stamp recv px py pz qx qy qz qw vx vy vz  pose-cov diag 6 | pose cov 36 | twist cov 36
  imu           stamp ax..  (for the INS-DR baseline and the context monitor)
  fix           stamp recv status lat lon alt cov_h cov_v
  gvel          stamp recv vx vy vz
  json/<topic>  list of (recv, payload) for String topics
Images and point clouds are not recorded (disk); the engines consume them online and the
context monitor records per-frame diagnostics instead.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String

REL = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
BE = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
ODOM = {"openvins": "/ov_msckf/odomimu", "fastlio": "/fast_lio/odom", "px4": "/ijcas/px4/odom",
        "gt": "/ijcas/gt/odom", "selected": "/ijcas/nav/selected"}
STRINGS = ["/ijcas/px4/status", "/ijcas/gt/faults", "/ijcas/run/meta", "/ijcas/diag/vision",
           "/ijcas/diag/lidar", "/ijcas/diag/gnss", "/ijcas/diag/imu", "/ijcas/nav/decision"]


def st(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class Recorder(Node):
    def __init__(self):
        super().__init__("ijcas_recorder",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.odom = {k: [] for k in ODOM}
        self.imu, self.fix, self.gvel = [], [], []
        self.json = {t: [] for t in STRINGS}
        self.first_wall = None
        for name, topic in ODOM.items():  # best effort accepts reliable and best-effort publishers
            self.create_subscription(Odometry, topic, lambda m, n=name: self.on_odom(n, m), BE)
        self.create_subscription(PoseWithCovarianceStamped, "/ov_msckf/poseimu",
                                 self.on_ov_pose, REL)
        self.ov_cov = []
        self.create_subscription(Imu, "/ijcas/imu", self.on_imu, REL)
        self.create_subscription(NavSatFix, "/ijcas/gnss/fix", self.on_fix, REL)
        self.create_subscription(TwistWithCovarianceStamped, "/ijcas/gnss/vel", self.on_vel, REL)
        for t in STRINGS:
            qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
            if t in ("/ijcas/gt/faults", "/ijcas/run/meta"):
                from rclpy.qos import DurabilityPolicy
                qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(String, t, lambda m, t=t: self.on_json(t, m), qos)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, name, m):
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        c = m.pose.covariance
        # columns 0-17 unchanged (readers of older records); 18-53 full 6x6 pose covariance and 54-89 full
        # 6x6 twist covariance, row-major, exactly as published by the estimator (2026-11-18, fusion)
        self.odom[name].append([st(m.header), self.now_s(), p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                                v.x, v.y, v.z, c[0], c[7], c[14], c[21], c[28], c[35],
                                *[float(x) for x in c], *[float(x) for x in m.twist.covariance]])

    def on_ov_pose(self, m):
        c = m.pose.covariance
        self.ov_cov.append([st(m.header), self.now_s(), c[0], c[7], c[14], c[21], c[28], c[35]])

    def on_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu.append([st(m.header), w.x, w.y, w.z, a.x, a.y, a.z])

    def on_fix(self, m):
        self.fix.append([st(m.header), self.now_s(), m.status.status, m.latitude, m.longitude,
                         m.altitude, m.position_covariance[0], m.position_covariance[8]])

    def on_vel(self, m):
        v = m.twist.twist.linear
        self.gvel.append([st(m.header), self.now_s(), v.x, v.y, v.z])

    def on_json(self, topic, m):
        try:
            payload = json.loads(m.data)
        except ValueError:
            payload = m.data
        self.json[topic].append([self.now_s(), payload])

    def save(self, path):
        arrays = {f"odom_{k}": np.asarray(v, float) for k, v in self.odom.items()}
        arrays.update(imu=np.asarray(self.imu, float), fix=np.asarray(self.fix, float),
                      gvel=np.asarray(self.gvel, float), ov_pose_cov=np.asarray(self.ov_cov, float))
        np.savez_compressed(path, **arrays)
        with open(str(path).replace(".npz", "_json.json"), "w") as f:
            json.dump(self.json, f)
        return {k: len(v) for k, v in arrays.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=900)
    a = ap.parse_args()
    rclpy.init()
    node = Recorder()
    end = time.time() + a.seconds
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except BaseException:
        pass
    counts = node.save(a.out)
    print("RECORDED " + json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
