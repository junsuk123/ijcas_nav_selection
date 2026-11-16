"""G1: capture samples of every rig topic during a run (ROS 2 node, system Python).

    python3 tools/g1_capture.py --seconds 60 --out $IJCAS_DATA/g1/capture.npz
Saves IMU/GNSS/GT series, camera_info, a handful of synchronised stereo pairs and LiDAR scans, and
the IMU message orientation fields (leak check).
"""
import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, PointCloud2

QOS = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)


def t_of(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class Capture(Node):
    def __init__(self, n_pairs, n_scans):
        super().__init__("ijcas_g1_capture")
        self.imu, self.imu_orient, self.fix, self.vel, self.gt = [], [], [], [], []
        self.info = {}
        self.imgs = {"cam0": {}, "cam1": {}}
        self.pairs, self.scans, self.n_pairs, self.n_scans = [], [], n_pairs, n_scans
        self.scan_stamps = []
        s = self.create_subscription
        s(Imu, "/ijcas/imu", self.on_imu, QOS)
        s(NavSatFix, "/ijcas/gnss/fix", lambda m: self.fix.append(
            [t_of(m.header), m.status.status, m.latitude, m.longitude, m.altitude]), QOS)
        s(TwistWithCovarianceStamped, "/ijcas/gnss/vel", lambda m: self.vel.append(
            [t_of(m.header), m.twist.twist.linear.x, m.twist.twist.linear.y,
             m.twist.twist.linear.z]), QOS)
        s(Odometry, "/ijcas/gt/odom", self.on_gt, QOS)
        for c in ("cam0", "cam1"):
            s(CameraInfo, f"/{c}/camera_info", lambda m, c=c: self.info.setdefault(
                c, {"K": list(m.k), "w": m.width, "h": m.height, "frame": m.header.frame_id}), 10)
            s(Image, f"/{c}/image_raw", lambda m, c=c: self.on_img(c, m), 10)
        s(PointCloud2, "/ijcas/lidar/points", self.on_scan, 10)

    def on_imu(self, m):
        self.imu.append([t_of(m.header), m.angular_velocity.x, m.angular_velocity.y,
                         m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y,
                         m.linear_acceleration.z])
        self.imu_orient.append([m.orientation.x, m.orientation.y, m.orientation.z,
                                m.orientation.w, m.orientation_covariance[0]])

    def on_gt(self, m):
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        self.gt.append([t_of(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z])

    def on_img(self, c, m):
        t = t_of(m.header)
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
        self.imgs[c][round(t, 4)] = img
        other = "cam1" if c == "cam0" else "cam0"
        if round(t, 4) in self.imgs[other] and len(self.pairs) < self.n_pairs and \
                (not self.pairs or t - self.pairs[-1][0] > 3.0):
            a, b = (img, self.imgs[other][round(t, 4)]) if c == "cam0" else \
                (self.imgs[other][round(t, 4)], img)
            self.pairs.append((t, a.copy(), b.copy()))
        for d in self.imgs.values():
            for k in [k for k in d if k < t - 1.0]:
                del d[k]

    def on_scan(self, m):
        self.scan_stamps.append(t_of(m.header))
        if t_of(m.header) < 14.0 or len(self.scans) >= self.n_scans or \
                (self.scans and t_of(m.header) - self.scans[-1][0] < 2.0):
            return
        arr = np.frombuffer(m.data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("intensity", "<f4"), ("t", "<f4")])
        self.scans.append((t_of(m.header), np.array(arr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--scans", type=int, default=10)
    a = ap.parse_args()
    rclpy.init()
    n = Capture(a.pairs, a.scans)
    end = time.time() + a.seconds
    try:
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
    except BaseException:
        pass
    np.savez_compressed(
        a.out, imu=np.array(n.imu), imu_orient=np.array(n.imu_orient), fix=np.array(n.fix),
        vel=np.array(n.vel), gt=np.array(n.gt),
        pair_t=np.array([p[0] for p in n.pairs]),
        pair_l=np.array([p[1] for p in n.pairs]) if n.pairs else np.zeros(0),
        pair_r=np.array([p[2] for p in n.pairs]) if n.pairs else np.zeros(0),
        scan_t=np.array([s[0] for s in n.scans]), scan_stamps=np.array(n.scan_stamps),
        scans=np.array([s[1] for s in n.scans], dtype=object), info=np.array([n.info], dtype=object))
    print("captured imu", len(n.imu), "fix", len(n.fix), "gt", len(n.gt), "pairs", len(n.pairs),
          "scans", len(n.scans), flush=True)


if __name__ == "__main__":
    main()
