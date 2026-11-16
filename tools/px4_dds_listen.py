"""Listen to PX4 uXRCE-DDS topics (px4_msgs, release/1.14) and summarise them (G0/G1).

    source env/ijcas_env.sh
    python3 tools/px4_dds_listen.py --seconds 120 --out results/g0/px4_probe/dds.json

Records message counts/rates, land-detector flags over time, EKF2 status flags and the
vehicle_odometry trajectory (NED, PX4 local frame) for comparison with the probe's ground truth.
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from px4_msgs.msg import (EstimatorStatusFlags, SensorCombined, SensorGps, VehicleLandDetected,
                          VehicleOdometry, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST, depth=10)
TOPICS = {
    "vehicle_odometry": VehicleOdometry,
    "vehicle_land_detected": VehicleLandDetected,
    "vehicle_status": VehicleStatus,
    "estimator_status_flags": EstimatorStatusFlags,
    "sensor_combined": SensorCombined,
    "vehicle_gps_position": SensorGps,
}


class Listener(Node):
    def __init__(self):
        super().__init__("ijcas_px4_dds_listen")
        self.counts = {k: 0 for k in TOPICS}
        self.first = {}
        self.odom = []
        self.land = []
        self.status = []
        self.flags = []
        for name, kind in TOPICS.items():
            self.create_subscription(kind, f"/fmu/out/{name}",
                                     lambda m, n=name: self.on(n, m), QOS)

    def on(self, name, msg):
        now = time.time()
        self.counts[name] += 1
        self.first.setdefault(name, now)
        if name == "vehicle_odometry":
            self.odom.append([msg.timestamp * 1e-6, *map(float, msg.position), *map(float, msg.q),
                              *map(float, msg.velocity), *map(float, msg.position_variance),
                              int(msg.reset_counter)])
        elif name == "vehicle_land_detected":
            self.land.append([msg.timestamp * 1e-6, bool(msg.landed), bool(msg.maybe_landed),
                              bool(msg.ground_contact), bool(msg.at_rest)])
        elif name == "vehicle_status" and self.counts[name] % 10 == 1:
            self.status.append([msg.timestamp * 1e-6, int(msg.arming_state), int(msg.nav_state),
                                bool(msg.failsafe)])
        elif name == "estimator_status_flags" and self.counts[name] % 10 == 1:
            self.flags.append({"t": msg.timestamp * 1e-6, "gps": bool(msg.cs_gps),
                               "baro_hgt": bool(msg.cs_baro_hgt), "mag_3d": bool(msg.cs_mag_3d),
                               "in_air": bool(msg.cs_in_air), "dead_reckoning":
                               bool(msg.cs_inertial_dead_reckoning),
                               "fake_pos": bool(msg.cs_fake_pos), "tilt_align":
                               bool(msg.cs_tilt_align), "yaw_align": bool(msg.cs_yaw_align)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rclpy.init()
    node = Listener()
    end = time.time() + a.seconds
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except BaseException:  # SIGINT from the probe script: still write what we have
        pass
    rates = {k: (node.counts[k] / max(1e-6, time.time() - node.first[k]) if k in node.first else 0.0)
             for k in TOPICS}
    with open(a.out, "w") as f:
        json.dump({"counts": node.counts, "rates_hz_wall": rates, "odometry": node.odom,
                   "land_detected": node.land, "vehicle_status": node.status,
                   "estimator_flags": node.flags}, f)
    print(json.dumps({"counts": node.counts}))
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
