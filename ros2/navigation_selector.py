"""navigation_selector node (online, 10 Hz, use_sim_time): the offline SelectionEngine on live facts.

    python3 ros2/navigation_selector.py --selector ontology|flat [--thresholds <yaml>]

Inputs: candidate outputs (/ijcas/px4/odom, /ov_msckf/odomimu, /glim_ros/odom), diagnostics
(/ijcas/diag/*, /ijcas/px4/status), GNSS fix, IMU.  Never /ijcas/gt/*.
Outputs (single authority for the navigation solution, Notion 05 state output contract):
  /ijcas/nav/selected    nav_msgs/Odometry in "map" (= PX4 local ENU) or nothing on no-valid-nav
  /ijcas/nav/decision    std_msgs/String JSON: t, selected, mode, states, reasons, costs,
                         source stamp, output stamp, latency of the decision (wall, ms)
  /tf                    map -> base_link_nav (only this node publishes it)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ijcas.selector.common import load_config  # noqa: E402
from ijcas.selector.engine import Fixed, HealthFsm, SelectionEngine  # noqa: E402
from ijcas.selector.facts import Snapshot, make_output  # noqa: E402
from ijcas.selector.flat import FlatRules  # noqa: E402
from ijcas.selector.ontology import OntologyReasoner  # noqa: E402

BE = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
REL = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
ODOM = {"px4": "/ijcas/px4/odom", "openvins": "/ov_msckf/odomimu", "glim": "/glim_ros/odom"}


def st(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class SelectorNode(Node):
    def __init__(self, kind: str, thresholds: str | None):
        super().__init__("ijcas_navigation_selector",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        cfg = load_config(thresholds)
        if kind == "fixed_px4":                      # F-G comparator for closed loop (G8)
            self.engine = Fixed("px4", cfg)
        elif kind == "h_fsm":
            self.engine = HealthFsm(cfg)
        else:
            reasoner = OntologyReasoner(cfg) if kind == "ontology" else FlatRules(cfg)
            self.engine = SelectionEngine(reasoner, cfg, name=kind)
        self.rate_hz = cfg["selection"]["rate_hz"]
        self.outputs, self.hist = {}, {}
        self.diag, self.px4 = {}, None
        self.fix, self.imu_rows = None, []
        for name, topic in ODOM.items():
            self.create_subscription(Odometry, topic, lambda m, n=name: self.on_odom(n, m), BE)
        for kind_ in ("vision", "lidar", "gnss", "imu"):
            self.create_subscription(String, f"/ijcas/diag/{kind_}",
                                     lambda m, k=kind_: self.on_diag(k, m), REL)
        self.create_subscription(String, "/ijcas/px4/status", self.on_px4, REL)
        self.create_subscription(NavSatFix, "/ijcas/gnss/fix", self.on_fix, REL)
        self.create_subscription(Imu, "/ijcas/imu", self.on_imu, REL)
        self.pub = self.create_publisher(Odometry, "/ijcas/nav/selected", REL)
        self.pub_dec = self.create_publisher(String, "/ijcas/nav/decision", REL)
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.period = 1.0 / self.rate_hz
        self.next_t = None
        self.create_timer(0.005, self.tick)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, name, m):
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        o = make_output(name, st(m.header), [p.x, p.y, p.z], [q.x, q.y, q.z, q.w], [v.x, v.y, v.z])
        self.outputs[name] = o
        self.hist.setdefault(name, []).append(o)

    def on_diag(self, kind, m):
        self.diag[kind] = (json.loads(m.data), self.now_s())

    def on_px4(self, m):
        self.px4 = (json.loads(m.data), self.now_s())

    def on_fix(self, m):
        self.fix = np.array([st(m.header), self.now_s(), m.status.status, m.latitude, m.longitude,
                             m.altitude, m.position_covariance[0], m.position_covariance[8]])

    def on_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu_rows.append([st(m.header), w.x, w.y, w.z, a.x, a.y, a.z])

    def tick(self):
        t = self.now_s()
        if t <= 0.0:
            return
        if self.next_t is None:
            self.next_t = t
        if t < self.next_t:
            return
        self.next_t += self.period
        w0 = time.perf_counter()
        snap = Snapshot(t=t, outputs=dict(self.outputs), output_history=self.hist,
                        diag=dict(self.diag), px4_status=self.px4, fix=self.fix,
                        imu_window=np.asarray(self.imu_rows) if self.imu_rows else None)
        self.hist, self.imu_rows = {}, []
        d, out = self.engine.step(snap)
        lat_ms = 1000 * (time.perf_counter() - w0)
        src = None if d.selected in (None, "ins_dr") else self.outputs[d.selected].stamp
        if out is not None:
            pos, quat, vel = out
            msg = Odometry()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.child_frame_id = "base_link_nav"
            (msg.pose.pose.position.x, msg.pose.pose.position.y,
             msg.pose.pose.position.z) = map(float, pos)
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) = map(float, quat)
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = \
                map(float, vel)
            self.pub.publish(msg)
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = "base_link_nav"
            (tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z) = map(float, pos)
            tf.transform.rotation = msg.pose.pose.orientation
            self.pub_tf.publish(TFMessage(transforms=[tf]))
        self.pub_dec.publish(String(data=json.dumps({
            "t": t, "selector": type(self.engine).__name__, "selected": d.selected,
            "mode": d.output_mode, "states": d.states,
            "reasons": d.reasons, "costs": d.costs, "source_stamp": src,
            "decision_latency_ms": lat_ms, "switched": d.switched})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", choices=("ontology", "flat", "fixed_px4", "h_fsm"), default="ontology")
    ap.add_argument("--thresholds", default=None)
    a = ap.parse_args()
    rclpy.init()
    node = SelectorNode(a.selector, a.thresholds)
    try:
        rclpy.spin(node)
    except BaseException:
        pass


if __name__ == "__main__":
    main()
