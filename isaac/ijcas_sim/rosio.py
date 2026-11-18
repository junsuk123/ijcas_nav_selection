"""ROS 2 output of the IJCAS rig (runs inside Isaac's Python with Isaac's bundled Humble rclpy).

Topic contract (all stamps are simulation time; /clock is published every physics step):

  inputs an engine/selector MAY use                     ground truth (evaluator only)
  /ijcas/imu            sensor_msgs/Imu  200 Hz FLU     /ijcas/gt/odom    nav_msgs/Odometry 50 Hz
  /ijcas/gnss/fix       sensor_msgs/NavSatFix 5 Hz      /ijcas/gt/faults  std_msgs/String (JSON)
  /ijcas/gnss/vel       geometry_msgs/TwistWithCovarianceStamped (ENU)
  /ijcas/baro           sensor_msgs/FluidPressure
  /ijcas/mag            sensor_msgs/MagneticField (FLU, tesla)
  /cam0|cam1/image_raw  sensor_msgs/Image mono8 20 Hz (+ camera_info)
  /ijcas/lidar/points   sensor_msgs/PointCloud2 10 Hz (x y z intensity t), sensor frame
  /ijcas/px4/odom       nav_msgs/Odometry (PX4 EKF2 via MAVLink, PX4 local frame -> ENU)
  /ijcas/px4/status     std_msgs/String (JSON: EKF2 flags, innovation test ratios, accuracy)
  /tf_static            base_link -> imu_link, cam0_optical, cam1_optical, lidar
  /ijcas/run/meta       std_msgs/String (JSON, transient local)

The IMU message never carries orientation (covariance[0] = -1): Pegasus' IMU dict includes the
true attitude, which would be a ground-truth leak.  Nothing under /ijcas/gt/ may be subscribed by
an engine or a selector (checked by tools/g1_leak_check.py).
"""
from __future__ import annotations

import array
import json
import math

import numpy as np

QOS_DEPTH = 50


def stamp(t: float):
    from builtin_interfaces.msg import Time
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


class RosIO:
    def __init__(self, node_name: str = "ijcas_sim"):
        import rclpy
        from geometry_msgs.msg import TwistWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import (CameraInfo, FluidPressure, Image, Imu, MagneticField,
                                     NavSatFix, PointCloud2)
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage

        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node(node_name)
        rel = QoSProfile(depth=QOS_DEPTH, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        p = self.node.create_publisher
        self.pub = {
            "clock": p(Clock, "/clock", 10),
            "imu": p(Imu, "/ijcas/imu", rel),
            "fix": p(NavSatFix, "/ijcas/gnss/fix", rel),
            "vel": p(TwistWithCovarianceStamped, "/ijcas/gnss/vel", rel),
            "baro": p(FluidPressure, "/ijcas/baro", rel),
            "mag": p(MagneticField, "/ijcas/mag", rel),
            "cam0": p(Image, "/cam0/image_raw", 10),
            "cam1": p(Image, "/cam1/image_raw", 10),
            "cam0_info": p(CameraInfo, "/cam0/camera_info", 10),
            "cam1_info": p(CameraInfo, "/cam1/camera_info", 10),
            "points": p(PointCloud2, "/ijcas/lidar/points", 10),
            "px4_odom": p(Odometry, "/ijcas/px4/odom", rel),
            "px4_status": p(String, "/ijcas/px4/status", rel),
            "gt_odom": p(Odometry, "/ijcas/gt/odom", rel),
            "gt_faults": p(String, "/ijcas/gt/faults", latched),
            "meta": p(String, "/ijcas/run/meta", latched),
            "tf_static": p(TFMessage, "/tf_static", latched),
        }
        self.msg = {"Clock": Clock, "Imu": Imu, "NavSatFix": NavSatFix, "Twist":
                    TwistWithCovarianceStamped, "Fluid": FluidPressure, "Mag": MagneticField,
                    "Image": Image, "Info": CameraInfo, "Cloud": PointCloud2, "Odom": Odometry,
                    "String": String, "TF": TFMessage}
        self.counts: dict[str, int] = {}

    def _send(self, key, msg):
        self.pub[key].publish(msg)
        self.counts[key] = self.counts.get(key, 0) + 1

    def spin_once(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    # --- simulation time -----------------------------------------------------------------
    def clock(self, t: float):
        m = self.msg["Clock"]()
        m.clock = stamp(t)
        self._send("clock", m)

    # --- proprioceptive / GNSS -----------------------------------------------------------
    def imu(self, t: float, gyro_flu, accel_flu, noise):
        m = self.msg["Imu"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = "imu_link"
        m.orientation_covariance[0] = -1.0
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro_flu)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = \
            map(float, accel_flu)
        gv = (noise.gyro_noise_density ** 2) * 200.0
        av = (noise.accel_noise_density ** 2) * 200.0
        for i in (0, 4, 8):
            m.angular_velocity_covariance[i] = gv
            m.linear_acceleration_covariance[i] = av
        self._send("imu", m)

    def gnss(self, t: float, g: dict):
        from sensor_msgs.msg import NavSatFix, NavSatStatus
        m = NavSatFix()
        m.header.stamp = stamp(t)
        m.header.frame_id = "gnss_link"
        m.status.status = NavSatStatus.STATUS_FIX if g["fix_type"] >= 3 else \
            NavSatStatus.STATUS_NO_FIX
        m.status.service = NavSatStatus.SERVICE_GPS
        if g["fix_type"] >= 3:
            m.latitude, m.longitude, m.altitude = g["lat_deg"], g["lon_deg"], g["alt_m"]
            h, v = g["eph_m"] ** 2, g["epv_m"] ** 2
            m.position_covariance = [h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, v]
            m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            m.latitude = m.longitude = m.altitude = float("nan")
            m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self._send("fix", m)
        if g["fix_type"] >= 3:
            v = self.msg["Twist"]()
            v.header = m.header
            v.twist.twist.linear.x, v.twist.twist.linear.y, v.twist.twist.linear.z = \
                map(float, g["vel_enu"])
            for i in (0, 7, 14):
                v.twist.covariance[i] = 0.05 ** 2
            self._send("vel", v)

    def baro(self, t: float, b: dict):
        m = self.msg["Fluid"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = "base_link"
        m.fluid_pressure = float(b["absolute_pressure"]) * 100.0   # hPa -> Pa
        self._send("baro", m)

    def mag(self, t: float, mg: dict):
        m = self.msg["Mag"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = "imu_link"
        frd = np.asarray(mg["magnetic_field"], float) * 1e-4      # gauss -> tesla
        m.magnetic_field.x, m.magnetic_field.y, m.magnetic_field.z = \
            float(frd[0]), float(-frd[1]), float(-frd[2])           # FRD -> FLU
        self._send("mag", m)

    # --- cameras / LiDAR -------------------------------------------------------------------
    def image(self, key: str, t: float, gray: np.ndarray, frame_id: str):
        m = self.msg["Image"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = frame_id
        m.height, m.width = int(gray.shape[0]), int(gray.shape[1])
        m.encoding = "mono8"
        m.step = m.width
        # array.array, not bytes: rclpy converts a bytes payload element by element (56 ms for
        # 640x480 vs 0.014 ms, results/g1/README)
        m.data = array.array("B", np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
        self._send(key, m)

    def camera_info(self, key: str, t: float, frame_id: str, k: np.ndarray, w: int, h: int):
        m = self.msg["Info"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = frame_id
        m.width, m.height = int(w), int(h)
        m.distortion_model = "plumb_bob"
        m.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        m.k = [float(v) for v in np.asarray(k).reshape(-1)]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [float(k[0][0]), 0.0, float(k[0][2]), 0.0, 0.0, float(k[1][1]), float(k[1][2]),
               0.0, 0.0, 0.0, 1.0, 0.0]
        self._send(key, m)

    def points(self, t: float, xyz: np.ndarray, intensity: np.ndarray, rel_t: np.ndarray,
               frame_id: str = "lidar"):
        from sensor_msgs.msg import PointField
        m = self.msg["Cloud"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = frame_id
        n = int(len(xyz))
        arr = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("intensity", "<f4"), ("t", "<f4")])
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["intensity"] = intensity
        arr["t"] = rel_t
        m.height, m.width = 1, n
        m.fields = [PointField(name=nm, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, nm in enumerate(("x", "y", "z", "intensity", "t"))]
        m.is_bigendian = False
        m.point_step = 20
        m.row_step = 20 * n
        m.is_dense = True
        m.data = array.array("B", arr.tobytes())
        self._send("points", m)

    # --- odometry (PX4 estimate, ground truth) ----------------------------------------------
    def odometry(self, key: str, t: float, frame: str, child: str, pos, quat_xyzw, vel_world,
                 body_rates=None, pose_cov=None, twist_cov=None):
        m = self.msg["Odom"]()
        m.header.stamp = stamp(t)
        m.header.frame_id = frame
        m.child_frame_id = child
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, pos)
        q = [float(v) for v in quat_xyzw]
        (m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z,
         m.pose.pose.orientation.w) = q
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = \
            map(float, vel_world)
        if body_rates is not None:
            m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = \
                map(float, body_rates)
        if pose_cov is not None:
            m.pose.covariance = [float(v) for v in pose_cov]
        if twist_cov is not None:
            m.twist.covariance = [float(v) for v in twist_cov]
        self._send(key, m)

    def json(self, key: str, payload):
        m = self.msg["String"]()
        m.data = json.dumps(payload)
        self._send(key, m)

    def static_transforms(self, transforms: list[dict]):
        from geometry_msgs.msg import TransformStamped
        msg = self.msg["TF"]()
        for tr in transforms:
            ts = TransformStamped()
            ts.header.stamp = stamp(0.0)
            ts.header.frame_id = tr["parent"]
            ts.child_frame_id = tr["child"]
            (ts.transform.translation.x, ts.transform.translation.y,
             ts.transform.translation.z) = map(float, tr["xyz"])
            (ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z,
             ts.transform.rotation.w) = map(float, tr["quat_xyzw"])
            msg.transforms.append(ts)
        self._send("tf_static", msg)

    def close(self):
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass
