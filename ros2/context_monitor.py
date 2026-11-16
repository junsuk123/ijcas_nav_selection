"""context_monitor (Notion 05 module): observation-only diagnostics of sensors and environment.

Publishes JSON on /ijcas/diag/{vision,lidar,gnss,imu} with the *source* timestamp of the data it
describes and the simulation time it was computed at.  It reads only sensor topics (never
/ijcas/gt/*, never scenario labels).  Engine diagnostics that the engines actually publish are
added where available: OpenVINS' tracked MSCKF/SLAM feature clouds.

vision  per cam0 frame (every frame): FAST features, 4x4 grid coverage, mean/std intensity,
        stereo ORB matches with |dv| < 2 px (left-right consistency), image age.
lidar   per scan: points, median/p90 range, normal-structure eigenvalues (voxel-downsampled
        cloud, local PCA normals): lambda_min/lambda_max of sum(n n^T) and the weakest direction,
        vertical-structure fraction (|n_z| < 0.3), ground fraction (|n_z| > 0.9).
gnss    per fix: status, reported sigma, and the measurement-only consistency residual
        |dp_fix - integral(v_fix)| over a 2 s window (position bias with valid velocity shows up
        here even when the reported accuracy stays nominal).
imu     at 2 Hz: rate, stamp monotonicity, max |gyro|, max |accel| (saturation proximity).
"""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from sensor_msgs.msg import FluidPressure, Image, Imu, MagneticField, NavSatFix, PointCloud2
from std_msgs.msg import String

REL = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
BE = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
EARTH_RADIUS = 6353000.0


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def lidar_structure(xyz: np.ndarray, voxel: float = 0.7, k: int = 12) -> dict:
    """Local-PCA normals on a voxel-downsampled cloud and their information matrix."""
    rng = np.linalg.norm(xyz, axis=1)
    keep = (rng > 0.5) & np.isfinite(rng)
    xyz = xyz[keep]
    if len(xyz) < 50:
        return {"points": int(len(xyz)), "lambda_ratio": 0.0}
    key = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = xyz[idx]
    if len(pts) < 30:
        return {"points": int(len(xyz)), "voxels": int(len(pts)), "lambda_ratio": 0.0}
    tree = cKDTree(pts)
    _, nn = tree.query(pts, k=min(k, len(pts)))
    nb = pts[nn] - pts[nn].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    planar = (w[:, 1] > 4.0 * w[:, 0])           # keep well-defined (planar) neighbourhoods
    n = normals[planar] if planar.sum() > 20 else normals
    info = n.T @ n / len(n)
    ew, ev = np.linalg.eigh(info)
    weakest = ev[:, 0]
    return {"points": int(len(xyz)), "voxels": int(len(pts)),
            "range_median_m": float(np.median(rng[keep])),
            "range_p90_m": float(np.percentile(rng[keep], 90)),
            "lambda": [float(x) for x in ew], "lambda_ratio": float(ew[0] / max(ew[2], 1e-9)),
            "weakest_dir_lidar": [float(x) for x in weakest],
            "vertical_structure_frac": float((np.abs(n[:, 2]) < 0.3).mean()),
            "ground_frac": float((np.abs(n[:, 2]) > 0.9).mean())}


class ContextMonitor(Node):
    def __init__(self):
        super().__init__("ijcas_context_monitor",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        import cv2
        self.cv2 = cv2
        self.fast = cv2.FastFeatureDetector_create(threshold=20)
        self.orb = cv2.ORB_create(500)
        self.right = deque(maxlen=6)
        self.left = deque(maxlen=6)
        self.pub = {k: self.create_publisher(String, f"/ijcas/diag/{k}", REL)
                    for k in ("vision", "lidar", "gnss", "imu")}
        self.create_subscription(Image, "/cam0/image_raw", self.on_left, BE)
        self.create_subscription(Image, "/cam1/image_raw", self.on_right, BE)
        self.create_subscription(PointCloud2, "/ijcas/lidar/points", self.on_scan, BE)
        self.create_subscription(NavSatFix, "/ijcas/gnss/fix", self.on_fix, REL)
        self.create_subscription(TwistWithCovarianceStamped, "/ijcas/gnss/vel", self.on_vel, REL)
        self.create_subscription(Imu, "/ijcas/imu", self.on_imu, REL)
        self.fixes = deque(maxlen=40)
        self.vels = deque(maxlen=40)
        self.imu_buf = []
        self.aux = {"mag": [], "baro": []}
        self.create_subscription(MagneticField, "/ijcas/mag",
                                 lambda m: self.aux["mag"].append(stamp(m.header)), REL)
        self.create_subscription(FluidPressure, "/ijcas/baro",
                                 lambda m: self.aux["baro"].append(stamp(m.header)), REL)
        self.last_imu_pub = -1.0
        self.ov_counts = {}
        for name in ("points_msckf", "points_slam"):
            self.create_subscription(PointCloud2, f"/ov_msckf/{name}",
                                     lambda m, n=name: self.ov_counts.__setitem__(
                                         n, (stamp(m.header), m.width * m.height)), BE)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def emit(self, key, payload):
        self.pub[key].publish(String(data=json.dumps(payload)))

    @staticmethod
    def _img(m):
        return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)

    def on_right(self, m):
        t = stamp(m.header)
        self.right.append((t, self._img(m)))
        for tl, payload, img in list(self.left):
            if abs(tl - t) < 1e-3:
                self.left.remove((tl, payload, img))
                payload["stereo_matches"] = self._stereo(img, self.right[-1][1])
                self.emit("vision", payload)

    def _stereo(self, img, rimg):
        ka, da = self.orb.detectAndCompute(img, None)
        kb, db = self.orb.detectAndCompute(rimg, None)
        if da is None or db is None or not len(da) or not len(db):
            return 0
        mm = self.cv2.BFMatcher(self.cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        return len([x for x in mm if abs(ka[x.queryIdx].pt[1] - kb[x.trainIdx].pt[1]) < 2.0
                    and ka[x.queryIdx].pt[0] - kb[x.trainIdx].pt[0] > 0])

    def on_left(self, m):
        t = stamp(m.header)
        img = self._img(m)
        kps = self.fast.detect(img, None)
        h, w = img.shape
        grid = np.zeros((4, 4), int)
        for kp in kps:
            grid[min(3, int(kp.pt[1] * 4 / h)), min(3, int(kp.pt[0] * 4 / w))] += 1
        payload = {"t": t, "computed_at": self.now(), "features": len(kps),
                   "grid_cells_ge10": int((grid >= 10).sum()), "mean": float(img.mean()),
                   "std": float(img.std()), "stereo_matches": None}
        for name, (ts, n) in self.ov_counts.items():
            payload[f"ov_{name}"] = n
            payload[f"ov_{name}_t"] = ts
        for tr, rimg in self.right:
            if abs(tr - t) < 1e-3:
                payload["stereo_matches"] = self._stereo(img, rimg)
                self.emit("vision", payload)
                return
        self.left.append((t, payload, img))     # right image not here yet: emit when it arrives

    def on_scan(self, m):
        t = stamp(m.header)
        arr = np.frombuffer(m.data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("intensity", "<f4"), ("t", "<f4")])
        xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(float)
        self.emit("lidar", {"t": t, "computed_at": self.now(), **lidar_structure(xyz)})

    def on_vel(self, m):
        v = m.twist.twist.linear
        self.vels.append((stamp(m.header), np.array([v.x, v.y, v.z])))

    def on_fix(self, m):
        t = stamp(m.header)
        ok = m.status.status >= 0 and math.isfinite(m.latitude)
        payload = {"t": t, "computed_at": self.now(), "fix": bool(ok),
                   "sigma_h": math.sqrt(max(m.position_covariance[0], 0.0)) if ok else None}
        if ok:
            lat, lon = math.radians(m.latitude), math.radians(m.longitude)
            if not self.fixes:
                self.lat0, self.lon0 = lat, lon
            p = np.array([(lon - self.lon0) * math.cos(self.lat0) * EARTH_RADIUS,
                          (lat - self.lat0) * EARTH_RADIUS, m.altitude])
            self.fixes.append((t, p))
            old = [f for f in self.fixes if t - f[0] >= 2.0 - 1e-6]
            if old:
                t0, p0 = old[-1]
                vs = [v for tv, v in self.vels if t0 <= tv <= t]
                if len(vs) >= 3:
                    disp = np.mean(vs, axis=0) * (t - t0)
                    payload["pos_vel_residual_m"] = float(np.linalg.norm((p - p0)[:2] - disp[:2]))
                    payload["window_s"] = t - t0
        else:
            self.fixes.clear()
        self.emit("gnss", payload)

    def on_imu(self, m):
        t = stamp(m.header)
        w, a = m.angular_velocity, m.linear_acceleration
        self.imu_buf.append((t, max(abs(w.x), abs(w.y), abs(w.z)), max(abs(a.x), abs(a.y), abs(a.z)),
                             (abs(w.x), abs(w.y), abs(w.z), abs(a.x), abs(a.y), abs(a.z))))
        now = self.now()
        if now - self.last_imu_pub >= 0.5 and len(self.imu_buf) > 2:
            ts = np.array([b[0] for b in self.imu_buf])
            self.emit("imu", {"t": float(ts[-1]), "computed_at": now,
                              "rate_hz": float((len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)),
                              "non_monotonic": int((np.diff(ts) <= 0).sum()),
                              "stamp_minus_clock_s": float(ts[-1] - now),
                              "max_abs_gyro": float(max(b[1] for b in self.imu_buf)),
                              "max_abs_accel": float(max(b[2] for b in self.imu_buf)),
                              "saturation_frac": self._clipping(),
                              **{f"{k}_rate_hz": self._aux_rate(k, now) for k in self.aux}})
            self.imu_buf.clear()
            self.last_imu_pub = now


def _clip_frac(rows: np.ndarray) -> float:
    """Fraction of samples pinned at the window maximum on any axis (clipping flat-tops)."""
    n = len(rows)
    if n < 10:
        return 0.0
    top = rows.max(axis=0)
    at = (rows >= top - 1e-9).sum(axis=0) - 1
    return float(max(0, at.max()) / n)


ContextMonitor._clipping = lambda self: _clip_frac(np.array([b[3] for b in self.imu_buf]))


def _aux_rate(self, key, now, window=1.0):
    stamps = [x for x in self.aux[key] if x > now - window]
    self.aux[key] = stamps
    return float(len(stamps) / window)


ContextMonitor._aux_rate = _aux_rate


def main():
    rclpy.init()
    node = ContextMonitor()
    try:
        rclpy.spin(node)
    except BaseException:
        pass


if __name__ == "__main__":
    main()
