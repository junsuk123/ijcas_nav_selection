"""Stereo camera pair + RTX LiDAR on the vehicle body, with measured calibration (Isaac-only).

Calibration is *measured from the stage*, not assumed: Isaac's camera-axes conventions are easy
to get wrong, so each camera is first mounted with an identity orientation, its optical frame is
read back (``camera_axes="ros"``: z forward, x right, y down), the correction to the desired
mount is applied, and the result is read back again and checked.  The exported extrinsics are
those read-backs.

Timing: annotator data are stamped with their ``rendering_time`` (the simulation time of the
render that produced them), never with the loop time; with asynchronous rendering that differs by
up to 0.2 s (results/g1/lidar_probe*).  LiDAR: the non-accumulating RTX output is one ~180 deg
sector per render (a snapshot at that render's pose); two consecutive sectors form one 360 deg
scan stamped at the first sector's render time, with per-point ``t`` = sector render time minus
the scan stamp (0 or +render_dt).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

LIDAR_CONFIG = "OS1_REV6_32ch10hz1024res"


def _wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)


def _xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], float)


def desired_optical_in_body(pitch_down_deg: float) -> np.ndarray:
    """Columns: optical x (right), y (down), z (forward, pitched down) expressed in body FLU."""
    p = np.radians(pitch_down_deg)
    z = np.array([np.cos(p), 0.0, -np.sin(p)])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass
class CameraSpec:
    name: str
    translation_body: np.ndarray
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    pitch_down_deg: float = 15.0


@dataclass
class Rig:
    body_path: str
    cameras: list = field(default_factory=list)
    lidar: object = None
    specs: list = field(default_factory=list)
    calibration: dict = field(default_factory=dict)
    _last_cam_rt: dict = field(default_factory=dict)
    _last_lidar_rt: float = -1.0
    _sectors: list = field(default_factory=list)

    @staticmethod
    def default_specs(baseline_m: float = 0.12) -> list:
        return [CameraSpec("cam0", np.array([0.15, baseline_m / 2, -0.03])),
                CameraSpec("cam1", np.array([0.15, -baseline_m / 2, -0.03]))]

    def build(self, specs=None, lidar_translation=(0.0, 0.0, 0.12)):
        from isaacsim.sensors.camera import Camera
        from isaacsim.sensors.rtx import LidarRtx
        from pxr import UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.specs = specs or self.default_specs()
        for s in self.specs:
            cam = Camera(prim_path=f"{self.body_path}/{s.name}", resolution=(s.width, s.height))
            usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(f"{self.body_path}/{s.name}"))
            h_ap = 20.955
            usd_cam.GetHorizontalApertureAttr().Set(h_ap)
            usd_cam.GetVerticalApertureAttr().Set(h_ap * s.height / s.width)
            usd_cam.GetFocalLengthAttr().Set(h_ap / 2.0 / np.tan(np.radians(s.hfov_deg) / 2.0))
            usd_cam.GetClippingRangeAttr().Set((0.05, 300.0))
            cam.set_local_pose(translation=s.translation_body,
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="ros")
            self.cameras.append(cam)
        self.lidar = LidarRtx(prim_path=f"{self.body_path}/lidar",
                              translation=np.asarray(lidar_translation, float),
                              orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                              config_file_name=LIDAR_CONFIG)
        self.lidar_translation = np.asarray(lidar_translation, float)

    def initialize(self):
        for cam in self.cameras:
            cam.initialize()
        self.lidar.initialize()
        self.lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

    # --- calibration (measured) --------------------------------------------------------------
    def _body_pose(self):
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        m = UsdGeom.Xformable(stage.GetPrimAtPath(self.body_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        mat = np.array(m).T            # Gf row-vector convention -> column-vector matrix
        return mat[:3, 3].copy(), mat[:3, :3].copy()

    def _measure(self, cam):
        pb, rb = self._body_pose()
        pc, qc = cam.get_world_pose(camera_axes="ros")
        rc = Rotation.from_quat(_xyzw(qc)).as_matrix()
        return rb.T @ (np.asarray(pc, float) - pb), rb.T @ rc

    def calibrate(self) -> dict:
        cams = {}
        for cam, s in zip(self.cameras, self.specs):
            _, m0 = self._measure(cam)
            desired = desired_optical_in_body(s.pitch_down_deg)
            corr = desired @ m0.T
            cam.set_local_pose(translation=s.translation_body,
                               orientation=_wxyz(Rotation.from_matrix(corr).as_quat()),
                               camera_axes="ros")
            t1, m1 = self._measure(cam)
            err_deg = float(np.degrees(Rotation.from_matrix(desired.T @ m1).magnitude()))
            k = np.asarray(cam.get_intrinsics_matrix(), float)
            cams[s.name] = {"width": s.width, "height": s.height, "K": k.round(6).tolist(),
                            "distortion": [0.0, 0.0, 0.0, 0.0],
                            "R_body_optical": m1.round(9).tolist(),
                            "p_body_optical": t1.round(6).tolist(),
                            "mount_error_deg": err_deg,
                            "translation_error_m": float(np.linalg.norm(t1 - s.translation_body))}
        pb, rb = self._body_pose()
        pl, ql = self.lidar.get_world_pose()
        rl = Rotation.from_quat(_xyzw(ql)).as_matrix()
        self.calibration = {
            "imu_frame": "base_link (FLU, body origin)",
            "cameras": cams,
            "baseline_m": float(np.linalg.norm(np.asarray(cams["cam0"]["p_body_optical"])
                                               - np.asarray(cams["cam1"]["p_body_optical"]))),
            "lidar": {"config": LIDAR_CONFIG, "R_body_lidar": (rb.T @ rl).round(9).tolist(),
                      "p_body_lidar": (rb.T @ (np.asarray(pl, float) - pb)).round(6).tolist()},
        }
        return self.calibration

    # --- capture -----------------------------------------------------------------------------
    def new_images(self):
        """[(name, rendering_time, gray uint8)] for cameras that produced a new render."""
        import cv2
        out = []
        for cam, s in zip(self.cameras, self.specs):
            frame = cam.get_current_frame()
            rt = float(frame.get("rendering_time", -1.0))
            if rt < 0 or self._last_cam_rt.get(s.name) == rt:
                continue
            rgba = cam.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                continue
            self._last_cam_rt[s.name] = rt
            gray = cv2.cvtColor(np.asarray(rgba)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            out.append((s.name, rt, gray))
        return out

    def new_scan(self):
        """A 360 deg scan (stamp, xyz, intensity, rel_t) once two new sectors are available."""
        frame = self.lidar.get_current_frame()
        rt = float(frame.get("rendering_time", -1.0)) if frame else -1.0
        buf = frame.get("IsaacExtractRTXSensorPointCloudNoAccumulator") if frame else None
        if rt < 0 or rt == self._last_lidar_rt or buf is None or not len(buf.get("data", [])):
            return None
        self._last_lidar_rt = rt
        xyz = np.asarray(buf["data"], np.float32).reshape(-1, 3)
        inten = np.asarray(buf.get("intensity", []), np.float32)
        if inten.shape[0] != xyz.shape[0]:
            inten = np.zeros(xyz.shape[0], np.float32)
        self._sectors.append((rt, xyz, inten))
        if len(self._sectors) < 2:
            return None
        (t0, x0, i0), (t1, x1, i1) = self._sectors[-2], self._sectors[-1]
        self._sectors.clear()
        xyz = np.vstack([x0, x1])
        # stamp = scan start (first sector), per-point t >= 0 (GLIM rejects negative offsets)
        rel = np.concatenate([np.zeros(len(x0), np.float32), np.full(len(x1), t1 - t0, np.float32)])
        return t0, xyz, np.concatenate([i0, i1]), rel
