"""Per-solution covariance in the common frame, from the values the estimators publish.

No fitted, calibrated or substituted values (user decision 2026-11-18: "use real").  Only frame changes
are applied; a block the estimator does not report is marked unavailable and that solution does not take
part in the CI of that block (ijcas/fusion/ci.py).

Sources (record.npz from ros2/recorder.py; odom rows: 0-17 as before, 18-53 pose covariance 6x6,
54-89 twist covariance 6x6, row-major, as published):
  px4      /ijcas/px4/odom - EKF2 variances (MAVLink ODOMETRY diagonal) already in local ENU (position,
           velocity) and body axes (attitude).  Records made before 2026-11-18 lack them; there only the
           EKF2 ESTIMATOR_STATUS pos_horiz_accuracy / pos_vert_accuracy (1 sigma) give a position block.
  openvins /ov_msckf/odomimu - pose 6x6 in ROS order (position, orientation) in the OpenVINS global
           frame; twist linear block in the IMU frame (OpenVINS publishes the body velocity).
  fastlio  /fast_lio/odom - pose 6x6 with FAST-LIO2's layout (rows/cols 0-2 rotation, 3-5 position,
           laserMapping.cpp publish_odometry); twist linear block = filter velocity covariance in the
           camera_init frame (IJCAS output adapter, config/engines/fast_lio/fast_lio_ijcas.patch).
Frame changes: VIO / LIO local frame -> PX4 local ENU by the causal 4-DOF alignment yaw (position and
world velocity); OpenVINS body velocity -> world with the solution attitude and the yaw; attitude blocks
are body-axis (right) perturbations and are not rotated (assumption for all three estimators).
Not represented (not reported by any estimator): the error of the causal alignment itself, the growth
of uncertainty while an output is propagated to the fusion time, and cross-block / cross-estimator terms.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ("px4", "openvins", "fastlio")
N_FULL = 90                                   # odom row width with full covariances


def _sym_pd(M: np.ndarray, floor: float = 0.0) -> bool:
    if not np.all(np.isfinite(M)):
        return False
    if not np.allclose(M, M.T, atol=1e-9 * max(1.0, float(np.abs(M).max()))):
        return False
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    return bool(ev.min() > floor)


def _yaw_rot(yaw: float | None) -> np.ndarray:
    return np.eye(3) if yaw is None else Rotation.from_euler("z", yaw).as_matrix()


def solution_blocks(name: str, row: np.ndarray | None, yaw: float | None = None,
                    quat_local: np.ndarray | None = None, px4_status: dict | None = None):
    """-> (blocks [3,3,3] for (pos, vel, att), avail [3] bool).  ``row``: the odom record row of the
    output used at this tick (None if none); ``yaw``: alignment yaw (VIO/LIO); ``quat_local``: the
    solution attitude in its own frame (OpenVINS body velocity)."""
    blocks = np.full((3, 3, 3), np.nan)
    avail = np.zeros(3, bool)
    full = row is not None and len(row) >= N_FULL
    Ry = _yaw_rot(yaw)
    if full:
        pc = np.asarray(row[18:54], float).reshape(6, 6)
        tc = np.asarray(row[54:90], float).reshape(6, 6)
        if name == "fastlio":
            Pp, Pa = pc[3:6, 3:6], pc[0:3, 0:3]
        else:
            Pp, Pa = pc[0:3, 0:3], pc[3:6, 3:6]
        Pv = tc[0:3, 0:3]
        if name == "openvins" and quat_local is not None:
            Rb = Rotation.from_quat(quat_local).as_matrix()          # body -> OpenVINS world
            Pv = Rb @ Pv @ Rb.T
        if name in ("openvins", "fastlio"):
            Pp, Pv = Ry @ Pp @ Ry.T, Ry @ Pv @ Ry.T
        for b, M in enumerate((Pp, Pv, Pa)):
            M = 0.5 * (M + M.T)
            if np.any(M != 0) and _sym_pd(M):
                blocks[b], avail[b] = M, True
    elif row is not None and name in ("openvins", "fastlio") and len(row) >= 18:
        d = np.asarray(row[12:18], float)                             # older records: diagonals only
        dp, da = (d[3:6], d[0:3]) if name == "fastlio" else (d[0:3], d[3:6])
        for b, dd, rot in ((0, dp, True), (2, da, False)):
            M = np.diag(dd)
            if rot:
                M = Ry @ M @ Ry.T
            if np.all(dd > 0) and _sym_pd(M):
                blocks[b], avail[b] = M, True
    if name == "px4" and not avail[0] and px4_status:
        sh, sv = px4_status.get("pos_horiz_accuracy"), px4_status.get("pos_vert_accuracy")
        if sh is not None and sv is not None and sh > 0 and sv > 0:
            blocks[0], avail[0] = np.diag([sh * sh, sh * sh, sv * sv]), True
    return blocks, avail
