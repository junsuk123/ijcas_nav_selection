"""G1 acceptance checks on one capture run (Notion G1 완료 기준).

    python3 tools/g1_checks.py $IJCAS_DATA/g1/warehouse_1 [--json results/g1/checks_warehouse.json]

1. common IMU/GNSS path   ROS /ijcas/gnss/fix == PX4 GPS_RAW_INT (same samples, 1e-7 deg);
                          ROS /ijcas/imu vs PX4 HIGHRES_IMU (PX4 filters/averages: compared on
                          50 Hz windows, difference must be noise-sized, no bias/scale/axis error)
2. stereo calibration     measured mount errors, K, baseline; epipolar residual of ORB matches
                          (parallel rig: |v_l - v_r| small, disparity > 0)
3. LiDAR timing/frame     per-point t in {-render_dt, 0}; 10 Hz stamps; floor points mapped to the
                          world with GT pose at (stamp + t) and the measured extrinsic lie on z = 0
4. ground-truth leakage   IMU messages carry no orientation; GT only under /ijcas/gt/; PX4 aiding
                          parameters (EV/OF/RNG off) read back from PX4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def stereo_epipolar(pl, pr, K):
    import cv2
    orb = cv2.ORB_create(2000)
    res = []
    for a, b in zip(pl, pr):
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        if not m:
            continue
        ua = np.array([ka[x.queryIdx].pt for x in m])
        ub = np.array([kb[x.trainIdx].pt for x in m])
        dv, du = ua[:, 1] - ub[:, 1], ua[:, 0] - ub[:, 0]
        good = np.abs(dv) < 5.0                     # discard outlier matches
        res.append({"matches": int(len(m)), "inliers": int(good.sum()),
                    "dv_median_px": float(np.median(np.abs(dv[good]))) if good.any() else None,
                    "dv_p90_px": float(np.percentile(np.abs(dv[good]), 90)) if good.any() else None,
                    "disparity_positive_frac": float((du[good] > 0).mean()) if good.any() else None})
    return res


def lidar_floor(scans, scan_t, gt, calib, floor_z=0.0):
    rbl = np.asarray(calib["lidar"]["R_body_lidar"])
    pbl = np.asarray(calib["lidar"]["p_body_lidar"])
    out = []
    for st, arr in zip(scan_t, scans):
        xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(float)
        rel = arr["t"].astype(float)
        tp = st + rel
        gi = lambda c: np.interp(tp, gt[:, 0], gt[:, c])
        pos = np.column_stack([gi(1), gi(2), gi(3)])
        # slerp-free: attitude changes < 0.1 rad within 50 ms; interpolate quaternion components
        q = np.column_stack([gi(4), gi(5), gi(6), gi(7)])
        rot = R.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))
        world = pos + rot.apply(xyz @ rbl.T + pbl)
        floor = np.abs(world[:, 2] - floor_z) < 0.4
        # same points, timing ignored (all at the scan stamp)
        p0 = np.array([np.interp(st, gt[:, 0], gt[:, c]) for c in (1, 2, 3)])
        q0 = np.array([np.interp(st, gt[:, 0], gt[:, c]) for c in (4, 5, 6, 7)])
        world_nt = p0 + R.from_quat(q0 / np.linalg.norm(q0)).apply(xyz @ rbl.T + pbl)
        if floor.sum() < 50:
            continue
        hz = np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum() / (gt[-1, 0] - gt[0, 0])
        out.append({"t": float(st), "points": int(len(xyz)), "floor_points": int(floor.sum()),
                    "rel_t_values": sorted({round(float(v), 4) for v in np.unique(rel)}),
                    "floor_z_mean_m": float(world[floor, 2].mean() - floor_z),
                    "floor_z_std_m": float(world[floor, 2].std()),
                    "floor_xy_shift_if_time_ignored_mean_m": float(np.median(np.linalg.norm(
                        world_nt[floor, :2] - world[floor, :2], axis=1))),
                    "mean_speed_mps": float(hz)})
    return out


def main(run: Path) -> dict:
    cap = np.load(run / "capture.npz", allow_pickle=True)
    man = json.loads((run / "manifest.json").read_text())
    px4 = json.loads((run / "px4_mavlink.json").read_text())
    calib = man["calibration"]
    res = {"run": str(run), "scene": man["scene"], "rtf": man["rtf"],
           "ros_counts": man["ros_counts"]}
    # 1 common GNSS / IMU path ---------------------------------------------------------------
    fix = cap["fix"]
    gps = np.asarray(px4["gps"], float)
    if len(fix) and len(gps):
        ok = fix[:, 1] >= 0
        diffs = []
        for row in gps:
            j = np.argmin(np.abs(fix[:, 0] - (row[0] - 1e-6)))
            if abs(fix[j, 0] - row[0]) < 1e-3 and ok[j]:
                diffs.append([fix[j, 2] - row[2] * 1e-7, fix[j, 3] - row[3] * 1e-7])
        diffs = np.abs(np.asarray(diffs))
        res["gnss_common"] = {"matched": int(len(diffs)),
                              "max_lat_lon_diff_deg": float(diffs.max()) if len(diffs) else None}
    imu = cap["imu"]
    him = np.asarray(px4["himu"], float)
    if len(imu) and len(him):
        flu2frd = np.array([1, -1, -1])
        d_acc, d_gyr, sig = [], [], []
        for row in him[10:]:
            w = (imu[:, 0] > row[0] - 0.02) & (imu[:, 0] <= row[0] + 1e-4)
            if w.sum() < 2:
                continue
            ros_acc = imu[w, 4:7].mean(0) * flu2frd
            ros_gyr = imu[w, 1:4].mean(0) * flu2frd
            d_acc.append(ros_acc - row[1:4])
            d_gyr.append(ros_gyr - row[4:7])
        d_acc, d_gyr = np.asarray(d_acc), np.asarray(d_gyr)
        res["imu_common"] = {"windows": int(len(d_acc)),
                             "accel_diff_mean": d_acc.mean(0).round(4).tolist(),
                             "accel_diff_std": d_acc.std(0).round(4).tolist(),
                             "gyro_diff_mean": d_gyr.mean(0).round(5).tolist(),
                             "gyro_diff_std": d_gyr.std(0).round(5).tolist()}
    # 2 stereo -------------------------------------------------------------------------------
    cams = calib["cameras"]
    res["stereo"] = {"K_cam0": cams["cam0"]["K"], "baseline_m": calib["baseline_m"],
                     "mount_error_deg": {k: v["mount_error_deg"] for k, v in cams.items()},
                     "translation_error_m": {k: v["translation_error_m"] for k, v in cams.items()},
                     "epipolar": stereo_epipolar(cap["pair_l"], cap["pair_r"], cams["cam0"]["K"])}
    # 3 LiDAR --------------------------------------------------------------------------------
    gt = cap["gt"]
    st = cap["scan_t"]
    stamps = cap["scan_stamps"] if "scan_stamps" in cap.files else st
    res["lidar"] = {"scan_period_s": float(np.median(np.diff(stamps))) if len(stamps) > 1 else None,
                    "scan_period_max_s": float(np.max(np.diff(stamps))) if len(stamps) > 1 else None,
                    "points_count": man["ros_counts"].get("points"),
                    "floor_check": lidar_floor(cap["scans"], st, gt, calib)}
    # 4 leakage ------------------------------------------------------------------------------
    io = cap["imu_orient"]
    res["leak"] = {"imu_orientation_is_constant_identity": bool(np.all(io[:, :4] == [0, 0, 0, 1])),
                   "imu_orientation_cov_unknown": bool(np.all(io[:, 4] == -1)),
                   "px4_param_readback": man.get("px4_param_readback")}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = main(Path(a.run))
    txt = json.dumps(r, indent=1, default=float)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
