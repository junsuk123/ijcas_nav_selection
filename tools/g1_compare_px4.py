"""PX4 EKF2 (MAVLink LOCAL_POSITION_NED + ATTITUDE_QUATERNION, PX4 lockstep time) vs ground truth.

    python3 tools/g1_compare_px4.py results/g0/px4_probe

Time: the probe logs Pegasus' PX4 time (``px4_time_s``) beside simulation time, so PX4's
``time_boot_ms`` maps exactly onto ground truth.  Frame: PX4 local NED (origin = first GNSS fix)
-> ENU; the origin offset is the mean difference over the 3 s hover before motion (translation only,
no rotation or scale fitted).  Attitude FRD->NED -> FLU->ENU.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

NED2ENU = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
FRD2FLU = R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def compare(out_dir: str) -> dict:
    rec = json.load(open(f"{out_dir}/px4_mavlink.json"))
    gt = np.loadtxt(f"{out_dir}/gt_enu.txt")
    ok = gt[:, 11] > 0
    to_sim = lambda tb: np.interp(tb, gt[ok, 11], gt[ok, 0])
    gi = lambda t, c: np.interp(t, gt[:, 0], gt[:, c])
    lp = np.asarray(rec["lpos"], float)
    at = np.asarray(rec["att"], float)
    tl = to_sim(lp[:, 0])
    p = np.c_[lp[:, 2], lp[:, 1], -lp[:, 3]]
    v = np.c_[lp[:, 5], lp[:, 4], -lp[:, 6]]
    gp = np.c_[gi(tl, 1), gi(tl, 2), gi(tl, 3)]
    gv = np.c_[gi(tl, 8), gi(tl, 9), gi(tl, 10)]
    onset = gt[np.argmax(np.linalg.norm(gt[:, 8:10], axis=1) > 0.3), 0]
    hover = (tl > onset - 3.0) & (tl < onset)
    off = (gp[hover] - p[hover]).mean(0)
    moving = tl > onset
    perr = np.linalg.norm(p + off - gp, axis=1)
    verr = np.linalg.norm(v - gv, axis=1)
    ta = to_sim(at[:, 0])
    Re = NED2ENU * R.from_quat(np.c_[at[:, 2], at[:, 3], at[:, 4], at[:, 1]]) * FRD2FLU
    gR = R.from_quat(np.c_[gi(ta, 4), gi(ta, 5), gi(ta, 6), gi(ta, 7)])
    aerr = np.degrees((gR.inv() * Re).magnitude())
    am = ta > onset
    yaw_e = np.degrees(np.angle(np.exp(1j * (Re.as_euler("ZYX")[:, 0] - gR.as_euler("ZYX")[:, 0]))))
    res = {"samples_pos": int(moving.sum()), "samples_att": int(am.sum()),
           "origin_offset_enu_m": off.round(3).tolist(),
           "pos_rmse_m": float(np.sqrt(np.mean(perr[moving] ** 2))),
           "pos_p95_m": float(np.percentile(perr[moving], 95)),
           "alt_rmse_m": float(np.sqrt(np.mean((p[moving, 2] + off[2] - gp[moving, 2]) ** 2))),
           "vel_rmse_mps": float(np.sqrt(np.mean(verr[moving] ** 2))),
           "att_rmse_deg": float(np.sqrt(np.mean(aerr[am] ** 2))),
           "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_e[am] ** 2))),
           "att_max_deg": float(aerr[am].max())}
    json.dump(res, open(f"{out_dir}/px4_vs_gt.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1]), indent=2))
