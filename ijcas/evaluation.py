"""Evaluation of candidate navigation outputs against simulator ground truth (offline only).

Ground truth is used here and in the Oracle comparator only; nothing in this module may be
imported by a selector (tests/test_leakage.py).

Alignment conventions (Notion 04 "ATE는 평가용 정렬 방식까지 명시"):
* ``initial``  4-DOF (x, y, z, yaw) alignment fitted on the first ``window_s`` seconds after the
               candidate's first output, then frozen: the error grows with drift, as it would for a
               user who aligned the local frame once at start-up.
* ``full``     4-DOF alignment on the whole run (the usual VIO/LIO ATE; hides drift direction).
* ``origin``   translation only on the first window (PX4 / GNSS: already gravity- and
               north-aligned through the magnetometer / GNSS).
No Sim(3) / scale alignment is ever applied (metric stereo / LiDAR / GNSS).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

GT_COLS = "t px py pz qx qy qz qw vx vy vz wx wy wz".split()


@dataclass
class Track:
    name: str
    stamp: np.ndarray       # header stamp (s, simulation time)
    recv: np.ndarray        # reception time (s, simulation time)
    pos: np.ndarray         # (N,3) in the candidate's own frame
    quat: np.ndarray        # (N,4) xyzw
    vel: np.ndarray         # (N,3)

    def __len__(self):
        return len(self.stamp)


def load_run(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    rec = np.load(run / "record.npz")
    gt = np.load(run / "gt.npz")["gt"]
    tracks = {}
    for name in ("openvins", "fastlio", "px4"):
        a = rec[f"odom_{name}"]
        if a.size == 0:
            tracks[name] = None
            continue
        tracks[name] = Track(name, a[:, 0], a[:, 1], a[:, 2:5], a[:, 5:9], a[:, 9:12])
    manifest = json.loads((run / "manifest.json").read_text())
    return {"gt": gt, "tracks": tracks, "record": rec, "manifest": manifest}


def gt_at(gt: np.ndarray, t: np.ndarray):
    pos = np.column_stack([np.interp(t, gt[:, 0], gt[:, c]) for c in (1, 2, 3)])
    vel = np.column_stack([np.interp(t, gt[:, 0], gt[:, c]) for c in (8, 9, 10)])
    from scipy.spatial.transform import Slerp
    rots = R.from_quat(gt[:, 4:8])
    tt = np.clip(t, gt[0, 0], gt[-1, 0])
    return pos, Slerp(gt[:, 0], rots)(tt), vel


def fit_4dof(src: np.ndarray, dst: np.ndarray):
    """yaw + translation minimising |R src + t - dst| (closed form)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - ms[:2], dst[:, :2] - md[:2]
    yaw = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    rot = R.from_euler("z", yaw)
    return rot, md - rot.apply(ms)


def errors(track: Track, gt: np.ndarray, mode: str = "initial", window_s: float = 10.0,
           t_from: float | None = None) -> dict:
    t = track.stamp
    ok = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    if t_from is not None:
        ok &= t >= t_from
    if ok.sum() < 10:
        return {"samples": int(ok.sum())}
    t, pos, quat, vel = t[ok], track.pos[ok], track.quat[ok], track.vel[ok]
    gp, gr, gv = gt_at(gt, t)
    if mode == "full":
        sel = np.ones(len(t), bool)
    else:
        sel = t <= t[0] + window_s
    if mode == "origin":
        rot, trans = R.identity(), (gp[sel] - pos[sel]).mean(0)
    else:
        rot, trans = fit_4dof(pos[sel], gp[sel])
    p_al = rot.apply(pos) + trans
    q_al = rot * R.from_quat(quat)
    e = p_al - gp
    en = np.linalg.norm(e, axis=1)
    yaw_e = np.degrees(np.angle(np.exp(1j * (q_al.as_euler("ZYX")[:, 0]
                                             - gr.as_euler("ZYX")[:, 0]))))
    att_e = np.degrees((gr.inv() * q_al).magnitude())
    return {"samples": int(len(t)), "t0": float(t[0]), "t1": float(t[-1]),
            "pos_rmse_m": float(np.sqrt(np.mean(en ** 2))),
            "pos_p95_m": float(np.percentile(en, 95)), "pos_final_m": float(en[-1]),
            "horiz_rmse_m": float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1)))),
            "alt_rmse_m": float(np.sqrt(np.mean(e[:, 2] ** 2))),
            "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_e ** 2))),
            "att_rmse_deg": float(np.sqrt(np.mean(att_e ** 2))),
            "alignment": {"mode": mode, "window_s": window_s,
                          "yaw_deg": float(np.degrees(rot.as_euler("ZYX")[0])),
                          "translation": trans.round(3).tolist()}}


def relative_drift(track: Track, gt: np.ndarray, seg_m: float = 10.0) -> dict:
    """RPE-style translation drift per ``seg_m`` metres of travelled path (no alignment needed
    beyond the per-segment start pose)."""
    t = track.stamp
    gp, gr, _ = gt_at(gt, t)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
    errs = []
    i = 0
    while True:
        j = np.searchsorted(dist, dist[i] + seg_m)
        if j >= len(t):
            break
        d_est = R.from_quat(track.quat[i]).inv().apply(track.pos[j] - track.pos[i])
        d_gt = gr[i].inv().apply(gp[j] - gp[i])
        errs.append(np.linalg.norm(d_est - d_gt))
        i = j
    errs = np.asarray(errs)
    return {"segments": int(len(errs)),
            "drift_per_10m_mean_m": float(errs.mean()) if len(errs) else None,
            "drift_per_10m_p95_m": float(np.percentile(errs, 95)) if len(errs) else None}


def timing(track: Track, run_start: float = 0.0) -> dict:
    lat = track.recv - track.stamp
    dt = np.diff(track.stamp)
    return {"first_output_s": float(track.stamp[0] - run_start),
            "rate_hz": float(1.0 / np.median(dt)) if len(dt) else None,
            "max_gap_s": float(dt.max()) if len(dt) else None,
            "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
            "latency_note": "reception sim time - header stamp (/clock at 200 Hz: 5 ms quantised)"}
