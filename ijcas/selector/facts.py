"""knowledge_state input: causal fact snapshots at the selector rate.

Offline (G4-G6) the snapshots are rebuilt from a run record using *reception* times, so a
selector at time t sees exactly what had arrived by t (output age = t - source stamp).  The same
``Snapshot`` is produced online by the ROS node.  Ground truth, scenario ids and fault labels are
not inputs: ``FactStream`` refuses record keys under the gt/ namespace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CANDIDATES = ("px4", "openvins", "fastlio")
# nav_msgs/Odometry twist frames as published (checked on recorded data, results/g5):
# OpenVINS reports the IMU-frame (body) velocity; FAST-LIO2 (IJCAS output adapter) and the PX4 republisher the odom/world one.
TWIST_IN_BODY = {"openvins": True, "fastlio": False, "px4": False}


def make_output(name, stamp, pos, quat, vel) -> "Output":
    vel = np.asarray(vel, float)
    if TWIST_IN_BODY.get(name):
        from scipy.spatial.transform import Rotation
        vel = Rotation.from_quat(quat).apply(vel)
    return Output(stamp, np.asarray(pos, float), np.asarray(quat, float), vel)
DIAG_TOPICS = {"vision": "/ijcas/diag/vision", "lidar": "/ijcas/diag/lidar",
               "gnss": "/ijcas/diag/gnss", "imu": "/ijcas/diag/imu"}
FORBIDDEN = ("odom_gt", "/ijcas/gt/faults", "gt")


@dataclass
class Output:
    stamp: float
    pos: np.ndarray
    quat: np.ndarray
    vel: np.ndarray


@dataclass
class Snapshot:
    t: float
    outputs: dict = field(default_factory=dict)        # name -> Output (latest received)
    output_history: dict = field(default_factory=dict)  # name -> list[Output] since last tick
    diag: dict = field(default_factory=dict)            # kind -> (payload, recv)
    px4_status: tuple | None = None                     # (payload, recv)
    fix: tuple | None = None                            # latest GNSS fix row
    imu_window: np.ndarray | None = None                # IMU rows received since last tick


class FactStream:
    def __init__(self, run_dir: str | Path, rate_hz: float = 10.0):
        run = Path(run_dir)
        rec = np.load(run / "record.npz")
        js = json.loads((run / "record_json.json").read_text())
        self.odom = {}
        for name in CANDIDATES:
            a = rec[f"odom_{name}"]
            self.odom[name] = a[np.argsort(a[:, 1])] if a.size else np.zeros((0, 18))
        self.imu = rec["imu"]
        self.fix = rec["fix"]
        self.json = {k: v for k, v in js.items() if k in DIAG_TOPICS.values()
                     or k == "/ijcas/px4/status"}
        for k in self.json:
            if any(f in k for f in FORBIDDEN):
                raise RuntimeError(f"ground-truth topic {k} must not feed the selector")
        self.dt = 1.0 / rate_hz
        man = json.loads((run / "manifest.json").read_text())
        self.duration = float(man["sim_s"])
        self.origin = man["origin"]

    def __iter__(self):
        idx = {n: 0 for n in CANDIDATES}
        jidx = {k: 0 for k in self.json}
        imu_i = 0
        fix_i = 0
        latest = {}
        diag = {}
        px4 = None
        fix = None
        t = self.dt
        while t <= self.duration + 1e-9:
            hist = {}
            for n, a in self.odom.items():
                j = idx[n]
                rows = []
                while j < len(a) and a[j, 1] <= t:
                    rows.append(a[j])
                    j += 1
                idx[n] = j
                if rows:
                    outs = [make_output(n, r[0], r[2:5], r[5:9], r[9:12]) for r in rows]
                    hist[n] = outs
                    latest[n] = outs[-1]
            for k, items in self.json.items():
                j = jidx[k]
                while j < len(items) and items[j][0] <= t:
                    payload = items[j][1]
                    if k == "/ijcas/px4/status":
                        px4 = (payload, items[j][0])
                    else:
                        kind = k.rsplit("/", 1)[-1]
                        diag[kind] = (payload, items[j][0])
                    j += 1
                jidx[k] = j
            while fix_i < len(self.fix) and self.fix[fix_i, 1] <= t:
                fix = self.fix[fix_i]
                fix_i += 1
            j0 = imu_i
            while imu_i < len(self.imu) and self.imu[imu_i, 0] <= t:
                imu_i += 1
            yield Snapshot(t=round(t, 6), outputs=dict(latest), output_history=hist,
                           diag=dict(diag), px4_status=px4, fix=fix,
                           imu_window=self.imu[j0:imu_i])
            t += self.dt
