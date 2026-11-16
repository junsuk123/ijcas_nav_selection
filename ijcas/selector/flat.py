"""Flat-K: the same inputs, domain knowledge and thresholds as Ontology-K, written as flat rules.

Every dependency the ontology declares (which indicators assess which measurement, which
measurements each estimator needs, AND/OR provisions, dead-reckoning limits, alignment, the shared
IMU) is hard-coded here per candidate.  The output must equal Ontology-K's on every tick
(tests/test_selector_equivalence.py); the comparison of the two is about maintenance (rule edits
for a new sensor/task, omissions, traceability), not accuracy (Notion 03, H3).
"""
from __future__ import annotations

from .common import LIMITED, UNKNOWN, UNUSABLE, USABLE, best, indicator_state, worst

TASKS = {"LocalFlight": ("Attitude", "LocalPosition", "Velocity"),
         "GlobalWaypoint": ("Attitude", "GlobalPosition", "Velocity")}
MAX_AGE = {"LocalFlight": 0.3, "GlobalWaypoint": 0.3}


class FlatRules:
    def __init__(self, cfg: dict):
        self.ind = {i["id"]: i for i in cfg["indicators"]}
        self.last_usable: dict = {}

    def _m(self, ids, sig, t):
        return worst(indicator_state(self.ind[i], sig, t)[0] for i in ids)

    def measurements(self, sig, t) -> dict:
        m = self._m
        return {
            "GnssPosition": m(["gnss_fix", "gnss_local_consistency", "gnss_pos_vel_residual",
                               "px4_pos_ratio"], sig, t),
            "GnssVelocity": m(["gnss_fix", "px4_vel_ratio"], sig, t),
            "VisualFeatures": m(["vis_age", "vis_brightness", "vis_coverage", "vis_features",
                                 "vis_stereo"], sig, t),
            "LidarStructure": m(["lidar_age", "lidar_geometry", "lidar_points"], sig, t),
            "Inertial": m(["imu_clock_offset", "imu_monotonic", "imu_rate", "imu_saturation"], sig, t),
            "OpenVINS": m(["ov_tracking"], sig, t),
            "PX4_EKF2": m(["px4_abs_pos_flag"], sig, t),
            "MagHeading": m(["mag_rate"], sig, t),
            "BaroHeight": m(["baro_rate"], sig, t),
        }

    def _dr(self, key, s, t, limit):
        if s == USABLE:
            self.last_usable[key] = t
        if limit == 0.0 and s == USABLE:
            return LIMITED
        if s in (UNUSABLE, UNKNOWN) and t - self.last_usable.get(key, -1e9) <= limit:
            return LIMITED
        return s

    @staticmethod
    def _align(s, al, t, stale):
        if al is None or not al.valid:
            return UNUSABLE
        if al.age(t) > stale or al.age(t) > 0.5:
            return worst([s, LIMITED])
        return s

    def evaluate(self, task, sig, t, outputs, alignment, stale_alignment_s):
        ms = self.measurements(sig, t)
        caps = TASKS[task]
        imu = ms["Inertial"]
        states = {}

        def gate(cid, essentials, extra=None):
            st = USABLE
            for e in essentials:
                if ms[e] in (UNUSABLE, UNKNOWN):
                    st = worst([st, ms[e]])
            if extra is not None:
                st = worst([st, extra])
            if cid != "ins_dr":
                out = outputs.get(cid)
                if out is None or t - out.stamp > MAX_AGE[task]:
                    st = UNUSABLE
            return st

        # PX4 EKF2 ---------------------------------------------------------------------------
        st = gate("px4", ["Inertial"], ms["PX4_EKF2"])
        cs = []
        for cap in caps:
            if cap == "GlobalPosition":
                s = self._dr(("PX4_EKF2", cap), ms["GnssPosition"], t, 5.0)
            elif cap == "LocalPosition":
                s = self._dr(("PX4_EKF2", cap), best([ms["GnssPosition"], ms["GnssVelocity"]]), t, 5.0)
            elif cap == "Velocity":
                s = self._dr(("PX4_EKF2", cap), best([ms["GnssVelocity"], ms["GnssPosition"]]), t, 5.0)
            elif cap == "Attitude":
                s = imu if ms.get("MagHeading", UNKNOWN) == USABLE else worst([imu, LIMITED])
            cs.append(s)
        states["px4"] = worst([st] + cs + ([imu] if imu != USABLE else []))
        # OpenVINS ---------------------------------------------------------------------------
        st = gate("openvins", ["Inertial", "VisualFeatures"], ms["OpenVINS"])
        cs = []
        for cap in caps:
            s = ms["VisualFeatures"]
            if cap == "GlobalPosition":
                s = self._align(s, alignment.get("openvins"), t, stale_alignment_s)
            cs.append(s)
        states["openvins"] = worst([st] + cs + ([imu] if imu != USABLE else []))
        # GLIM -------------------------------------------------------------------------------
        st = gate("glim", ["Inertial", "LidarStructure"])
        cs = []
        for cap in caps:
            s = ms["LidarStructure"]
            if cap == "GlobalPosition":
                s = self._align(s, alignment.get("glim"), t, stale_alignment_s)
            cs.append(s)
        states["glim"] = worst([st] + cs + ([imu] if imu != USABLE else []))
        # INS dead reckoning --------------------------------------------------------------------
        st = gate("ins_dr", ["Inertial"])
        cs = []
        for cap in caps:
            s = self._dr(("INS_DR", cap), imu, t, 0.0)
            if cap == "GlobalPosition":
                s = self._align(s, alignment.get("ins_dr"), t, stale_alignment_s)
            cs.append(s)
        states["ins_dr"] = worst([st] + cs + ([imu] if imu != USABLE else []))
        return states, {k: [] for k in states}, ms
