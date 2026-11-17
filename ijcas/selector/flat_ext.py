"""Flat-K with the same two knowledge changes as ijcas/knowledge/changes/*.ttl (H3 experiment).
Every line below is an edit the flat model needs; the ontology needs none in code."""
from .common import UNKNOWN, UNUSABLE, USABLE, best, worst
from .flat import MAX_AGE, TASKS, FlatRules

TASKS_EXT = {**TASKS, "HeightHold": ("Height", "Velocity")}
MAX_AGE_EXT = {**MAX_AGE, "HeightHold": 0.3}


class FlatRulesExt(FlatRules):
    def evaluate(self, task, sig, t, outputs, alignment, stale):
        if task != "HeightHold":
            states, reasons, ms = super().evaluate(task, sig, t, outputs, alignment, stale)
        else:
            ms = self.measurements(sig, t)
            imu = ms["Inertial"]
            states = {}

            def gate(cid, ess, extra=None):
                st = USABLE
                for e in ess:
                    if ms[e] in (UNUSABLE, UNKNOWN):
                        st = worst([st, ms[e]])
                if extra is not None:
                    st = worst([st, extra])
                if cid != "ins_dr":
                    o = outputs.get(cid)
                    if o is None or t - o.stamp > MAX_AGE_EXT[task]:
                        st = UNUSABLE
                return st
            px4_v = self._dr(("PX4_EKF2", "Velocity"), best([ms["GnssVelocity"], ms["GnssPosition"]]), t, 5.0)
            px4_h = best([ms["GnssPosition"], ms["BaroHeight"]])
            states["px4"] = worst([gate("px4", ["Inertial"], ms["PX4_EKF2"]), px4_h, px4_v]
                                  + ([imu] if imu != USABLE else []))
            states["openvins"] = worst([gate("openvins", ["Inertial", "VisualFeatures"], ms["OpenVINS"]),
                                        ms["VisualFeatures"]] + ([imu] if imu != USABLE else []))
            states["fastlio"] = worst([gate("fastlio", ["Inertial", "LidarStructure"]), ms["LidarStructure"]]
                                   + ([imu] if imu != USABLE else []))
            dr_v = self._dr(("INS_DR", "Velocity"), imu, t, 0.0)
            states["ins_dr"] = worst([gate("ins_dr", ["Inertial"]), UNUSABLE, dr_v]
                                     + ([imu] if imu != USABLE else []))   # no Height provision
            reasons = {k: [] for k in states}
        # C2: GNSS-only candidate
        caps = TASKS_EXT[task]
        g = ms["GnssPosition"] if ms["GnssPosition"] in (UNUSABLE, UNKNOWN) else USABLE
        if outputs.get("gnss_only") is None or t - outputs["gnss_only"].stamp > MAX_AGE_EXT[task]:
            g = UNUSABLE
        cap_states = []
        for cap in caps:
            if cap in ("GlobalPosition", "Height"):
                cap_states.append(ms["GnssPosition"])
            elif cap == "Velocity":
                cap_states.append(ms["GnssVelocity"])
            else:
                cap_states.append(UNUSABLE)                      # no attitude
        states["gnss_only"] = worst([g] + cap_states)
        reasons["gnss_only"] = []
        return states, reasons, ms
