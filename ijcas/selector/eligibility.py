"""Eligibility criteria shared by every selector (docs/ONTOLOGY_REDESIGN_KO.md 5.1, 7.1).

IMU reception (adopted 2026-11-18, user decision).  Every candidate's estimation procedure requires
the IMU stream: ontology ``:requiresEssential :Inertial`` (v1, ijcas/knowledge/ijcas_nav.ttl) /
``nav:requiresInput nav:ImuSamples`` (v2 draft).  Before this criterion the learned selectors' policy
(ijcas/rgat/features.py CandidateBank) and H-FSM did not check the IMU at all.

The criterion is a *reception* check on the samples the selector itself received by time t (the
Snapshot IMU window, stamp <= t): rate over the last ``window_s`` >= ``min_rate_hz``.  Before the
selector has run for ``window_s`` the rate is taken over the elapsed time.  It is not an IMU quality
judgement: bias steps, saturation and clock offsets are not tested here (profile decision).
Default values are the frozen ``imu_rate`` indicator (invalid below 150 Hz, valid for 1 s;
config/selector/thresholds_validated.yaml); the nominal rate is 200 Hz.
"""
from __future__ import annotations

from collections import deque

IMU_MIN_RATE_HZ = 150.0
IMU_WINDOW_S = 1.0


class ImuReception:
    """Call ``update`` exactly once per selector tick, then ``ok`` / ``rate`` for that tick."""

    def __init__(self, min_rate_hz: float = IMU_MIN_RATE_HZ, window_s: float = IMU_WINDOW_S,
                 start_t: float = 0.0):
        self.min_rate_hz = float(min_rate_hz)
        self.window_s = float(window_s)
        self.start_t = float(start_t)        # since when the selector has been receiving (offline: sim 0)
        self.stamps: deque = deque()

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "ImuReception":
        e = ((cfg or {}).get("eligibility") or {}).get("imu_reception") or {}
        return cls(e.get("min_rate_hz", IMU_MIN_RATE_HZ), e.get("window_s", IMU_WINDOW_S))

    def update(self, t: float, imu_rows) -> None:
        if imu_rows is not None:
            for row in imu_rows:
                self.stamps.append(float(row[0]))
        lo = t - self.window_s
        while self.stamps and self.stamps[0] <= lo:
            self.stamps.popleft()

    def rate(self, t: float) -> float:
        span = min(self.window_s, t - self.start_t)
        return len(self.stamps) / span if span > 0 else 0.0

    def ok(self, t: float) -> bool:
        return self.rate(t) >= self.min_rate_hz

    def reason(self, t: float) -> str:
        r = self.rate(t)
        return (f"imu_reception:{r:.0f}Hz>={self.min_rate_hz:.0f}" if r >= self.min_rate_hz
                else f"imu_reception:{r:.0f}Hz<{self.min_rate_hz:.0f}")


IMU_SAT_FRAC = 0.05
IMU_SAT_VALID_S = 1.0
# re-alignment after a saturation is accepted only with a fit residual <= this (the 'mature'
# alignment limit already used for GNSS witnesses, ijcas/handover.py Alignment.mature)
IMU_SAT_REALIGN_MAX_RESIDUAL_M = 1.5


class ImuSaturation:
    """IMU quality criterion (adopted 2026-11-18, user decision): IMU saturation reported by the IMU
    diagnostic (``saturation_frac`` = share of samples pinned at the window maximum on any axis, one
    report per 0.5 s, ros2/context_monitor.py).  Saturated while a report with value > ``frac`` is at
    most ``valid_s`` old.  Threshold: every healthy run of the v1 data (390 runs, t >= 20 s) reported
    exactly 0.0; saturation-fault windows reach 0.99.

    The consequence follows the output's definition (applied by the selectors, not here):
      * output re-referenced by an absolute input (PX4 EKF2 global position <- GNSS): not excluded by
        this check; its own GNSS conditions apply.
      * local engines (VIO, LIO): their state may be corrupted, so their alignment is invalidated at
        every saturation report (like R08's essential-input loss).  They need a new alignment after the
        saturation ends whose fit residual is <= IMU_SAT_REALIGN_MAX_RESIDUAL_M (a diverged engine
        cannot be fitted), and cannot testify against GNSS until that alignment is mature again.
      * INS propagation: eligible only with an anchor taken after the last saturation report.
    A missing or stale diagnostic leaves the state undetermined: this check then excludes nothing
    (the IMU stream itself is checked by ImuReception).  Call ``update`` once per tick.
    """

    def __init__(self, frac: float = IMU_SAT_FRAC, valid_s: float = IMU_SAT_VALID_S, enabled: bool = False):
        # Disabled by default since 2026-11-18 17:00: on test1 (seeds 101-130, post hoc) it raised the
        # oracle RMSE 0.833 -> 1.321 m (docs/STOP_RECORD_20261118_KO.md).  Kept as an opt-in policy.
        self.enabled = bool(enabled)
        self.frac = float(frac)
        self.valid_s = float(valid_s)
        self.last_sat_t = float("-inf")       # source stamp of the latest saturated report
        self.last_report_t = float("-inf")

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "ImuSaturation":
        e = ((cfg or {}).get("eligibility") or {}).get("imu_saturation") or {}
        return cls(e.get("frac", IMU_SAT_FRAC), e.get("valid_s", IMU_SAT_VALID_S), e.get("enabled", False))

    def update(self, t: float, diag_imu) -> None:
        """diag_imu: Snapshot.diag['imu'] = (payload, reception time) or None."""
        if not diag_imu:
            return
        p = diag_imu[0]
        st = float(p["t"])
        self.last_report_t = max(self.last_report_t, st)
        v = p.get("saturation_frac")
        if v is not None and float(v) > self.frac:
            self.last_sat_t = max(self.last_sat_t, st)

    def active(self, t: float) -> bool:
        return self.enabled and t - self.last_sat_t <= self.valid_s

    def state(self, t: float) -> str:
        if self.active(t):
            return "saturated"
        return "ok" if t - self.last_report_t <= self.valid_s else "undetermined"

    def clean_since(self, anchor_t: float | None) -> bool:
        """True if a state anchored at ``anchor_t`` was taken after the last saturation report."""
        if not self.enabled:
            return True
        return anchor_t is not None and anchor_t > self.last_sat_t
