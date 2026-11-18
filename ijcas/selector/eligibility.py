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
