"""Fault schedule for the IJCAS scenarios (pure python; used by the Isaac sensors and publishers).

A scenario declares a list of events; each event hits one target over [start_s, end_s) of
simulation time.  Sensors ask ``schedule.active(target, t)`` and apply what they get, so PX4 and
the ROS engines always see the same faulted measurement.  The labels themselves are ground
truth: they are published only under /ijcas/gt/ and must never reach a selector.

Kinds and parameters (all validated):

  gnss   outage            -                                  no fix (fix_type 0, no position)
         position_bias     vector_m [e,n,u], ramp_s           reported accuracy unchanged
         velocity_invalid  sigma_mps, bias_mps [e,n,u]
  camera brightness        scale (0..1]                       both cameras, research degradation
         occlusion         fraction (0..1), side left|right|both
         blur              kernel_px (odd >= 3)
         drop              probability (0..1)
         delay             mean_s, jitter_s
  lidar  dropout           fraction (0..1)
         range_limit       max_m
         drop              probability
         delay             mean_s, jitter_s
  imu    gyro_bias_step    rad_s [x,y,z] (FLU body)
         accel_bias_step   mps2 [x,y,z] (FLU body)
         saturation        gyro_rad_s, accel_mps2
         time_offset       offset_s (ROS stamps of the IMU only; PX4 keeps lockstep time)
  baro   loss | bias       bias_m
  mag    loss | bias       bias_gauss [x,y,z] (FRD body)
  gnss_ros delay           mean_s, jitter_s                    ROS-side latency only
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

KINDS: dict[str, dict[str, set[str]]] = {
    "gnss": {"outage": set(), "position_bias": {"vector_m", "ramp_s"},
             "velocity_invalid": {"sigma_mps", "bias_mps"}},
    "camera": {"brightness": {"scale"}, "occlusion": {"fraction", "side"},
               "blur": {"kernel_px"}, "drop": {"probability"}, "delay": {"mean_s", "jitter_s"}},
    "lidar": {"dropout": {"fraction"}, "range_limit": {"max_m"}, "drop": {"probability"},
              "delay": {"mean_s", "jitter_s"}},
    "imu": {"gyro_bias_step": {"rad_s"}, "accel_bias_step": {"mps2"},
            "saturation": {"gyro_rad_s", "accel_mps2"}, "time_offset": {"offset_s"}},
    "baro": {"loss": set(), "bias": {"bias_m"}},
    "mag": {"loss": set(), "bias": {"bias_gauss"}},
    "gnss_ros": {"delay": {"mean_s", "jitter_s"}},
}


@dataclass(frozen=True)
class FaultEvent:
    target: str
    kind: str
    start_s: float
    end_s: float = math.inf
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.target not in KINDS or self.kind not in KINDS[self.target]:
            raise ValueError(f"unknown fault {self.target}.{self.kind}")
        required = KINDS[self.target][self.kind]
        missing = required - set(self.params)
        extra = set(self.params) - required
        if missing or extra:
            raise ValueError(f"{self.target}.{self.kind}: missing {sorted(missing)}, "
                             f"unexpected {sorted(extra)}")
        if not self.end_s > self.start_s:
            raise ValueError(f"{self.target}.{self.kind}: end_s must be after start_s")

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.end_s

    def ramp(self, t: float) -> float:
        """0..1 progress of a ramped effect (position_bias)."""
        r = float(self.params.get("ramp_s", 0.0))
        return 1.0 if r <= 0.0 else min(1.0, max(0.0, (t - self.start_s) / r))

    def label(self) -> dict:
        return {"target": self.target, "kind": self.kind, "start_s": self.start_s,
                "end_s": None if math.isinf(self.end_s) else self.end_s,
                "params": self.params}


class FaultSchedule:
    def __init__(self, events=()):
        self.events: tuple[FaultEvent, ...] = tuple(events)

    @classmethod
    def from_config(cls, items) -> "FaultSchedule":
        events = []
        for item in items or []:
            item = dict(item)
            end = item.pop("end_s", None)
            events.append(FaultEvent(target=item.pop("target"), kind=item.pop("kind"),
                                     start_s=float(item.pop("start_s")),
                                     end_s=math.inf if end is None else float(end),
                                     params=dict(item.pop("params", {}))))
            if item:
                raise ValueError(f"unexpected fault keys {sorted(item)}")
        return cls(events)

    def active(self, target: str, t: float) -> list[FaultEvent]:
        return [e for e in self.events if e.target == target and e.active(t)]

    def first(self, target: str, kind: str, t: float) -> FaultEvent | None:
        for e in self.active(target, t):
            if e.kind == kind:
                return e
        return None

    def labels(self, t: float) -> list[dict]:
        return [e.label() for e in self.events if e.active(t)]

    def describe(self) -> list[dict]:
        return [e.label() for e in self.events]
