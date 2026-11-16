"""Camera / LiDAR degradations of the fault schedule (pure numpy + OpenCV, testable outside Isaac).

These are research degradations (Notion 04): image brightness scaling and occlusion are not a
photometric model of night or fog, and LiDAR point dropout / range limits are not a weather model.
"""
from __future__ import annotations

import numpy as np

from .faults import FaultSchedule


def degrade_image(gray: np.ndarray, name: str, t: float, faults: FaultSchedule,
                  rng: np.random.Generator) -> tuple[np.ndarray | None, float]:
    """(image or None if dropped, extra delay s)."""
    img = gray
    delay = 0.0
    for e in faults.active("camera", t):
        if e.kind == "drop" and rng.random() < float(e.params["probability"]):
            return None, 0.0
        if e.kind == "brightness":
            img = np.clip(img.astype(np.float32) * float(e.params["scale"]), 0, 255).astype(np.uint8)
        elif e.kind == "occlusion":
            side = e.params["side"]
            if side == "both" or (side == "left" and name == "cam0") or \
                    (side == "right" and name == "cam1"):
                img = img.copy()
                w = int(round(img.shape[1] * float(e.params["fraction"])))
                if w > 0:
                    img[:, :w] = 0
        elif e.kind == "blur":
            import cv2
            k = int(e.params["kernel_px"]) | 1
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif e.kind == "delay":
            delay += max(0.0, float(e.params["mean_s"]) + float(e.params["jitter_s"])
                         * rng.standard_normal())
    return img, delay


def degrade_scan(xyz: np.ndarray, inten: np.ndarray, rel_t: np.ndarray, t: float,
                 faults: FaultSchedule, rng: np.random.Generator):
    """(xyz, intensity, rel_t) or None if the scan is dropped, plus extra delay s."""
    delay = 0.0
    keep = np.ones(len(xyz), bool)
    for e in faults.active("lidar", t):
        if e.kind == "drop" and rng.random() < float(e.params["probability"]):
            return None, 0.0
        if e.kind == "dropout":
            keep &= rng.random(len(xyz)) >= float(e.params["fraction"])
        elif e.kind == "range_limit":
            keep &= np.linalg.norm(xyz, axis=1) <= float(e.params["max_m"])
        elif e.kind == "delay":
            delay += max(0.0, float(e.params["mean_s"]) + float(e.params["jitter_s"])
                         * rng.standard_normal())
    return (xyz[keep], inten[keep], rel_t[keep]), delay


class DelayQueue:
    """Holds messages until simulation time reaches their release time (stamps unchanged)."""

    def __init__(self):
        self.items: list = []

    def push(self, release_t: float, payload):
        self.items.append((release_t, payload))

    def pop_due(self, now: float) -> list:
        due = [p for r, p in self.items if r <= now]
        self.items = [(r, p) for r, p in self.items if r > now]
        return due
