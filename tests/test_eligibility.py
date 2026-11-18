"""IMU reception criterion shared by every selector (ijcas/selector/eligibility.py, 2026-11-18)."""
from pathlib import Path

import numpy as np

from ijcas.selector.common import load_config
from ijcas.selector.eligibility import ImuReception
from ijcas.selector.engine import HealthFsm
from ijcas.selector.facts import Output, Snapshot

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config/selector/thresholds_validated.yaml")
DT = 0.05


def _imu(t0, t1, rate=200.0):
    s = np.arange(t0 + 1.0 / rate, t1 + 1e-9, 1.0 / rate)
    rows = np.zeros((len(s), 7))
    rows[:, 0] = s
    rows[:, 6] = 9.80665
    return rows


def test_rate_and_startup():
    rx = ImuReception()
    t = 0.0
    oks = []
    for _ in range(40):                     # 2 s of a 200 Hz stream starting at 0
        rx.update(t + DT, _imu(t, t + DT))
        t += DT
        oks.append(rx.ok(t))
    assert all(oks)                          # partial window: rate over the elapsed time
    assert abs(rx.rate(t) - 200.0) < 1.0


def test_stream_stop_fails_within_window():
    rx = ImuReception()
    t = 0.0
    for _ in range(40):
        rx.update(t + DT, _imu(t, t + DT))
        t += DT
    gap = []
    for _ in range(30):                      # stream stops
        t += DT
        rx.update(t, np.zeros((0, 7)))
        gap.append(rx.ok(t))
    assert gap[:5] == [True] * 5             # <= 0.25 s silent: still >= 150 samples in 1 s
    assert not any(gap[6:])                  # >= 0.35 s silent: below 150 Hz
    assert rx.rate(t) == 0.0


def test_missing_rows_are_not_a_stream():
    rx = ImuReception()
    rx.update(0.5, None)
    assert not rx.ok(0.5) and rx.rate(0.5) == 0.0


def _snap(t, imu_rows):
    q = np.array([0.0, 0.0, 0.0, 1.0])
    out = {"px4": Output(t - 0.01, np.zeros(3), q, np.zeros(3))}
    return Snapshot(t=round(t, 6), outputs=out, output_history={}, diag={"gnss": ({"fix": True, "t": t}, t)},
                    px4_status=None, fix=None, imu_window=imu_rows)


def test_hfsm_needs_imu_reception():
    sel = HealthFsm(CFG)
    t = 0.0
    for _ in range(20):
        d, out = sel.step(_snap(t + DT, _imu(t, t + DT)))
        t += DT
    assert d.selected == "px4" and d.output_mode == "normal" and out is not None
    for _ in range(10):                      # IMU stops for 0.5 s: no candidate, no INS fallback
        t += DT
        d, out = sel.step(_snap(t, np.zeros((0, 7))))
    assert d.selected is None and d.output_mode == "no_valid_nav" and out is None
