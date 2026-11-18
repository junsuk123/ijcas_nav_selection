"""IMU reception criterion shared by every selector (ijcas/selector/eligibility.py, 2026-11-18)."""
from pathlib import Path

import numpy as np

from ijcas.selector.common import load_config
from ijcas.selector.eligibility import ImuReception
from ijcas.selector.engine import HealthFsm
from ijcas.selector.facts import Output, Snapshot

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config/selector/thresholds_validated.yaml")
CFG_SAT = {**CFG, "eligibility": {"imu_saturation": {"enabled": True}}}   # opt-in policy
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


# ---- IMU quality: saturation (v2 policy, 2026-11-18) -------------------------------------------
from ijcas.selector.eligibility import ImuSaturation  # noqa: E402
from ijcas.selector.engine import SelectionEngine, _NullReasoner  # noqa: E402


def _diag(t, frac):
    return ({"t": t, "saturation_frac": frac, "rate_hz": 200.0, "non_monotonic": 0,
             "stamp_minus_clock_s": 0.0}, t)


def test_saturation_policy_is_off_by_default():
    s = ImuSaturation.from_cfg(CFG)
    s.update(5.0, _diag(5.0, 0.9))
    assert not s.active(5.1) and s.clean_since(None)


def test_saturation_state_and_anchor_rule():
    s = ImuSaturation(enabled=True)
    assert s.state(1.0) == "undetermined"            # no report yet: excludes nothing
    s.update(1.0, _diag(1.0, 0.0))
    assert s.state(1.2) == "ok" and not s.active(1.2)
    s.update(5.0, _diag(5.0, 0.6))
    assert s.active(5.5) and s.state(5.5) == "saturated"
    assert not s.active(6.1)                          # report older than 1 s
    assert not s.clean_since(4.0) and s.clean_since(5.2) and not s.clean_since(None)
    s.update(7.0, _diag(7.0, 0.04))                   # below the 0.05 threshold
    assert s.last_sat_t == 5.0


def test_saturation_invalidates_local_alignment():
    eng = SelectionEngine(_NullReasoner(), CFG_SAT, name="t")
    al = eng.align["fastlio"]
    al.yaw, al.trans, al.updated_at, al.established_at = 0.0, np.zeros(3), 1.0, 1.0
    assert al.valid
    snap = Snapshot(t=10.0, outputs={}, output_history={}, diag={"imu": _diag(9.9, 0.9)},
                    px4_status=None, fix=None, imu_window=None)
    eng._update_alignment(snap, {})
    assert not al.valid and al.invalidations[-1] == (10.0, "imu_saturation")
    assert not eng.align["openvins"].valid


def test_hfsm_no_ins_fallback_with_anchor_before_saturation():
    sel = HealthFsm(CFG_SAT)
    t = 0.0
    for _ in range(20):                               # px4 selected -> DR anchored on it
        d, _ = sel.step(_snap(t + DT, _imu(t, t + DT)))
        t += DT
    assert d.selected == "px4" and sel.inner.dr.valid
    q = np.array([0.0, 0.0, 0.0, 1.0])
    for _ in range(4):                                # px4 output stops, IMU saturated (report)
        t += DT
        snap = Snapshot(t=round(t, 6), outputs={"px4": Output(0.5, np.zeros(3), q, np.zeros(3))},
                        output_history={}, diag={"imu": _diag(t, 0.8)}, px4_status=None, fix=None,
                        imu_window=_imu(t - DT, t))
        d, out = sel.step(snap)
    assert d.selected is None and d.output_mode == "no_valid_nav"


def test_realignment_after_saturation_needs_small_residual():
    from ijcas.handover import Alignment
    al = Alignment()
    al.yaw, al.trans, al.residual_m = 0.0, np.zeros(3), 3.0
    assert al.valid                                   # no saturation seen: residual not gated
    al.require_residual_m = 1.5
    assert not al.valid
    al.residual_m = 0.8
    assert al.valid
