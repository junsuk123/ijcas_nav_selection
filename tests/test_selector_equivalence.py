"""G4: Flat-K and Ontology-K are logically equivalent on identical inputs (Notion 03: flat rules
with the same input/knowledge/thresholds must give the same decision; H3).

Fuzzed snapshots sweep every indicator across its thresholds, stale/missing facts, output ages,
alignment validity/age and the dead-reckoning memory; states must match on every tick.  If a run
record exists ($IJCAS_DATA/g2/S00_corridor_1), the full engines are also replayed side by side.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ijcas.selector.common import load_config  # noqa: E402
from ijcas.selector.facts import Output, Snapshot  # noqa: E402
from ijcas.selector.flat import FlatRules  # noqa: E402
from ijcas.selector.ontology import OntologyReasoner  # noqa: E402

# design thresholds and the G3-validated ones (indicators disabled / levels removed there)
THRESHOLDS = [None, str(ROOT / "config/selector/thresholds_validated.yaml")]


class FakeAlign:
    def __init__(self, valid, updated):
        self.valid, self.updated = valid, updated

    def age(self, t):
        return float("inf") if not self.valid else t - self.updated


def random_snapshot(rng, t):
    def maybe(p=0.85):
        return rng.random() < p
    diag = {}
    if maybe():
        st = t - rng.choice([0.05, 0.3, 0.8, 2.0])
        payload = {"t": st, "fix": bool(maybe(0.8))}
        if maybe():
            payload["pos_vel_residual_m"] = float(rng.uniform(0, 4))
        diag["gnss"] = (payload, t)
    if maybe():
        st = t - rng.choice([0.03, 0.2, 0.6, 1.5])
        payload = {"t": st, "features": float(rng.uniform(0, 1500)),
                   "grid_cells_ge10": float(rng.integers(0, 16)), "mean": float(rng.uniform(0, 200)),
                   "stereo_matches": None if not maybe(0.8) else float(rng.uniform(0, 300))}
        if maybe():
            payload.update(ov_points_msckf=int(rng.integers(0, 30)), ov_points_slam=int(rng.integers(0, 30)),
                           ov_points_msckf_t=st, ov_points_slam_t=st)
        diag["vision"] = (payload, t)
    if maybe():
        st = t - rng.choice([0.05, 0.3, 0.7, 2.0])
        diag["lidar"] = ({"t": st, "points": float(rng.uniform(0, 40000)),
                          "lambda_ratio": float(rng.uniform(0, 0.8))}, t)
    if maybe():
        st = t - rng.choice([0.1, 0.5, 1.5])
        payload = {"t": st, "rate_hz": float(rng.choice([200, 120])),
                   "non_monotonic": int(rng.choice([0, 0, 0, 2])),
                   "stamp_minus_clock_s": float(rng.choice([0.0, 0.0, 0.002, 0.03, -0.04, -0.2]))}
        if maybe():
            payload["saturation_frac"] = float(rng.choice([0.0, 0.005, 0.005, 0.05, 0.3]))
        if maybe():
            payload["mag_rate_hz"] = float(rng.choice([50, 50, 0]))
            payload["baro_rate_hz"] = float(rng.choice([50, 50, 0]))
        diag["imu"] = (payload, t)
    px4 = None
    if maybe():
        px4 = ({"t": t - rng.choice([0.05, 0.5, 2.0]), "flags": int(rng.choice([831, 831, 815, 0])),
                "vel_ratio": float(rng.uniform(0, 1.5)), "pos_horiz_ratio": float(rng.uniform(0, 1.5))}, t)
    outs = {}
    for c in ("px4", "openvins", "glim"):
        if maybe(0.9):
            outs[c] = Output(t - float(rng.choice([0.01, 0.1, 0.25, 0.4, 2.0])), np.zeros(3),
                             np.array([0, 0, 0, 1.0]), np.zeros(3))
    return Snapshot(t=t, outputs=outs, diag=diag, px4_status=px4)


@pytest.mark.parametrize("thr", THRESHOLDS)
@pytest.mark.parametrize("task", ["GlobalWaypoint", "LocalFlight"])
def test_fuzzed_equivalence(task, thr):
    cfg = load_config(thr)
    onto, flat = OntologyReasoner(cfg), FlatRules(cfg)
    rng = np.random.default_rng(42)
    mismatches = []
    t = 0.0
    for k in range(4000):
        t += 0.1
        snap = random_snapshot(rng, t)
        from ijcas.selector.common import signals
        sig = signals(snap)
        align = {"openvins": FakeAlign(bool(rng.random() < 0.7), t - rng.choice([0.0, 2.0, 40.0])),
                 "glim": FakeAlign(bool(rng.random() < 0.7), t - rng.choice([0.0, 2.0, 40.0])),
                 "ins_dr": FakeAlign(bool(rng.random() < 0.5), t - rng.choice([0.0, 5.0]))}
        so, _, mo = onto.evaluate(task, sig, t, snap.outputs, align, 30.0)
        sf, _, mf = flat.evaluate(task, sig, t, snap.outputs, align, 30.0)
        if so != sf:
            mismatches.append((k, so, sf))
        common = set(mo) & set(mf)
        assert all(mo[c] == mf[c] for c in common), (k, {c: (mo[c], mf[c]) for c in common})
    assert not mismatches, mismatches[:5]


RUN = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data")) / "g2/S00_corridor_1"


@pytest.mark.skipif(not (RUN / "record.npz").exists(), reason="no recorded run")
@pytest.mark.parametrize("thr", THRESHOLDS)
def test_recorded_run_equivalence(thr):
    from ijcas.selector.engine import SelectionEngine
    from ijcas.selector.facts import FactStream
    cfg = load_config(thr)
    a = SelectionEngine(OntologyReasoner(cfg), cfg)
    b = SelectionEngine(FlatRules(cfg), cfg)
    n = 0
    for snap in FactStream(RUN, 10.0):
        da, oa = a.step(snap)
        db, ob = b.step(snap)
        assert da.states == db.states and da.selected == db.selected and \
            da.output_mode == db.output_mode, (snap.t, da.states, db.states)
        n += 1
    assert n > 1000
