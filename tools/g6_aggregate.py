"""G6: replay every selector on every run of a set and aggregate (Notion 04 통계).

    python3 tools/g6_aggregate.py --set test --thresholds config/selector/thresholds_validated.yaml

Per run (the independent unit; ticks are never counted as samples): replay metrics of
ijcas.replay plus fault-based metrics computed with the scenario labels (evaluation only):
  detection_delay_s  fault onset -> the affected candidate leaves 'Usable' in the selector's
                     states (Flat-K / Ontology-K) or stops being selected (others)
  false_alarm_share  S00 only: share of ticks a candidate is marked not usable while its own
                     global error is below half the alarm limit
Aggregates: per scenario and selector the mean over runs, and the paired difference to
Ontology-K with a run-level bootstrap 95 % CI (10 000 resamples); per scene as well.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
AFFECTS = {"gnss": ["px4"], "camera": ["openvins"], "lidar": ["fastlio"],
           "imu": ["px4", "openvins", "fastlio"]}
METRICS = ["availability", "output_fraction", "false_normal_s", "no_valid_nav_s",
           "pos_rmse_m", "pos_p95_m", "switches", "max_jump_m", "median_jump_m", "max_vel_jump_mps",
           "max_yaw_jump_deg", "limited_dr_s"]


def one(args):
    run, thr = args
    from ijcas.replay import replay
    from ijcas.selector.common import load_config
    cfg = load_config(thr)
    try:
        r = replay(run, cfg, keep_trace=True)
    except Exception as e:  # recorded with the reason
        return {"run": str(run), "error": f"{type(e).__name__}: {e}"}
    man = json.loads((Path(run) / "manifest.json").read_text())
    faults = man["scenario"]["faults"]
    for name, s in r["selectors"].items():
        tr = s.pop("trace")
        delays = []
        for f in faults:
            for cand in AFFECTS.get(f["target"], []):
                t0 = f["start_s"]
                after = [x for x in tr if x[0] >= t0]
                # comparators: only meaningful if the affected candidate was selected at onset
                hit = None
                if name not in ("Flat-K", "Ontology-K") and after and after[0][1] == cand:
                    hit = next((x[0] for x in after if x[1] != cand), None)
                delays.append({"target": f["target"], "kind": f["kind"], "candidate": cand,
                               "delay_s": None if hit is None else hit - t0})
        s["fault_events"] = delays
    return r


def states_delay(run, thr):
    """Detection delays from the reasoners' candidate states (needs a second, state-level pass)."""
    from ijcas.selector.common import USABLE, load_config
    from ijcas.selector.engine import SelectionEngine
    from ijcas.selector.facts import FactStream
    from ijcas.selector.flat import FlatRules
    from ijcas.selector.ontology import OntologyReasoner
    cfg = load_config(thr)
    man = json.loads((Path(run) / "manifest.json").read_text())
    faults = man["scenario"]["faults"]
    out = {}
    for name, mk in (("Ontology-K", lambda: OntologyReasoner(cfg)), ("Flat-K", lambda: FlatRules(cfg))):
        eng = SelectionEngine(mk(), cfg)
        states = []
        for snap in FactStream(run, cfg["selection"]["rate_hz"]):
            d, _ = eng.step(snap)
            states.append((snap.t, d.states))
        ev = []
        for f in faults:
            for cand in AFFECTS.get(f["target"], []):
                hit = next((t for t, st in states if t >= f["start_s"] and st.get(cand) != USABLE), None)
                ev.append({"target": f["target"], "kind": f["kind"], "candidate": cand,
                           "delay_s": None if hit is None else hit - f["start_s"]})
        out[name] = ev
    return out


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def aggregate(results):
    by = {}
    for r in results:
        if "error" in r:
            continue
        by.setdefault(r["scenario"], []).append(r)
    agg = {}
    for sid, runs in sorted(by.items()):
        sels = runs[0]["selectors"].keys()
        a = {"runs": len(runs), "scenes": sorted({r["scene"] for r in runs}), "selectors": {}}
        for name in sels:
            m = {}
            for k in METRICS:
                vals = [r["selectors"][name][k] for r in runs]
                vals = [v for v in vals if v is not None]
                m[k] = float(np.mean(vals)) if vals else None
                if name != "Ontology-K":
                    diffs = [r["selectors"][name][k] - r["selectors"]["Ontology-K"][k] for r in runs
                             if r["selectors"][name][k] is not None and
                             r["selectors"]["Ontology-K"][k] is not None]
                    m[f"{k}_minus_ontology"] = {"mean": float(np.mean(diffs)) if diffs else None,
                                                "ci95": bootstrap_ci(diffs)}
            delays = [e["delay_s"] for r in runs for e in r["selectors"][name].get("fault_events", [])
                      if e["delay_s"] is not None]
            m["detection_delay_median_s"] = float(np.median(delays)) if delays else None
            m["per_scene_availability"] = {
                sc: float(np.mean([r["selectors"][name]["availability"] for r in runs if r["scene"] == sc]))
                for sc in a["scenes"]}
            a["selectors"][name] = m
        agg[sid] = a
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = sorted(p for p in (DATA / "runs" / a.set).iterdir() if (p / "DONE").exists())
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, [(r, a.thresholds) for r in runs]))
        sdel = list(ex.map(states_delay, runs, [a.thresholds] * len(runs)))
    for r, sd in zip(results, sdel):
        if "error" in r:
            continue
        for name, ev in sd.items():
            r["selectors"][name]["fault_events"] = ev
    failed = [r for r in results if "error" in r]
    out = Path(a.out or ROOT / f"results/g6/{a.set}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_run.json").write_text(json.dumps(results, indent=1, default=str))
    agg = aggregate(results)
    (out / "aggregate.json").write_text(json.dumps({"failed_runs": failed, "scenarios": agg},
                                                   indent=1, default=str))
    for sid, s in agg.items():
        print(f"== {sid} runs {s['runs']} scenes {s['scenes']}")
        for name, m in s["selectors"].items():
            d = m.get("availability_minus_ontology", {})
            print(f"  {name:11s} avail {m['availability']:.3f} false-normal {m['false_normal_s']:6.1f}s "
                  f"noNav {m['no_valid_nav_s']:5.1f}s rmse {m['pos_rmse_m'] if m['pos_rmse_m'] is None else round(m['pos_rmse_m'], 2)} "
                  f"sw {m['switches']:.1f} det {m['detection_delay_median_s']} "
                  f"dAvail {None if not d else d}")
    print("failed runs:", len(failed))


if __name__ == "__main__":
    main()
