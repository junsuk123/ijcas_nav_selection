"""G3 step 3: R07 switching parameters on the validation set (after the thresholds are fixed).

    python3 tools/g3_selection_grid.py config/selector/thresholds_validated.yaml [--apply]

Grid: margin x minimum dwell.  Each setting replays Ontology-K (= Flat-K, G4) on every
validation run.  Rule fixed before looking at the numbers: among the settings whose mean
availability is within 0.005 of the best and whose mean false-normal time is within 0.5 s of
the lowest, take the one with the fewest mean switches (ties: smaller margin, shorter dwell).
--apply writes the chosen values into the thresholds file (still before any test-set replay).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
MARGINS = (0.08, 0.15, 0.25, 0.35, 0.5)   # extended after 0.25 came out at the grid edge
DWELLS = (2.0, 5.0)


def one(args):
    run, cfg = args
    from ijcas.replay import replay
    r = replay(run, cfg, only=("Ontology-K",))["selectors"]["Ontology-K"]
    return {k: r[k] for k in ("availability", "false_normal_s", "no_valid_nav_s", "switches",
                              "max_jump_m", "pos_rmse_m")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thresholds")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--reuse", default=None, help="earlier selection_grid.json: skip settings in it")
    a = ap.parse_args()
    base = yaml.safe_load(Path(a.thresholds).read_text())
    runs = sorted(p for p in (DATA / "runs/validation").iterdir() if (p / "DONE").exists())
    rows = json.loads(Path(a.reuse).read_text())["grid"] if a.reuse else []
    done = {(r["margin"], r["min_dwell_s"]) for r in rows}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, dw in itertools.product(MARGINS, DWELLS):
            if (m, dw) in done:
                continue
            cfg = copy.deepcopy(base)
            cfg["selection"]["margin"], cfg["selection"]["min_dwell_s"] = m, dw
            res = list(ex.map(one, [(r, cfg) for r in runs]))
            agg = {k: float(np.mean([x[k] for x in res if x[k] is not None])) for k in res[0]}
            rows.append({"margin": m, "min_dwell_s": dw, **agg})
            print(json.dumps(rows[-1]), flush=True)
    best_av = max(r["availability"] for r in rows)
    best_fn = min(r["false_normal_s"] for r in rows)
    ok = [r for r in rows if r["availability"] >= best_av - 0.005 and r["false_normal_s"] <= best_fn + 0.5]
    choice = min(ok, key=lambda r: (r["switches"], r["margin"], r["min_dwell_s"]))
    out = {"grid": rows, "rule": "avail within 0.005 of best, false-normal within 0.5 s of best, "
                                 "then fewest switches", "choice": choice}
    (ROOT / "results/g3").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/g3/selection_grid.json").write_text(json.dumps(out, indent=1))
    print("CHOICE", json.dumps(choice))
    if a.apply:
        txt = Path(a.thresholds).read_text()
        base["selection"]["margin"] = choice["margin"]
        base["selection"]["min_dwell_s"] = choice["min_dwell_s"]
        head = "".join(l for l in txt.splitlines(True) if l.startswith("#"))
        Path(a.thresholds).write_text(head + yaml.safe_dump(base, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
