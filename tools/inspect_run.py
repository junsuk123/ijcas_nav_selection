"""Diagnosis: selection timeline of one run (segments of constant selection) with errors.

    python3 tools/inspect_run.py RUN_DIR [--selector Ontology-K] [--thresholds FILE]

Each line: [t0, t1) selected / output mode, mean and max global error of the output, the states
of all candidates at the segment start, and the global error of every candidate (truth used for
the printout only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.evaluation import gt_at  # noqa: E402
from ijcas.replay import make_selectors, px4_to_world  # noqa: E402
from ijcas.selector.common import load_config  # noqa: E402
from ijcas.selector.facts import FactStream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--selector", default="Ontology-K")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--reasons", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = load_config(a.thresholds)
    gt = np.load(run / "gt.npz")["gt"]
    man = json.loads((run / "manifest.json").read_text())
    offset, _ = px4_to_world(run, gt, man)
    sel = make_selectors(cfg, gt, offset)[a.selector]
    print("faults:", json.dumps(man["scenario"]["faults"]))
    rows = []
    for snap in FactStream(run, cfg["selection"]["rate_hz"]):
        d, out = sel.step(snap)
        gp = gt_at(gt, np.array([snap.t]))[0][0]
        err = None if out is None else float(np.linalg.norm(np.asarray(out[0]) + offset - gp))
        cerr = {}
        if hasattr(sel, "global_state"):
            for c in ("px4", "openvins", "glim"):
                g = sel.global_state(c, snap)
                cerr[c] = None if g is None else round(float(np.linalg.norm(g[0] + offset - gp)), 1)
        rows.append((snap.t, d.selected, d.output_mode, err, dict(d.states), cerr,
                     getattr(d, "reasons", None)))
    seg = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or (rows[i][1], rows[i][2]) != (rows[seg][1], rows[seg][2]):
            e = [r[3] for r in rows[seg:i] if r[3] is not None]
            r0 = rows[seg]
            st = {k: v[:3] for k, v in r0[4].items()}
            print(f"[{r0[0]:6.1f},{rows[i - 1][0]:6.1f}] {str(r0[1]):9s} {r0[2]:13s} "
                  f"err mean {np.mean(e) if e else float('nan'):6.2f} max {max(e) if e else float('nan'):6.2f} "
                  f"states {st} cand_err {r0[5]}")
            if a.reasons and r0[6]:
                print("      ", str(r0[6])[:400])
            seg = i


if __name__ == "__main__":
    main()
