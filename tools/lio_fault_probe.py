"""LIO behaviour around injected LiDAR faults (used to set the S04 levels for FAST-LIO2).

    python3 tools/lio_fault_probe.py RUN [RUN ...]

The engine's track is 4-DOF aligned to ground truth on its first 10 s of output (then frozen, as in
ijcas.evaluation 'initial'), and summarised before / during / after the LiDAR fault window:
max position error, error at the end of the window and 30 s later, the longest output gap, and the
number of outputs inside the window.  Pre-stated criterion for an S04 level: the engine 'fails' if
its error exceeds the 2 m alarm limit inside the window or within 30 s after it, or its output stops
for more than 1 s (it would then be ineligible).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.evaluation import fit_4dof, gt_at, load_run  # noqa: E402

ALARM = 2.0


def probe(run_dir, engine="fastlio"):
    r = load_run(run_dir)
    tr = r["tracks"][engine]
    faults = [f for f in r["manifest"]["scenario"]["faults"] if f["target"] == "lidar"]
    a, b = (faults[0]["start_s"], faults[0]["end_s"]) if faults else (np.inf, np.inf)
    t = tr.stamp
    gp, _, _ = gt_at(r["gt"], t)
    w = t <= t[0] + 10.0
    rot, trans = fit_4dof(tr.pos[w], gp[w])
    e = np.linalg.norm(rot.apply(tr.pos) + trans - gp, axis=1)

    def seg(lo, hi):
        m = (t >= lo) & (t < hi)
        return float(e[m].max()) if m.any() else None

    def at(x):
        i = np.searchsorted(t, x)
        return float(e[min(i, len(e) - 1)]) if len(e) else None
    inside = (t >= a) & (t < b)
    gaps = np.diff(np.concatenate([[a], t[inside], [b]])) if np.isfinite(a) else np.array([0.0])
    out = {"run": Path(run_dir).name, "fault": faults[0] if faults else None,
           "max_err_before_m": seg(t[0], a), "max_err_during_m": seg(a, b),
           "max_err_30s_after_m": seg(b, b + 30.0), "err_end_m": at(b), "err_end_plus30_m": at(b + 30.0),
           "outputs_in_window": int(inside.sum()), "longest_gap_s": float(gaps.max()),
           "final_err_m": float(e[-1])}
    worst = max(x for x in (out["max_err_during_m"], out["max_err_30s_after_m"]) if x is not None) \
        if out["max_err_during_m"] is not None or out["max_err_30s_after_m"] is not None else 0.0
    out["fails"] = bool(worst > ALARM or out["longest_gap_s"] > 1.0)
    return out


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        print(json.dumps(probe(rd)))
