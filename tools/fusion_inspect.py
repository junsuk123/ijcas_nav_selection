"""Integrity of recorded runs for the fusion study, as JSON (read by matlab/ijcas_collect.m).

    python3 tools/fusion_inspect.py --set fusion_dev [--out file.json]

Per run: DONE marker, manifest error, simulated duration, output counts of the three estimators,
whether the full covariance columns exist and the share of outputs that report each block
(position, velocity, attitude) with a positive definite matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
EST = ("px4", "openvins", "fastlio")


def _pd_share(M):
    if len(M) == 0:
        return 0.0
    ok = np.isfinite(M).all((1, 2)) & (np.abs(M).sum((1, 2)) > 0)
    ev = np.full(len(M), -1.0)
    if ok.any():
        ev[ok] = np.linalg.eigvalsh(0.5 * (M[ok] + np.swapaxes(M[ok], 1, 2))).min(1)
    return float(np.mean(ev > 0))


def inspect(run: Path) -> dict:
    out = {"run": run.name, "done": (run / "DONE").exists()}
    try:
        man = json.loads((run / "manifest.json").read_text())
        out.update(error=man.get("error"), sim_s=man.get("sim_s"))
        rec = np.load(run / "record.npz")
        for e in EST:
            a = rec[f"odom_{e}"]
            d = {"outputs": int(len(a)), "full_covariance": bool(a.ndim == 2 and a.shape[1] >= 90)}
            if d["full_covariance"] and len(a):
                pc = a[:, 18:54].reshape(-1, 6, 6)
                tc = a[:, 54:90].reshape(-1, 6, 6)
                pos, att = (pc[:, 3:6, 3:6], pc[:, 0:3, 0:3]) if e == "fastlio" else (pc[:, 0:3, 0:3], pc[:, 3:6, 3:6])
                d.update(pos_reported=_pd_share(pos), vel_reported=_pd_share(tc[:, 0:3, 0:3]),
                         att_reported=_pd_share(att))
            out[e] = d
    except Exception as ex:  # reported, not hidden
        out["inspect_error"] = repr(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [inspect(r) for r in sorted((DATA / "runs" / a.set).iterdir()) if r.is_dir()]
    s = json.dumps({"set": a.set, "runs": rows}, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(s if not a.out else a.out)


if __name__ == "__main__":
    main()
