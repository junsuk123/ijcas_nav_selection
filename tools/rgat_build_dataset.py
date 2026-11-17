"""Build graph-selector tensors for every finished run of a set (resumable, parallel).

    python3 tools/rgat_build_dataset.py --set validation --thresholds config/selector/thresholds_validated.yaml

Writes $IJCAS_DATA/rgat/dataset/<set>/<run>.npz (see ijcas/rgat/dataset.py).  Only runs with a
DONE marker are used.  Nothing is evaluated or printed about selection performance here, so
building the test set does not open its results.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))


def _one(args):
    run, out, thresholds = args
    from ijcas.rgat.dataset import build_run
    from ijcas.selector.common import load_config
    try:
        t0 = time.time()
        arr = build_run(run, load_config(thresholds))
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arr)
        tmp.rename(out)
        return f"{run.name} ok {time.time() - t0:.0f}s"
    except Exception as e:  # recorded, never silently dropped
        (out.with_suffix(".FAILED")).write_text(repr(e))
        return f"{run.name} FAILED {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("validation", "test"))
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs-root", default=str(DATA / "runs"))
    a = ap.parse_args()
    src = Path(a.runs_root) / a.set
    dst = DATA / "rgat" / "dataset" / a.set
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(r, dst / f"{r.name}.npz", a.thresholds) for r in sorted(src.iterdir())
            if (r / "DONE").exists() and not (dst / f"{r.name}.npz").exists()]
    print(f"{len(jobs)} runs to build -> {dst}", flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
