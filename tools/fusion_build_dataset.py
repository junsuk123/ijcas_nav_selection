"""Build fusion tensors for every finished run of a set (resumable, parallel).

    python3 tools/fusion_build_dataset.py --set fusion_dev [--runs-root DIR] [--workers 4]

Writes $IJCAS_DATA/fusion/dataset/<set>/<run>.npz (ijcas/fusion/dataset.py).  Nothing about
performance is computed here.
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
THRESHOLDS = ROOT / "config/selector/thresholds_validated.yaml"


def _one(args):
    run, out = args
    from ijcas.fusion.dataset import build_fusion_run
    from ijcas.selector.common import load_config
    try:
        t0 = time.time()
        arr = build_fusion_run(run, load_config(THRESHOLDS))
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arr)
        tmp.rename(out)
        return f"{run.name} ok {time.time() - t0:.0f}s"
    except Exception as e:  # recorded, never silently dropped
        out.with_suffix(".FAILED").write_text(repr(e))
        return f"{run.name} FAILED {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--runs-root", default=str(DATA / "runs"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src = Path(a.runs_root) / a.set
    dst = DATA / "fusion" / "dataset" / a.set
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(r, dst / f"{r.name}.npz") for r in sorted(src.iterdir())
            if (r / "DONE").exists() and not (dst / f"{r.name}.npz").exists()]
    print(f"{len(jobs)} runs to build -> {dst}", flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
