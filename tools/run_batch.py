"""Resumable batch of IJCAS runs: python3 tools/run_batch.py --set validation [--only S01,S02]

Each run: resolve (scenario, seed) -> config/generated/<id>_<seed>.yaml -> scripts/run_stack.sh.
A run is done when <out>/DONE exists; failures are recorded in <out>/FAILED with the reason and
kept (never silently dropped).  Progress: $IJCAS_DATA/runs/batch_<set>.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.scenarios import (ALL, OUTDOOR, OUTDOOR_DEV_SEEDS, OUTDOOR_TEST_SEEDS, TEST2_SEEDS,  # noqa: E402
                             TEST_SEEDS, VALIDATION_SEEDS, resolve)

DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("validation", "test", "test2", "fusion_dev", "outdoor_dev",
                                     "outdoor_test", "outdoor_pilot"), required=True)
    ap.add_argument("--seeds", default="", help="override the seeds of the set (pilots)")
    ap.add_argument("--only", default="")
    ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args()
    # fusion_dev (2026-11-18): seeds 1-10 re-recorded with the full estimator covariances (ros2/recorder.py)
    seeds = {"validation": VALIDATION_SEEDS, "test": TEST_SEEDS, "test2": TEST2_SEEDS,
             "fusion_dev": VALIDATION_SEEDS, "outdoor_dev": OUTDOOR_DEV_SEEDS,
             "outdoor_test": OUTDOOR_TEST_SEEDS, "outdoor_pilot": [901]}[a.set]
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    # outdoor sets (2026-11-19): urban / solar farm / tunnel road x 3, 6, 9 m/s (ijcas/scenarios.py)
    sids = OUTDOOR if a.set.startswith("outdoor") else \
        [s for s in ALL if not (a.set in ("validation", "fusion_dev") and s == "S09")]
    if a.only:
        sids = [s for s in sids if s in a.only.split(",")]
    gen = ROOT / "config/generated"
    gen.mkdir(parents=True, exist_ok=True)
    log = DATA / "runs" / f"batch_{a.set}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(sid, seed) for seed in seeds for sid in sids]
    for i, (sid, seed) in enumerate(jobs):
        out = DATA / "runs" / a.set / f"{sid}_{seed:03d}"
        if (out / "DONE").exists():
            continue
        cfg = resolve(sid, seed)
        cpath = gen / f"{sid}_{seed:03d}.yaml"
        cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for attempt in range(a.retries + 1):
            if out.exists():
                subprocess.run(["rm", "-rf", str(out)])
            t0 = time.time()
            r = subprocess.run(["bash", str(ROOT / "scripts/run_stack.sh"), str(cpath), cfg["scene"],
                                str(seed), str(out)], capture_output=True, text=True, timeout=1800)
            man = out / "manifest.json"
            ok = man.exists() and json.loads(man.read_text()).get("error") is None and \
                (out / "record.npz").exists()
            with log.open("a") as f:
                f.write(f"{time.strftime('%F %T')} [{i + 1}/{len(jobs)}] {sid} seed {seed} "
                        f"scene {cfg['scene']} attempt {attempt} ok={ok} wall={time.time() - t0:.0f}s "
                        f"{r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else ''}\n")
            if ok:
                (out / "DONE").write_text(json.dumps(cfg))
                break
            (out / "FAILED").write_text(f"rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


if __name__ == "__main__":
    main()
