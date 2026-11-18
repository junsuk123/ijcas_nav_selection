"""v2 one-time test on test2 (seeds 201-230), pre-registered in docs/RGAT_DESIGN_KO.md 13.4.

    python3 tools/rgat_test2.py build-v1      # test2 tensors with the v1 code (commit 878eac1 worktree)
    python3 tools/rgat_test2.py run           # once: refuses unless the v2 freeze and v1 models match

Combinations: (a) v1 frozen models + v1 policy dataset, (b) v1 models + v2 policy dataset,
(c) v2 models + v2 policy dataset; comparators (F-*, Ontology-K, H-FSM, Oracle) on both datasets.
Q1 = rgat_dyn (b) - (a), Q2 = (c) - (b), Q3 = (c) - (a): paired run bootstrap (4000, seed mean, 95 %),
on all runs, S10 runs and S10 saturation runs; RMSE (output ticks), availability, no-valid-nav time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["IJCAS_RGAT_VERSION"] = "v2"
import rgat_run as RR  # noqa: E402
from ijcas.rgat.learn import load_run  # noqa: E402

DATA = RR.DATA
V1_COMMIT = "878eac1"
RES1, RES2 = ROOT / "results" / "rgat", ROOT / "results" / "rgat_v2"
DS1 = DATA / "rgat" / "dataset_v1" / "test2"
DS2 = DATA / "rgat" / "dataset_v2" / "test2"
OUT = RES2 / "test2"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check():
    f2 = json.loads((RES2 / "FROZEN.json").read_text())
    bad = [c for c, h in f2["code"].items() if sha(ROOT / c) != h]
    bad += [m for m, h in f2["models"].items() if sha(RES2 / m) != h]
    f1 = json.loads((RES1 / "FROZEN.json").read_text())
    bad += [f"v1:{m}" for m, h in f1["models"].items() if sha(RES1 / m) != h]  # v1 code changed on purpose
    if bad:
        raise SystemExit(f"FROZEN mismatch, refusing test2: {bad}")


def cmd_build_v1(a):
    """Build test2 tensors with the frozen v1 code in a detached worktree (nothing is evaluated)."""
    wt = Path(a.worktree)
    if not wt.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(wt), V1_COMMIT], check=True)
    code = (
        "import sys, numpy as np\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(wt)!r})\n"
        "from ijcas.rgat.dataset import build_run\n"
        "from ijcas.selector.common import load_config\n"
        f"cfg = load_config({str(wt / 'config/selector/thresholds_validated.yaml')!r})\n"
        "run, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "arr = build_run(run, cfg)\n"
        "tmp = out.with_name(out.stem + '.tmp.npz'); np.savez_compressed(tmp, **arr); tmp.rename(out)\n")
    DS1.mkdir(parents=True, exist_ok=True)
    runs = sorted(r for r in (DATA / "runs" / "test2").iterdir() if (r / "DONE").exists())
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        out = DS1 / f"{r.name}.npz"
        if out.exists():
            return f"{r.name} exists"
        p = subprocess.run([sys.executable, "-c", code, str(r), str(out)], capture_output=True, text=True,
                           cwd=str(wt))
        return f"{r.name} {'ok' if p.returncode == 0 else 'FAILED ' + p.stderr[-300:]}"
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(one, runs):
            print(msg, flush=True)


def _rmse(per_seed_runs, idx):
    """Seed-mean of pooled RMSE over the selected runs; per_seed_runs: [seed][run] metric dicts."""
    vals = []
    for runs in per_seed_runs:
        ss = sum(runs[i]["sum_sq_p"] for i in idx)
        n = sum(runs[i]["n_out"] for i in idx)
        vals.append(np.sqrt(ss / n) if n else np.nan)
    return float(np.mean(vals))


def _avail(per_seed_runs, idx):
    return float(np.mean([np.average([runs[i]["availability"] for i in idx], weights=[runs[i]["ticks"] for i in idx])
                          for runs in per_seed_runs]))


def _nvn(per_seed_runs, idx):
    return float(np.mean([np.mean([runs[i]["no_valid_nav_s"] for i in idx]) for runs in per_seed_runs]))


def bootstrap(x, y, idx, fn, n=4000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = np.asarray(idx)
    d0 = fn(x, idx) - fn(y, idx)
    ds = [fn(x, s) - fn(y, s) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"diff": d0, "ci95": [float(lo), float(hi)], "x": fn(x, idx), "y": fn(y, idx)}


def cmd_run(a):
    check()
    p1 = sorted(DS1.glob("*.npz"))
    p2 = sorted(DS2.glob("*.npz"))
    assert p1 and [p.name for p in p1] == [p.name for p in p2], "v1/v2 test2 datasets differ in runs"
    OUT.mkdir(parents=True, exist_ok=True)
    runs1 = [load_run(p) for p in p1]
    runs2 = [load_run(p) for p in p2]
    res = {"a_v1model_v1policy": RR.eval_learned(RES1, runs1, a.device),
           "b_v1model_v2policy": RR.eval_learned(RES1, runs2, a.device),
           "c_v2model_v2policy": RR.eval_learned(RES2, runs2, a.device),
           "baselines_v1policy": RR.eval_baselines(runs1),
           "baselines_v2policy": RR.eval_baselines(runs2)}
    for k in ("a_v1model_v1policy", "b_v1model_v2policy", "c_v2model_v2policy"):
        (OUT / k).mkdir(exist_ok=True)
        base = res["baselines_v1policy" if k.startswith("a") else "baselines_v2policy"]
        RR._write_test(OUT / k, res[k], base, f"test2 (seeds 201-230) {k}")
    meta = [r["meta"] for r in runs2]
    subsets = {"all": list(range(len(meta))),
               "S10": [i for i, m in enumerate(meta) if m["scenario"] == "S10"],
               "S10_saturation": [i for i, m in enumerate(meta) if m["scenario"] == "S10"
                                  and m["variant"].get("imu") == "saturation"]}
    per = {k: [s["per_run"] for s in res[k]["rgat_dyn"]] for k in
           ("a_v1model_v1policy", "b_v1model_v2policy", "c_v2model_v2policy")}
    q = {"Q1_policy (b-a)": ("b_v1model_v2policy", "a_v1model_v1policy"),
         "Q2_retraining (c-b)": ("c_v2model_v2policy", "b_v1model_v2policy"),
         "Q3_total (c-a)": ("c_v2model_v2policy", "a_v1model_v1policy")}
    for name in ("H-FSM", "Ontology-K"):
        per[f"{name} v2"] = [res["baselines_v2policy"][name]["per_run"]]
        q[f"c - {name} (v2 policy)"] = ("c_v2model_v2policy", f"{name} v2")
    out = {}
    for qn, (x, y) in q.items():
        out[qn] = {sub: {"rmse": bootstrap(per[x], per[y], idx, _rmse),
                         "availability": bootstrap(per[x], per[y], idx, _avail),
                         "no_valid_nav_s": bootstrap(per[x], per[y], idx, _nvn)}
                   for sub, idx in subsets.items() if idx}
    (OUT / "questions.json").write_text(json.dumps(out, indent=1))
    lines = ["# test2 pre-registered questions (rgat_dyn, paired run bootstrap 4000, seed mean)", "",
             "| question | subset | RMSE diff [m] (95 % CI) | availability diff | no-valid-nav s/run diff |",
             "|---|---|---|---|---|"]
    for qn, subs in out.items():
        for sub, d in subs.items():
            f = lambda r, nd=3: f"{r['diff']:+.{nd}f} [{r['ci95'][0]:+.{nd}f}, {r['ci95'][1]:+.{nd}f}]"  # noqa
            lines.append(f"| {qn} | {sub} | {f(d['rmse'])} | {f(d['availability'])} | {f(d['no_valid_nav_s'], 1)} |")
    (OUT / "questions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build-v1", "run"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worktree", default=str(DATA / "rgat" / "v1_worktree"))
    a = ap.parse_args()
    {"build-v1": cmd_build_v1, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
