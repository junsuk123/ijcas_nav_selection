"""Graph-selector experiments: train on validation -> val summary -> freeze -> test (once).

    python3 tools/rgat_run.py train  [--variants ...] [--seeds 0,1,2]
    python3 tools/rgat_run.py val-summary
    python3 tools/rgat_run.py freeze
    python3 tools/rgat_run.py test   [--workers 8]     # refuses unless FROZEN.json hashes match

Datasets: $IJCAS_DATA/rgat/dataset/{validation,test}/*.npz (tools/rgat_build_dataset.py).
Results:  results/rgat/{runs/<variant>/seed<k>/, val_summary.{json,md}, FROZEN.json, test/}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.rgat import experiment as X  # noqa: E402
from ijcas.rgat.graph import CANDIDATES  # noqa: E402
from ijcas.rgat.learn import (infer, load_run, metrics_for_selection, metrics_rule, oracle_sel,  # noqa: E402
                              pool, select, stack)

DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
DS = DATA / "rgat" / "dataset"
RES = ROOT / "results" / "rgat"
ORDER = ["rgat_dyn", "rgcn_dyn", "rgat_dense", "rgat_static", "rgat_nomem", "gru_flat",
         "cr_plain", "cr_e0", "cr_gstar", "cr_gstar_kd"]          # cr_*: retrieval selector (design 12)
CODE = ["ijcas/rgat/graph.py", "ijcas/rgat/features.py", "ijcas/rgat/dataset.py",
        "ijcas/rgat/model.py", "ijcas/rgat/learn.py", "ijcas/rgat/experiment.py", "ijcas/rgat/retrieval.py",
        "config/selector/thresholds_validated.yaml", "ijcas/knowledge/ijcas_nav.ttl",
        "ijcas/selector/engine.py", "ijcas/handover.py", "tools/rgat_run.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a")

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


# ------------------------------------------------------------------------------- train
def cmd_train(a):
    paths = sorted(str(p) for p in (DS / "validation").glob("*.npz"))
    assert paths, "no validation dataset"
    variants = a.variants.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    for seed in seeds:
        for v in [x for x in ORDER if x in variants]:
            out = RES / "runs" / v / f"seed{seed}"
            if (out / "report.json").exists() and not a.force:
                continue
            log = logger(out / "train.log")
            needs = v in ("rgat_static", "rgat_nomem", "cr_gstar", "cr_gstar_kd")
            struct = RES / "runs" / "rgat_dyn" / f"seed{seed}" if needs else None
            if struct is not None and not (struct / "report.json").exists():
                log(f"skip {v}: needs rgat_dyn seed{seed} first")
                continue
            if v.startswith("cr_"):
                from ijcas.rgat.retrieval import run_cr_variant
                run_cr_variant(v, paths, out, seed=seed, device=a.device, structure_from=struct,
                               teacher_from=struct if v == "cr_gstar_kd" else None, log=log)
                continue
            X.run_variant(v, paths, out, seed=seed, device=a.device, structure_from=struct, log=log)


# ------------------------------------------------------------------------------- val summary
def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def cmd_val_summary(a):
    rows = {}
    for vdir in sorted((RES / "runs").glob("*")):
        reps = [json.loads((d / "report.json").read_text()) for d in sorted(vdir.glob("seed*"))
                if (d / "report.json").exists()]
        if not reps:
            continue
        r = {"seeds": len(reps)}
        for k in ("pos_rmse_m", "vel_rmse_mps", "att_rmse_deg", "availability",
                  "switches_per_run", "false_normal_s_per_run", "response_mean_s", "recovery_mean_s"):
            r[k] = _mean_std([x["val"][k] for x in reps])
        for reg in ("normal", "single", "compound", "recovery"):
            r[f"rmse_{reg}"] = _mean_std([x["val"]["regimes"].get(reg, {}).get("pos_rmse_m") for x in reps])
        r["train_min"] = _mean_std([x["train_s"] / 60 for x in reps])
        if "val_hold_rate" in reps[0]:
            r["hold_rate"] = _mean_std([x["val_hold_rate"] for x in reps])
            r["dims"] = [[g["D"] for g in x["groups"]] for x in reps]
        r["params"] = reps[0]["params"]
        r["select"] = [x["select"] for x in reps]
        if "prune" in reps[0]:
            r["kept_gated"] = [len(json.loads((vdir / f"seed{x['seed']}" / "structure.json").read_text())["kept_gated"])
                               for x in reps]
        rows[vdir.name] = r
    # rule-based / fixed / oracle on the same val runs
    _, va_p = X.split(sorted(str(p) for p in (DS / "validation").glob("*.npz")))
    va = [load_run(p) for p in va_p]
    base = baselines(va)
    out = {"learned": rows, "baselines": base}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "val_summary.json").write_text(json.dumps(out, indent=1))
    (RES / "val_summary.md").write_text(render(out, "val (validation seeds 8-10)"))
    print((RES / "val_summary.md").read_text())


def baselines(runs):
    out = {}
    for k, c in enumerate(CANDIDATES):
        out[f"F-{c}"] = pool([metrics_for_selection(r, np.where(r["valid"][:, k], k, -1)) for r in runs])
    out["Ontology-K"] = pool([metrics_rule(r, "ont") for r in runs])
    out["H-FSM"] = pool([metrics_rule(r, "hfsm") for r in runs])
    out["Oracle"] = pool([metrics_for_selection(r, oracle_sel(r)) for r in runs])
    return out


def _fmt(x, nd=3):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return "-" if x[0] is None else f"{x[0]:.{nd}f} ± {x[1]:.{nd}f}"
    return "-" if x is None else f"{x:.{nd}f}"


def render(res, title):
    L = [f"# Graph selector results: {title}", ""]
    L += ["| method | pos RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | availability | switches/run | "
          "false-normal s/run | response s | recovery s |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["learned"].items():
        L.append(f"| {n} (n={r['seeds']}) | {_fmt(r['pos_rmse_m'])} | {_fmt(r['vel_rmse_mps'])} | "
                 f"{_fmt(r['att_rmse_deg'], 2)} | {_fmt(r['availability'])} | {_fmt(r['switches_per_run'], 2)} | "
                 f"{_fmt(r['false_normal_s_per_run'], 2)} | {_fmt(r['response_mean_s'], 2)} | "
                 f"{_fmt(r['recovery_mean_s'], 2)} |")
    for n, r in res["baselines"].items():
        L.append(f"| {n} | {_fmt(r['pos_rmse_m'])} | {_fmt(r['vel_rmse_mps'])} | {_fmt(r['att_rmse_deg'], 2)} | "
                 f"{_fmt(r['availability'])} | {_fmt(r['switches_per_run'], 2)} | "
                 f"{_fmt(r['false_normal_s_per_run'], 2)} | {_fmt(r['response_mean_s'], 2)} | "
                 f"{_fmt(r['recovery_mean_s'], 2)} |")
    L += ["", "| method | normal | single | compound | recovery |", "|---|---|---|---|---|"]
    for n, r in res["learned"].items():
        L.append(f"| {n} | " + " | ".join(_fmt(r[f"rmse_{g}"]) for g in ("normal", "single", "compound", "recovery")) + " |")
    for n, r in res["baselines"].items():
        L.append(f"| {n} | " + " | ".join(_fmt(r["regimes"].get(g, {}).get("pos_rmse_m"))
                                           for g in ("normal", "single", "compound", "recovery")) + " |")
    cr = {n: r for n, r in res["learned"].items() if "hold_rate" in r}
    if cr:
        L += ["", "| retrieval | hold rate | stored floats | case-space dims per seed |", "|---|---|---|---|"]
        for n, r in cr.items():
            L.append(f"| {n} | {_fmt(r['hold_rate'])} | {r['params']} | {r['dims']} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------- freeze / test
def cmd_freeze(a):
    files = {c: sha(ROOT / c) for c in CODE}
    models = {}
    for d in sorted((RES / "runs").glob("*/seed*")):
        for f in ("model.pt", "model.npz", "normalizer.json", "report.json", "structure.json"):
            if (d / f).exists():
                models[str((d / f).relative_to(RES))] = sha(d / f)
    frozen = {"frozen_at": time.strftime("%F %T"), "code": files, "models": models,
              "test_dataset_built_before_freeze": (DS / "test").exists()}
    (RES / "FROZEN.json").write_text(json.dumps(frozen, indent=1))
    print(f"frozen {len(models)} model files, {len(files)} code/config files -> {RES / 'FROZEN.json'}")


def check_frozen():
    fz = json.loads((RES / "FROZEN.json").read_text())
    bad = [c for c, h in fz["code"].items() if sha(ROOT / c) != h]
    bad += [m for m, h in fz["models"].items() if sha(RES / m) != h]
    if bad:
        raise SystemExit(f"FROZEN mismatch, refusing to evaluate test: {bad}")
    return fz


def cmd_test(a):
    check_frozen()
    paths = sorted(str(p) for p in (DS / "test").glob("*.npz"))
    assert paths, "no test dataset"
    runs = [load_run(p) for p in paths]
    out_dir = RES / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_method, learned = {}, {}
    for vdir in sorted((RES / "runs").glob("*")):
        seeds = []
        for d in sorted(vdir.glob("seed*")):
            if (d / "model.npz").exists():                      # retrieval selector (design 12)
                from ijcas.rgat.retrieval import load_cr, predict
                cb, norm, cfg, rep = load_cr(d, a.device)
                pr = predict(cb, runs, norm)
                pis = [p for p, _, _ in pr]
                holds = [h for _, h, _ in pr]
            else:
                model, norm, cfg, rep = X.load_model(d, a.device)
                Xt, Vt, _ = stack(runs, norm)
                pa = infer(model, Xt, Vt, a.device)["pi"]
                pis = [pa[b, :len(r["t"])] for b, r in enumerate(runs)]
                holds = [None] * len(runs)
            per = []
            for b, r in enumerate(runs):
                sel, ev = select(pis[b], r["valid"], r["t"], cfg, hold=holds[b])
                m = metrics_for_selection(r, sel)
                m.update(run=r["meta"]["id"], scenario=r["meta"]["scenario"], scene=r["meta"]["scene"],
                         events=ev[:50], hold_rate=None if holds[b] is None else float(np.mean(holds[b])))
                per.append(m)
            seeds.append({"seed": d.name, "pooled": pool(per), "per_run": per, "params": rep.get("params"),
                          "dims": [g["D"] for g in rep["groups"]] if "groups" in rep else None})
        learned[vdir.name] = seeds
    base = {}
    for name, f in [(f"F-{c}", (lambda k: lambda r: metrics_for_selection(r, np.where(r["valid"][:, k], k, -1)))(k))
                    for k, c in enumerate(CANDIDATES)] + \
            [("Ontology-K", lambda r: metrics_rule(r, "ont")), ("H-FSM", lambda r: metrics_rule(r, "hfsm")),
             ("Oracle", lambda r: metrics_for_selection(r, oracle_sel(r)))]:
        per = []
        for r in runs:
            m = f(r)
            m.update(run=r["meta"]["id"], scenario=r["meta"]["scenario"], scene=r["meta"]["scene"])
            per.append(m)
        base[name] = {"pooled": pool(per), "per_run": per}
    (out_dir / "results.json").write_text(json.dumps({"learned": learned, "baselines": base}, default=str))
    summary = {"learned": {}, "baselines": {k: v["pooled"] for k, v in base.items()}}
    for v, seeds in learned.items():
        P = [s["pooled"] for s in seeds]
        r = {"seeds": len(P)}
        for k in ("pos_rmse_m", "vel_rmse_mps", "att_rmse_deg", "availability", "switches_per_run",
                  "false_normal_s_per_run", "response_mean_s", "recovery_mean_s"):
            r[k] = _mean_std([p[k] for p in P])
        for reg in ("normal", "single", "compound", "recovery"):
            r[f"rmse_{reg}"] = _mean_std([p["regimes"].get(reg, {}).get("pos_rmse_m") for p in P])
        if seeds[0]["per_run"][0].get("hold_rate") is not None:
            r["hold_rate"] = _mean_std([float(np.mean([m["hold_rate"] for m in s_["per_run"]])) for s_ in seeds])
            r["params"], r["dims"] = seeds[0]["params"], [s_["dims"] for s_ in seeds]
        summary["learned"][v] = r
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "summary.md").write_text(render(summary, "test (seeds 101-130)"))
    print((out_dir / "summary.md").read_text())


def cmd_latency(a):
    """Selector-only operation cost on one validation run (CPU, 1 thread, B = 1)."""
    _, va_p = X.split(sorted(str(p) for p in (DS / "validation").glob("*.npz")))
    r = load_run(va_p[0])
    out = {}
    for d in sorted((RES / "runs").glob("*/seed0")):
        if (d / "model.npz").exists():
            from ijcas.rgat.retrieval import latency as cr_latency, load_cr
            cb, norm, cfg, rep = load_cr(d, "cpu")
            out[d.parent.name] = cr_latency(cb, norm, r)
            continue
        model, norm, cfg, rep = X.load_model(d, "cpu")
        Xv, Vv, _ = stack([r], norm)
        res = X.latency(model, Xv, Vv, sparse=True)
        if getattr(model, "gated", False):
            res["dense_masked"] = X.latency(model, Xv, Vv, sparse=False)
        res["params"] = rep["params"]
        if (d / "structure.json").exists():
            s = json.loads((d / "structure.json").read_text())
            res["permanent_pruning_ratio"] = 1 - len(s["kept_gated"]) / s["n_gated_full"]
        out[d.parent.name] = res
    (RES / "latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "val-summary", "freeze", "test", "latency"))
    ap.add_argument("--variants", default=",".join(ORDER))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"train": cmd_train, "val-summary": cmd_val_summary, "freeze": cmd_freeze, "test": cmd_test,
     "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
