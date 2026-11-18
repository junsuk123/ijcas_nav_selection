"""Navigation-solution fusion experiments (called by matlab/ijcas_*.m or directly).

    python3 tools/fusion_run.py train  --set fusion_dev --variants rgat,gru --seeds 0,1,2
    python3 tools/fusion_run.py eval   --set fusion_dev --split val
    python3 tools/fusion_run.py export --set fusion_dev --run S10_003 --variant rgat --seed 0 --out f.mat

Datasets: $IJCAS_DATA/fusion/dataset/<set>/*.npz (tools/fusion_build_dataset.py).
Results:  results/fusion/<set>/{runs/<variant>/seed<k>/, val_summary.{json,md}}.
Split of fusion_dev (pre-registered, docs/FUSION_DESIGN_KO.md 5): validation seeds 1, 3, 6, 8;
training seeds 2, 4, 5, 7, 9, 10.  Ground truth enters only the loss and the metrics.

Variants (same inputs, eligibility, covariances and inflation unless stated):
  rgat           proposed: ontology-constrained recurrent R-GAT, scalar weights + learned inflation (B)
  rgat_block     weights per solution and block (position, velocity, attitude)
  rgat_noonto    no ontology constraints (definitional / fusion edges gated like the others)
  rgat_nomem     node memory reset every tick
  rgat_dense     all candidate edges always on (no gates)
  rgcn           gated R-GCN aggregation
  gru            flat GRU on the concatenated node inputs
  rgat_A         reported covariances without inflation (choice A, learned weights only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.fusion import evaluate as E  # noqa: E402
from ijcas.fusion import graph as FG  # noqa: E402
from ijcas.fusion.train import FitCfg, fit, infer_weights, stack_runs  # noqa: E402
from ijcas.fusion.weights import FusionGru, FusionNormalizer, FusionRelGNN, feature_dims  # noqa: E402

DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
SPLITS = {"fusion_dev": {"train": (2, 4, 5, 7, 9, 10), "val": (1, 3, 6, 8)}}
VARIANTS = ("rgat", "rgat_block", "rgat_noonto", "rgat_nomem", "rgat_dense", "rgcn", "gru", "rgat_A")
BASELINES = ("fixed_gnss_ins", "fixed_vio", "fixed_lio", "equal_mean", "equal_ci", "ci_min_trace",
             "ci_trace_normalized", "independent", "oracle_single")
PHASES = (FitCfg(epochs=30, gate_mode="soft", lam_E=0.0), FitCfg(epochs=10, gate_mode="hard_st", lam_E=0.1))


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def runs_of(set_name: str, split: str) -> list:
    seeds = SPLITS[set_name][split]
    ov = os.environ.get(f"IJCAS_FUSION_{split.upper()}_SEEDS")       # pipeline smoke tests only
    if ov:
        seeds = tuple(int(x) for x in ov.split(","))
    paths = sorted((DATA / "fusion" / "dataset" / set_name).glob("*.npz"))
    return [p for p in paths if int(p.stem.split("_")[1]) in seeds]


def build_model(variant: str):
    dims = feature_dims()
    if variant == "gru":
        return FusionGru(dims, hidden=96)
    kw = {"head": "block" if variant == "rgat_block" else "scalar",
          "ontology_constraints": variant != "rgat_noonto",
          "memory": variant != "rgat_nomem", "gated": variant != "rgat_dense",
          "agg": "gcn" if variant == "rgcn" else "gat"}
    return FusionRelGNN(dims, FG.edges(kw["ontology_constraints"]), **kw)


def res_dir(set_name):
    return ROOT / "results" / "fusion" / set_name


def cmd_train(a):
    tr = [load(p) for p in runs_of(a.set, "train")]
    assert tr, "no training runs"
    norm = FusionNormalizer().fit(tr)
    data = stack_runs(tr, norm)
    for v in a.variants.split(","):
        for seed in (int(s) for s in a.seeds.split(",")):
            out = res_dir(a.set) / "runs" / v / f"seed{seed}"
            if (out / "model.pt").exists() and not a.force:
                continue
            out.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            model = build_model(v)
            if v == "rgat_A":
                model.inflation = False
            logf = (out / "train.log").open("w")
            t0 = time.time()
            hist = []
            for i, ph in enumerate(PHASES):
                cfg = FitCfg(**{**ph.__dict__, "epochs": a.epochs or ph.epochs, "batch": a.batch})
                if v in ("gru", "rgat_dense") and cfg.gate_mode != "soft":
                    cfg.gate_mode = "soft"
                hist += fit(model, data, cfg, device=a.device, seed=seed, phase_name=f"P{i}",
                            log=lambda s: (logf.write(s + "\n"), logf.flush()))
            torch.save({"state": model.state_dict(), "variant": v}, out / "model.pt")
            (out / "normalizer.json").write_text(json.dumps(norm.stats))
            (out / "report.json").write_text(json.dumps({"variant": v, "seed": seed, "train_s": time.time() - t0,
                                                         "train_runs": [r["meta"]["id"] for r in tr],
                                                         "history": hist}, indent=1))
            print(f"{v} seed {seed}: trained {time.time() - t0:.0f} s", flush=True)


def load_model(d: Path, device="cpu"):
    ck = torch.load(d / "model.pt", map_location=device, weights_only=False)
    m = build_model(ck["variant"])
    m.load_state_dict(ck["state"])
    if ck["variant"] == "rgat_A":
        m.inflation = False
    return m, FusionNormalizer(json.loads((d / "normalizer.json").read_text()))


def learned_outputs(d: Path, runs: list, device="cpu"):
    m, norm = load_model(d, device)
    data = stack_runs(runs, norm)
    W, U = infer_weights(m, data, device)
    outs = []
    for b, r in enumerate(runs):
        n = len(r["t"])
        u = None if not getattr(m, "inflation", True) else U[b, :n]
        outs.append(E.fuse_run(r, W[b, :n], "ci", u))
    return outs


def cmd_eval(a):
    runs = [load(p) for p in runs_of(a.set, a.split)]
    assert runs, "no runs"
    res = {"baselines": {}, "learned": {}}
    for meth in BASELINES:
        per = []
        for r in runs:
            kind = meth if meth in ("equal_mean", "independent", "ci_trace_normalized") else "ci"
            per.append({**E.metrics(r, E.fuse_run(r, E.weights_for(r, meth), kind)),
                        "run": r["meta"]["id"], "scenario": r["meta"]["scenario"]})
        res["baselines"][meth] = {"pooled": E.pool(per), "per_run": per}
    for vdir in sorted((res_dir(a.set) / "runs").glob("*")):
        seeds = []
        for d in sorted(vdir.glob("seed*")):
            if not (d / "model.pt").exists():
                continue
            outs = learned_outputs(d, runs, a.device)
            per = [{**E.metrics(r, o), "run": r["meta"]["id"], "scenario": r["meta"]["scenario"]}
                   for r, o in zip(runs, outs)]
            seeds.append({"seed": d.name, "pooled": E.pool(per), "per_run": per})
        if seeds:
            res["learned"][vdir.name] = seeds
    out = res_dir(a.set)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.split}_summary.json").write_text(json.dumps(res, indent=1, default=float))
    keys = ("pos_rmse_m", "vel_rmse_mps", "att_rmse_deg", "pos_p95_m", "rte10_m", "availability",
            "nees_pos_mean", "nees_pos_over95", "enc_mean")
    L = [f"# Fusion {a.split} ({a.set}, {len(runs)} runs)", "",
         "| method | " + " | ".join(keys) + " | contrib (gi, vio, lio) |", "|---" * (len(keys) + 2) + "|"]
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for name, P in [(k, v["pooled"]) for k, v in res["baselines"].items()] + \
            [(k, {kk: (float(np.mean([s["pooled"][kk] for s in v if s["pooled"].get(kk) is not None]))
                       if any(s["pooled"].get(kk) is not None for s in v) else None) for kk in keys} |
              {"contrib_mean": np.mean([s["pooled"]["contrib_mean"] for s in v], 0).round(3).tolist()})
             for k, v in res["learned"].items()]:
        L.append(f"| {name} | " + " | ".join(fmt(P.get(k)) for k in keys) + f" | {P.get('contrib_mean')} |")
    (out / f"{a.split}_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_export(a):
    """Per-tick arrays of one run for MATLAB figures (scipy.io.savemat, R2025a reads v5 MAT)."""
    from scipy.io import savemat
    path = DATA / "fusion" / "dataset" / a.set / f"{a.run}.npz"
    r = load(path)
    d = res_dir(a.set) / "runs" / a.variant / f"seed{a.seed}"
    if (d / "model.pt").exists():
        o = learned_outputs(d, [r], a.device)[0]
        label = f"{a.variant} seed{a.seed}"
    else:                                                    # no model yet: trace-normalised CI
        o = E.fuse_run(r, E.weights_for(r, "ci_trace_normalized"), "ci_trace_normalized")
        label = "ci_trace_normalized"
    faults = np.array([[f["start_s"], f["end_s"]] for f in r["meta"]["faults"]]).reshape(-1, 2)
    w = o["w"][..., 0] if o["w"].ndim == 3 else o["w"]
    savemat(a.out, {"t": r["t"], "gt_pos": r["gt_pos"], "pos3": r["pos3"], "valid3": r["valid3"].astype(float),
                    "fused_pos": o["pos"], "fused_vel": o["vel"], "gt_vel": r["gt_vel"],
                    "err_fused": np.linalg.norm(o["pos"] - r["gt_pos"], axis=1), "err3": r["err3_pos"],
                    "weights": w, "contrib": o["contrib"], "faults": faults,
                    "log10_gamma": np.nan_to_num(o["log10_gamma"], nan=np.nan) if o["log10_gamma"] is not None
                    else np.zeros((len(r["t"]), 3, 3)),
                    "label": label, "run": a.run, "scenario": r["meta"]["scenario"]}, do_compression=True)
    print(a.out)


def cmd_graph(a):
    """Fusion graph (nodes, kinds, edges, protected flag) and ontology facts as JSON (MATLAB Figure 5)."""
    from ijcas.fusion import ontology as O
    E_ = FG.edges(True)
    out = {"nodes": list(FG.NODES), "kinds": [FG.NODE_KIND[n] for n in FG.NODES],
           "edges": [{"src": e.src, "dst": e.dst, "rel": e.rel, "protected": e.protected} for e in E_],
           "shared_sources": {f"{k[0]}|{k[1]}": v for k, v in O.shared_sources().items()},
           "required_method": O.required_method(), "requires_alignment": O.requires_alignment()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval", "export", "graph"))
    ap.add_argument("--set", default="fusion_dev")
    ap.add_argument("--split", default="val")
    ap.add_argument("--variants", default="rgat,gru")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run", default="")
    ap.add_argument("--variant", default="rgat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fusion_run.mat")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "export": cmd_export, "graph": cmd_graph}[a.cmd](a)


if __name__ == "__main__":
    main()
