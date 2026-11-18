"""Experiment pipeline on the validation recordings (design 6, 9) - test data is never touched here.

Split (by run, never by tick): train = validation seeds 1-7, val = validation seeds 8-10.
Variants (identical inputs, policies, selection rule, tuning procedure):
  rgat_dyn    proposed: wide gated R-GAT -> offline structural pruning on val -> fine-tune on E_*
              (soft Hard-Concrete, then straight-through binary gates with hysteresis)
  rgcn_dyn    same pipeline, gated R-GCN aggregation (its own pruning)
  rgat_dense  wide candidate graph, no gates (all candidate edges always on)
  rgat_static E_* of rgat_dyn, no gates (fixed sparse graph)
  rgat_nomem  E_* of rgat_dyn with gates, node memory reset every tick
  gru_flat    non-graph GRU on the concatenated inputs, parameter count matched
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .graph import CANDIDATES, candidate_edges, restrict
from .learn import (EVAL_REGIMES, Normalizer, Phase, SelectCfg, fit, infer, load_run,
                    metrics_for_selection, pool, select, stack)
from .model import FlatGru, RecurrentRelGNN, n_params

TRAIN_SEEDS = tuple(range(1, 8))
VAL_SEEDS = (8, 9, 10)
SELECT_GRID = [SelectCfg(m, d, 1.0) for m in (0.0, 0.05, 0.1, 0.2, 0.3) for d in (0.5, 1.0, 2.0)]
DEFAULT_SELECT = SelectCfg(0.1, 2.0, 1.0)
PRUNE = {"tau_use": 0.05, "tol_rel": 0.02, "tol_abs_m": 0.02, "tol_avail": 0.005}
# one batch = all training runs (per-tick cost is launch-bound, independent of batch size)
SCHEDULE = {"A": 40, "C1": 10, "C2": 10, "lam_E": 0.1, "lam_S": 0.1, "lr": 2e-3, "chunk": 200,
            "batch": None}


def split(paths, train_seeds=None, val_seeds=None):
    """Run-level split.  Defaults are read at call time, so a caller (tools/rgat_run.py, v2) can set
    TRAIN_SEEDS / VAL_SEEDS of this module once for the whole experiment."""
    train_seeds = TRAIN_SEEDS if train_seeds is None else train_seeds
    val_seeds = VAL_SEEDS if val_seeds is None else val_seeds
    tr, va = [], []
    for p in paths:
        seed = int(Path(p).stem.split("_")[1])
        (tr if seed in train_seeds else va if seed in val_seeds else []).append(p)
    return tr, va


# ------------------------------------------------------------------------------- evaluation
def evaluate(model, runs, X, V, cfg: SelectCfg, device, pis=None):
    if pis is None:
        pis = infer(model, X, V, device)["pi"]
    per = []
    for b, r in enumerate(runs):
        n = len(r["t"])
        sel, _ = select(pis[b, :n], r["valid"], r["t"], cfg)
        per.append(metrics_for_selection(r, sel))
    return pool(per), per


def tune_select(model, runs, X, V, device, log=print):
    """Pre-registered rule: minimum pooled position RMSE on val; among settings within 2 % of
    it, the fewest switches; then the larger dwell (more conservative)."""
    pis = infer(model, X, V, device)["pi"]
    rows = []
    for c in SELECT_GRID:
        P, _ = evaluate(model, runs, X, V, c, device, pis)
        rows.append((c, P))
    best = min(P["pos_rmse_m"] for _, P in rows)
    ok = [(c, P) for c, P in rows if P["pos_rmse_m"] <= best * 1.02]
    c, P = min(ok, key=lambda x: (x[1]["switches_per_run"], -x[0].dwell_s))
    log(f"select tuned: margin {c.margin} dwell {c.dwell_s} -> rmse {P['pos_rmse_m']:.3f} "
        f"switches {P['switches_per_run']:.2f}")
    return c, [(vars(c), {k: P[k] for k in ("pos_rmse_m", "availability", "switches_per_run")})
               for c, P in rows]


def edge_usage(model, runs, X, V, device):
    """Activation rate of every gated edge per regime on val (operation-mode binary gates)."""
    out = infer(model, X, V, device)
    z = out["z"]                                                    # [B,T,Eg]
    names = model.gated_names
    rate = {}
    for ri, rname in enumerate(EVAL_REGIMES):
        num, den = np.zeros(len(names)), 0
        for b, r in enumerate(runs):
            m = r["regime"] == ri
            n = len(r["t"])
            num += z[b, :n][m].sum(0)
            den += m.sum()
        if den:
            rate[rname] = (num / den).tolist()
    return names, rate, out


def prune(model, runs, X, V, device, cfg=DEFAULT_SELECT, log=print):
    """Offline structural pruning (design 6.1): only edges rarely used in *every* regime are
    removal candidates; each removal is kept only if no regime loses RMSE / availability beyond
    the tolerance relative to the unpruned model (greedy, least-used first)."""
    full = list(model.edges)
    names, rate, _ = edge_usage(model, runs, X, V, device)
    max_rate = {n: max(rate[r][i] for r in rate) for i, n in enumerate(names)}
    base, _ = evaluate(model, runs, X, V, cfg, device)
    tol = PRUNE

    def acceptable(P):
        for r, b in base["regimes"].items():
            q = P["regimes"].get(r)
            if q is None or b["pos_rmse_m"] is None:
                continue
            if q["pos_rmse_m"] > b["pos_rmse_m"] * (1 + tol["tol_rel"]) + tol["tol_abs_m"]:
                return False
            if q["availability"] < b["availability"] - tol["tol_avail"]:
                return False
        return P["pos_rmse_m"] <= base["pos_rmse_m"] * (1 + tol["tol_rel"]) + tol["tol_abs_m"]

    kept = set(names)
    cands = sorted([n for n in names if max_rate[n] < tol["tau_use"]], key=lambda n: max_rate[n])
    log(f"prune: {len(names)} gated edges, {len(cands)} rarely used (< {tol['tau_use']})")
    trials = []
    for n in cands:
        trial = kept - {n}
        model.set_edges(restrict(full, trial))
        P, _ = evaluate(model, runs, X, V, cfg, device)
        ok = acceptable(P)
        trials.append({"edge": n, "max_rate": max_rate[n], "accepted": ok,
                       "pos_rmse_m": P["pos_rmse_m"]})
        if ok:
            kept = trial
    model.set_edges(restrict(full, kept))
    log(f"prune: kept {len(kept)} / {len(names)} gated edges")
    return kept, {"usage": {"names": names, "rate": rate}, "base": base, "trials": trials,
                  "tolerance": tol}


# ------------------------------------------------------------------------------- variants
def build(variant: str, dims, edges, gated=True):
    if variant == "gru_flat":
        ref = n_params(RecurrentRelGNN(dims, candidate_edges()))
        h = min(range(16, 257, 4), key=lambda hh: abs(n_params(FlatGru(dims, hh)) - ref))
        return FlatGru(dims, h)
    agg = "gcn" if variant.startswith("rgcn") else "gat"
    return RecurrentRelGNN(dims, edges, agg=agg, gated=gated, memory=variant != "rgat_nomem")


def run_variant(variant: str, paths: list, out_dir: Path, seed: int = 0, device="cuda",
                structure_from: Path | None = None, sched: dict | None = None, log=print,
                train_seeds=None, val_seeds=None):
    sched = {**SCHEDULE, **(sched or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_p, va_p = split(paths, train_seeds, val_seeds)
    tr = [load_run(p) for p in tr_p]
    va = [load_run(p) for p in va_p]
    norm = Normalizer().fit(tr)
    Xt, Vt, Ct = stack(tr, norm)
    Xv, Vv, _ = stack(va, norm)
    dims = norm.in_dims
    full = candidate_edges()
    lamE, lamS, lr = sched["lam_E"], sched["lam_S"], sched["lr"]
    kw = dict(device=device, chunk=sched["chunk"], batch=sched["batch"], seed=seed, log=log)
    t0 = time.time()
    report = {"variant": variant, "seed": seed, "train_runs": [Path(p).stem for p in tr_p],
              "val_runs": [Path(p).stem for p in va_p], "schedule": sched}
    total = sched["A"] + sched["C1"] + sched["C2"]
    torch.manual_seed(seed)
    if variant in ("rgat_dyn", "rgcn_dyn"):
        model = build(variant, dims, full)
        hist = fit(model, Xt, Vt, Ct, [Phase(sched["A"], "soft", (0.0, lamE), 0.3, lamS, lr)], **kw)
        kept, prep = prune(model, va, Xv, Vv, device, log=log)
        report["prune"] = prep
        hist += fit(model, Xt, Vt, Ct, [Phase(sched["C1"], "soft", (lamE, lamE), 0, lamS, lr),
                                        Phase(sched["C2"], "hard_st", (lamE, lamE), 0, lamS, lr / 2)], **kw)
    elif variant in ("rgat_static", "rgat_nomem"):
        kept = set(json.loads((structure_from / "structure.json").read_text())["kept_gated"])
        edges = restrict(full, kept)
        if variant == "rgat_static":
            model = build(variant, dims, edges, gated=False)
            hist = fit(model, Xt, Vt, Ct, [Phase(total, "none", (0, 0), 0, lamS, lr)], **kw)
        else:
            model = build(variant, dims, edges)
            hist = fit(model, Xt, Vt, Ct, [Phase(sched["A"] + sched["C1"], "soft", (lamE, lamE), 0, lamS, lr),
                                            Phase(sched["C2"], "hard_st", (lamE, lamE), 0, lamS, lr / 2)], **kw)
    elif variant in ("rgat_dense", "gru_flat"):
        model = build(variant, dims, full, gated=False)
        kept = None
        hist = fit(model, Xt, Vt, Ct, [Phase(total, "none", (0, 0), 0, lamS, lr)], **kw)
    else:
        raise ValueError(variant)
    report["train_s"] = time.time() - t0
    report["history"] = hist
    cfg, grid = tune_select(model, va, Xv, Vv, device, log=log)
    P, per = evaluate(model, va, Xv, Vv, cfg, device)
    report.update(select=vars(cfg), select_grid=grid, val=P, params=n_params(model))
    if hasattr(model, "g_idx") and model.gated:
        names, rate, _ = edge_usage(model, va, Xv, Vv, device)
        report["val_edge_usage"] = {"names": names, "rate": rate}
    torch.save({"state": model.state_dict(), "variant": variant, "dims": dims,
                "edges": [vars(e) for e in model.edges] if hasattr(model, "edges") else None,
                "hidden": getattr(model, "hidden", None)}, out_dir / "model.pt")
    (out_dir / "normalizer.json").write_text(json.dumps(norm.stats))
    if kept is not None:
        (out_dir / "structure.json").write_text(json.dumps(
            {"kept_gated": sorted(kept), "n_gated_full": sum(not e.protected for e in full),
             "protected": [e.name for e in full if e.protected and e.rel != "self"]}, indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"{variant} seed {seed}: val rmse {P['pos_rmse_m']:.3f} avail {P['availability']:.3f} "
        f"switches {P['switches_per_run']:.2f} train {report['train_s'] / 60:.1f} min")
    return model, report


def load_model(d: Path, device="cpu"):
    ck = torch.load(d / "model.pt", map_location=device)
    from .graph import Edge
    v = ck["variant"]
    if v == "gru_flat":
        m = FlatGru(ck["dims"], ck["hidden"])
    else:
        edges = [Edge(**e) for e in ck["edges"]]
        agg = "gcn" if v.startswith("rgcn") else "gat"
        m = RecurrentRelGNN(ck["dims"], edges, agg=agg, gated=v not in ("rgat_dense", "rgat_static"),
                            memory=v != "rgat_nomem")
    m.load_state_dict(ck["state"], strict=False)
    rep = json.loads((d / "report.json").read_text())
    norm = Normalizer(json.loads((d / "normalizer.json").read_text()))
    return m, norm, SelectCfg(**rep["select"]), rep


def latency(model, X1, V1, n: int = 3000, sparse: bool = True):
    """Per-tick CPU latency of one vehicle (B = 1, single thread), operation mode."""
    torch.set_num_threads(1)
    model = copy.deepcopy(model).cpu().eval()
    st = model.init_state(1, "cpu")
    ts, active = [], []
    with torch.no_grad():
        for k in range(min(n, X1.shape[1])):
            x, v = X1[:, k], V1[:, k]
            if not v.any():
                v = torch.ones_like(v)
            t0 = time.perf_counter()
            _, st, aux = model.step(x, st, v, "eval" if sparse else "eval_dense")
            ts.append(time.perf_counter() - t0)
            if "z" in aux:
                active.append(float(aux["z"].mean()))
    ts = np.asarray(ts[50:]) * 1000
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99)), "max_ms": float(ts.max()),
            "active_gated_fraction": float(np.mean(active)) if active else None}
