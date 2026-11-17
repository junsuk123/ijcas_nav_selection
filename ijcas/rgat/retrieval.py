"""Retrieval-based navigation-algorithm selection (CR, design 12): the R-GAT is used offline only.

Operation per tick, without any graph network:
  s_t   = Phi(y_<=t)                 node inputs of the R-GAT (normalised value, missing mask, log age)
                                     + history of the value channels (EMA 2 s / 10 s, 10 s decaying peak)
  q_t^k = P_k s_t^k                  fixed linear map of candidate k's case space (columns of N_k and of
                                     the nodes with a relation into N_k in E: E0 or the pruned E*)
  c^_k  = RBF-weighted stored cost of the 16 nearest representative cases of candidate k
  pi    = softmax(-c^) over the eligible candidates -> the switching rule shared with every variant
  hold  = some eligible candidate's query is farther from its nearest case than the 99th percentile of
          the training ticks -> no learned switch at this tick (policy switches still happen)

Stored per case: the kernel-weighted mean of the *measured* candidate costs of the training ticks
around it (recomputed every epoch; never a free parameter) and its support.  Training (supervised
variants): P and the case positions, PCA / k-means initialised, minimise the same objective as the
graph models (expected regret + KL(q || pi)); cr_gstar_kd adds KL(pi_RGAT || pi).
Cases come from the training runs (validation seeds 1-7) only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .features import SPEC
from .graph import CANDIDATES, NAV_OF, NODES, candidate_edges, restrict
from .learn import (AUX, Normalizer, SelectCfg, costs, infer, load_run, masked_softmax,
                    metrics_for_selection, pool, select, stack)

HIST = {"ema_s": (2.0, 10.0), "peak_s": 10.0}
CR = {"d": 16, "M": 256, "k": 16, "d_plain": 32, "M_plain": 1024, "epochs": 30, "batch": 4096,
      "lr": 3e-3, "tau_s": 1.0, "lam_kd": 0.5, "abstain_q": 0.99, "min_support": 1.0,
      "init_sub": 60000}
VARIANTS = ("cr_plain", "cr_e0", "cr_gstar", "cr_gstar_kd")
CHANNELS = ("val", "mask", "dt", "ema2", "ema10", "peak")
K = len(CANDIDATES)


# ------------------------------------------------------------------------------- situation features
def columns() -> list[tuple[str, str, str]]:
    """(node, feature, channel) of every column of s_t, in storage order."""
    return [(n, f, ch) for n in NODES for ch in CHANNELS for f in SPEC[n]]


def _node_slices(norm: Normalizer):
    fmax = max(norm.in_dims)
    out = []
    for i, n in enumerate(NODES):
        F = len(SPEC[n])
        out.append((i * fmax, F))
    return out, fmax


class History:
    """Causal EMA / decaying-peak state of the value channels (O(D) per tick)."""

    def __init__(self, n_val: int):
        self.n = n_val
        self.reset()

    def reset(self):
        self.e = None
        self.p = np.zeros(self.n, np.float32)

    def update(self, v: np.ndarray, dt: float) -> np.ndarray:
        (t1, t2), tp = HIST["ema_s"], HIST["peak_s"]
        if self.e is None:
            self.e = np.stack([v, v]).astype(np.float32)
        else:
            a = np.array([1 - np.exp(-dt / t1), 1 - np.exp(-dt / t2)], np.float32)[:, None]
            self.e += a * (v[None] - self.e)
        self.p = np.maximum(np.abs(v), self.p * np.float32(np.exp(-dt / tp)))
        return np.concatenate([self.e[0], self.e[1], self.p])


def situation(run: dict, norm: Normalizer) -> np.ndarray:
    """s_t for every tick of a run -> [T, D_all] float32 (same values the runtime computes)."""
    Xp = norm.pack(run).reshape(len(run["t"]), -1)
    sl, _ = _node_slices(norm)
    t = run["t"]
    dt = np.diff(t, prepend=t[0] - 0.05)
    vals = np.concatenate([Xp[:, o:o + F] for o, F in sl], 1)
    h = History(vals.shape[1])
    H = np.stack([h.update(vals[i], float(dt[i])) for i in range(len(t))])
    nv = vals.shape[1]
    out, c = [], 0
    for (o, F) in sl:
        out += [Xp[:, o:o + 3 * F],                                   # val, mask, dt
                H[:, c:c + F], H[:, nv + c:nv + c + F], H[:, 2 * nv + c:2 * nv + c + F]]
        c += F
    return np.concatenate(out, 1).astype(np.float32)


def groups_for(variant: str, structure: dict | None = None) -> list[dict]:
    meta = columns()
    if variant == "cr_plain":
        cols = [i for i, (_, f, _) in enumerate(meta) if not f.startswith("state.")]
        return [{"cands": list(range(K)), "cols": cols, "d": CR["d_plain"], "M": CR["M_plain"],
                 "nodes": list(NODES)}]
    E = candidate_edges() if variant == "cr_e0" else restrict(candidate_edges(), set(structure["kept_gated"]))
    out = []
    for k, c in enumerate(CANDIDATES):
        nk = NAV_OF[c]
        nodes = {e.src for e in E if e.dst == nk} | {nk}
        out.append({"cands": [k], "cols": [i for i, (n, _, _) in enumerate(meta) if n in nodes],
                    "d": CR["d"], "M": CR["M"], "nodes": sorted(nodes)})
    return out


# ------------------------------------------------------------------------------- case base
class CaseBase:
    """Per-group projection + representative cases + stored measured costs (torch for training)."""

    def __init__(self, groups: list[dict], device="cpu"):
        self.groups = groups
        self.device = device
        self.P, self.B, self.lg, self.mu, self.sd = [], [], [], [], []
        self.Z, self.S, self.d99, self.cmean = [], [], [], None

    # ---- forward on standardised inputs
    def _x(self, g, S):
        if not hasattr(self, "_cols"):
            self._cols = {}
        key = (g, S.device)
        if key not in self._cols:
            self._cols[key] = torch.as_tensor(self.groups[g]["cols"], device=S.device)
        x = (S.index_select(1, self._cols[key]) - self.mu[g]) / self.sd[g]
        return x.clamp(-10, 10)

    def neighbours(self, g, S):
        q = self._x(g, S) @ self.P[g].T
        d2 = torch.cdist(q, self.B[g]).pow(2)
        d2k, idx = torch.topk(d2, CR["k"], dim=1, largest=False)
        w = torch.softmax(-self.lg[g].exp() * d2k, 1)
        return w, idx, d2k[:, 0].clamp_min(0).sqrt()

    def estimate(self, S):
        """-> c_hat [n, K], dmin [n, K] (distance to the nearest case in the candidate's space)."""
        n = S.shape[0]
        ch = torch.zeros(n, K, device=S.device)
        dmin = torch.zeros(n, K, device=S.device)
        for g, grp in enumerate(self.groups):
            w, idx, dm = self.neighbours(g, S)
            for j, k in enumerate(grp["cands"]):
                ok = (self.S[g][j] >= CR["min_support"]).float()[idx]          # [n, k]
                wz = w * ok
                num = (wz * self.Z[g][j][idx]).sum(1)
                den = wz.sum(1)
                ch[:, k] = torch.where(den > 1e-9, num / den.clamp_min(1e-9), self.cmean[k])
                dmin[:, k] = dm
        return ch, dmin

    @torch.no_grad()
    def refresh_costs(self, S_cpu, C_cpu, V_cpu, chunk=32768):
        """Stored case costs = kernel-weighted mean of measured costs of the training ticks."""
        dev = self.device
        self.Z, self.S = [], []
        for g, grp in enumerate(self.groups):
            num = torch.zeros(len(grp["cands"]), grp["M"], device=dev)
            den = torch.zeros_like(num)
            for s in range(0, len(S_cpu), chunk):
                Sb = S_cpu[s:s + chunk].to(dev)
                w, idx, _ = self.neighbours(g, Sb)
                for j, k in enumerate(grp["cands"]):
                    v = V_cpu[s:s + chunk, k].to(dev).float()[:, None]
                    c = C_cpu[s:s + chunk, k].to(dev)[:, None]
                    num[j].scatter_add_(0, idx.reshape(-1), (w * v * c).reshape(-1))
                    den[j].scatter_add_(0, idx.reshape(-1), (w * v).reshape(-1))
            Z = torch.where(den > 0, num / den.clamp_min(1e-12),
                            torch.stack([self.cmean[k] for k in grp["cands"]])[:, None].expand_as(num))
            self.Z.append(Z)
            self.S.append(den)

    # ---- persistence
    def to_npz(self, path: Path, extra: dict):
        arr = {"cmean": self.cmean.cpu().numpy()}
        for g, grp in enumerate(self.groups):
            arr[f"g{g}_cols"] = np.asarray(grp["cols"])
            arr[f"g{g}_cands"] = np.asarray(grp["cands"])
            for name, lst in (("mu", self.mu), ("sd", self.sd), ("P", self.P), ("B", self.B),
                              ("lg", self.lg), ("Z", self.Z), ("S", self.S)):
                arr[f"g{g}_{name}"] = lst[g].detach().cpu().numpy()
            arr[f"g{g}_d99"] = np.asarray(self.d99[g])
        arr["meta"] = np.asarray(json.dumps({**extra, "groups": [
            {k: v for k, v in grp.items() if k != "cols"} for grp in self.groups]}))
        np.savez(path, **arr)

    @classmethod
    def from_npz(cls, path: Path, device="cpu"):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        groups = []
        for g, grp in enumerate(meta["groups"]):
            groups.append({**grp, "cols": z[f"g{g}_cols"].tolist()})
        cb = cls(groups, device)
        T = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        for g in range(len(groups)):
            cb.mu.append(T(z[f"g{g}_mu"])); cb.sd.append(T(z[f"g{g}_sd"]))
            cb.P.append(T(z[f"g{g}_P"])); cb.B.append(T(z[f"g{g}_B"])); cb.lg.append(T(z[f"g{g}_lg"]))
            cb.Z.append(T(z[f"g{g}_Z"])); cb.S.append(T(z[f"g{g}_S"])); cb.d99.append(float(z[f"g{g}_d99"]))
        cb.cmean = T(z["cmean"])
        return cb, meta

    def n_stored(self) -> int:
        n = int(self.cmean.numel())
        for g in range(len(self.groups)):
            n += sum(int(x.numel()) for x in (self.mu[g], self.sd[g], self.P[g], self.B[g], self.lg[g],
                                              self.Z[g], self.S[g])) + 1
        return n


# ------------------------------------------------------------------------------- data
def _ticks(runs, norm):
    S = np.concatenate([situation(r, norm) for r in runs])
    CV = [costs(r) for r in runs]
    C = np.concatenate([c for c, _ in CV])
    V = np.concatenate([v for _, v in CV])
    return torch.from_numpy(S), torch.from_numpy(C), torch.from_numpy(V)


def _init(cb: CaseBase, S, C, V, seed: int, log=print):
    from sklearn.cluster import MiniBatchKMeans
    dev = cb.device
    g_rng = np.random.default_rng(seed)
    sub = torch.from_numpy(g_rng.choice(len(S), min(CR["init_sub"], len(S)), replace=False))
    cm = []
    for k in range(K):
        m = V[:, k]
        cm.append(C[m, k].mean() if m.any() else torch.tensor(0.0))
    cb.cmean = torch.stack(cm).to(dev)
    for g, grp in enumerate(cb.groups):
        x = S[:, grp["cols"]]
        mu = x.mean(0)
        sd = x.std(0).clamp_min(1e-3)
        cb.mu.append(mu.to(dev)); cb.sd.append(sd.to(dev))
        xs = ((x[sub] - mu) / sd).clamp(-10, 10).to(dev)
        torch.manual_seed(seed)
        _, _, Vp = torch.pca_lowrank(xs, q=grp["d"], center=True)
        P = Vp[:, :grp["d"]].T.contiguous()
        q = (xs @ P.T).cpu().numpy()
        km = MiniBatchKMeans(n_clusters=grp["M"], random_state=seed, n_init=3, batch_size=4096).fit(q)
        B = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
        d2 = torch.cdist(torch.as_tensor(q, device=dev), B).pow(2).min(1).values
        cb.P.append(P); cb.B.append(B)
        cb.lg.append(torch.log(1.0 / d2.median().clamp_min(1e-6)).reshape(()))
        log(f"cr init group {g}: D={len(grp['cols'])} d={grp['d']} M={grp['M']} nodes={grp['nodes']}")


def _set_d99(cb: CaseBase, S, chunk=32768):
    with torch.no_grad():
        for g in range(len(cb.groups)):
            dm = torch.cat([cb.neighbours(g, S[s:s + chunk].to(cb.device))[2].cpu()
                            for s in range(0, len(S), chunk)])
            cb.d99.append(float(torch.quantile(dm[torch.randperm(len(dm))[:200000]], CR["abstain_q"])))


def _loss(cb: CaseBase, Sb, Cb, Vb, pt=None):
    ch, _ = cb.estimate(Sb)
    lg = -ch / CR["tau_s"]
    pi = masked_softmax(lg, Vb)
    cmin = torch.where(Vb, Cb, torch.full_like(Cb, 1e9)).min(-1, keepdim=True).values
    l_nav = ((Cb - cmin).clamp_min(0) * pi).sum(-1).mean()
    q = masked_softmax(-(Cb - cmin).clamp_min(0) / AUX["tau"], Vb)
    logp = torch.log_softmax(lg.masked_fill(~Vb, -1e9), -1).masked_fill(~Vb, 0.0)
    l_ce = (torch.special.xlogy(q, q) - q * logp).sum(-1).mean()
    l_kd = (torch.special.xlogy(pt, pt) - pt * logp).sum(-1).mean() if pt is not None else torch.zeros(())
    return l_nav, l_ce, l_kd


@torch.no_grad()
def _objective(cb: CaseBase, S, C, V, idx, teacher=None, chunk=32768):
    """Full training-split objective (the quantity minimised), used to keep the best epoch."""
    tot, n = 0.0, 0
    for s in range(0, len(idx), chunk):
        b = idx[s:s + chunk]
        pt = teacher[b].to(cb.device) if teacher is not None else None
        l_nav, l_ce, l_kd = _loss(cb, S[b].to(cb.device), C[b].to(cb.device), V[b].to(cb.device), pt)
        tot += float(l_nav + AUX["lam_ce"] * l_ce + CR["lam_kd"] * l_kd.to(l_nav.device)) * len(b)
        n += len(b)
    return tot / max(n, 1)


def _snapshot(cb: CaseBase):
    return [[t.detach().clone() for t in lst] for lst in (cb.P, cb.B, cb.lg, cb.Z, cb.S)]


def _restore(cb: CaseBase, snap):
    cb.P, cb.B, cb.lg, cb.Z, cb.S = [[t.clone() for t in lst] for lst in snap]


def fit_cr(cb: CaseBase, S, C, V, teacher=None, seed=0, supervised=True, log=print):
    """Stability (2026-11-17, from the training curves only): cosine learning-rate decay, and the
    epoch with the lowest full training-split objective is kept (stored costs jump when a case
    absorbs or loses FAST-LIO divergence ticks, so the last epoch is not necessarily the best)."""
    _init(cb, S, C, V, seed, log)
    cb.refresh_costs(S, C, V)
    hist = []
    if supervised:
        dev = cb.device
        params = []
        for g in range(len(cb.groups)):
            for t in (cb.P[g], cb.B[g], cb.lg[g]):
                t.requires_grad_(True)
                params.append(t)
        opt = torch.optim.Adam(params, lr=CR["lr"])
        anyv = V.any(1)
        idx_all = torch.nonzero(anyv).squeeze(1)
        gen = torch.Generator().manual_seed(seed)
        best = (_objective(cb, S, C, V, idx_all, teacher), -1, _snapshot(cb))
        for ep in range(CR["epochs"]):
            for gr in opt.param_groups:
                gr["lr"] = CR["lr"] * 0.5 * (1 + np.cos(np.pi * ep / CR["epochs"]))
            t0 = time.time()
            perm = idx_all[torch.randperm(len(idx_all), generator=gen)]
            tot = {"nav": 0.0, "ce": 0.0, "kd": 0.0, "n": 0}
            for s in range(0, len(perm), CR["batch"]):
                b = perm[s:s + CR["batch"]]
                Sb, Cb, Vb = S[b].to(dev), C[b].to(dev), V[b].to(dev)
                pt = teacher[b].to(dev) if teacher is not None else None
                l_nav, l_ce, l_kd = _loss(cb, Sb, Cb, Vb, pt)
                loss = l_nav + AUX["lam_ce"] * l_ce
                if pt is not None:
                    loss = loss + CR["lam_kd"] * l_kd
                    tot["kd"] += float(l_kd)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(loss) and torch.isfinite(gn):
                    opt.step()
                tot["nav"] += float(l_nav); tot["ce"] += float(l_ce); tot["n"] += 1
            for t in params:
                t.requires_grad_(False)
            cb.refresh_costs(S, C, V)
            obj = _objective(cb, S, C, V, idx_all, teacher)
            if obj < best[0]:
                best = (obj, ep, _snapshot(cb))
            for t in params:
                t.requires_grad_(True)
            rec = {k: round(v / max(tot["n"], 1), 5) for k, v in tot.items() if k != "n"}
            rec.update(epoch=ep, train_objective=round(obj, 5), sec=round(time.time() - t0, 1))
            hist.append(rec)
            log(json.dumps(rec))
        for t in params:
            t.requires_grad_(False)
        _restore(cb, best[2])
        log(json.dumps({"kept_epoch": best[1], "train_objective": round(best[0], 5)}))
        hist.append({"kept_epoch": best[1], "train_objective": best[0]})
    _set_d99(cb, S)
    return hist


# ------------------------------------------------------------------------------- inference / evaluation
@torch.no_grad()
def predict(cb: CaseBase, runs, norm, chunk=32768):
    """-> per run (pi [n,K], hold [n] bool, dmin [n,K])."""
    out = []
    for r in runs:
        S = torch.from_numpy(situation(r, norm))
        ch, dm = [], []
        for s in range(0, len(S), chunk):
            c, d = cb.estimate(S[s:s + chunk].to(cb.device))
            ch.append(c.cpu()); dm.append(d.cpu())
        ch, dm = torch.cat(ch), torch.cat(dm)
        V = torch.from_numpy(r["valid"])
        pi = masked_softmax(-ch / CR["tau_s"], V).numpy()
        d99 = np.zeros(K)
        for g, grp in enumerate(cb.groups):
            for k in grp["cands"]:
                d99[k] = cb.d99[g]
        far = dm.numpy() > d99[None]
        hold = (far & r["valid"]).any(1)
        out.append((pi, hold, dm.numpy()))
    return out


def evaluate_cr(runs, preds, cfg: SelectCfg):
    per = []
    for r, (pi, hold, _) in zip(runs, preds):
        sel, _ = select(pi, r["valid"], r["t"], cfg, hold=hold)
        per.append(metrics_for_selection(r, sel))
    return pool(per), per


def tune_select_cr(runs, preds, log=print):
    from .experiment import SELECT_GRID
    rows = [(c, evaluate_cr(runs, preds, c)[0]) for c in SELECT_GRID]
    best = min(P["pos_rmse_m"] for _, P in rows)
    ok = [(c, P) for c, P in rows if P["pos_rmse_m"] <= best * 1.02]
    c, P = min(ok, key=lambda x: (x[1]["switches_per_run"], -x[0].dwell_s))
    log(f"select tuned: margin {c.margin} dwell {c.dwell_s} -> rmse {P['pos_rmse_m']:.3f} "
        f"switches {P['switches_per_run']:.2f}")
    return c, [(vars(c), {k: P[k] for k in ("pos_rmse_m", "availability", "switches_per_run")})
               for c, P in rows]


def run_cr_variant(variant: str, paths: list, out_dir: Path, seed: int = 0, device="cuda",
                   structure_from: Path | None = None, teacher_from: Path | None = None, log=print):
    from .experiment import TRAIN_SEEDS, VAL_SEEDS, load_model, split
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_p, va_p = split(paths, TRAIN_SEEDS, VAL_SEEDS)
    tr = [load_run(p) for p in tr_p]
    va = [load_run(p) for p in va_p]
    norm = Normalizer().fit(tr)
    structure = json.loads((structure_from / "structure.json").read_text()) if structure_from else None
    groups = groups_for(variant, structure)
    t0 = time.time()
    S, C, V = _ticks(tr, norm)
    teacher = None
    if variant == "cr_gstar_kd":
        model, tnorm, _, _ = load_model(teacher_from, device)
        Xt, Vt, _ = stack(tr, tnorm)
        pis = infer(model, Xt, Vt, device, keep_gates=False)["pi"]
        teacher = torch.from_numpy(np.concatenate([pis[b, :len(r["t"])] for b, r in enumerate(tr)]))
        del model, Xt
    torch.manual_seed(seed)
    cb = CaseBase(groups, device)
    hist = fit_cr(cb, S, C, V, teacher, seed, supervised=variant != "cr_plain", log=log)
    train_s = time.time() - t0
    preds = predict(cb, va, norm)
    cfg, grid = tune_select_cr(va, preds, log)
    P, per = evaluate_cr(va, preds, cfg)
    hold_rate = float(np.mean(np.concatenate([h for _, h, _ in preds])))
    report = {"variant": variant, "seed": seed, "train_runs": [Path(p).stem for p in tr_p],
              "val_runs": [Path(p).stem for p in va_p], "cr": CR, "hist_cfg": HIST,
              "structure_from": str(structure_from) if structure_from else None,
              "teacher_from": str(teacher_from) if teacher_from else None,
              "groups": [{"cands": g["cands"], "nodes": g["nodes"], "D": len(g["cols"]), "d": g["d"],
                          "M": g["M"]} for g in groups],
              "train_s": train_s, "history": hist, "select": vars(cfg), "select_grid": grid, "val": P,
              "val_hold_rate": hold_rate, "params": cb.n_stored()}
    cb.to_npz(out_dir / "model.npz", {"variant": variant, "seed": seed})
    (out_dir / "normalizer.json").write_text(json.dumps(norm.stats))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"{variant} seed {seed}: val rmse {P['pos_rmse_m']:.3f} avail {P['availability']:.3f} "
        f"switches {P['switches_per_run']:.2f} hold {hold_rate:.3f} stored {cb.n_stored()} "
        f"train {train_s / 60:.1f} min")
    return cb, report


def load_cr(d: Path, device="cpu"):
    cb, meta = CaseBase.from_npz(d / "model.npz", device)
    norm = Normalizer(json.loads((d / "normalizer.json").read_text()))
    rep = json.loads((d / "report.json").read_text())
    return cb, norm, SelectCfg(**rep["select"]), rep


# ------------------------------------------------------------------------------- operation (numpy, B = 1)
class Runtime:
    """What runs on the vehicle: history update, projection, case search, cost estimate.  No torch."""

    def __init__(self, cb: CaseBase, norm: Normalizer):
        self.norm = norm
        self.sl, _ = _node_slices(norm)
        self.nv = sum(F for _, F in self.sl)
        self.hist = History(self.nv)
        self.g = []
        for g, grp in enumerate(cb.groups):
            np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
            sd = np_(cb.sd[g])
            P = np_(cb.P[g])
            self.g.append({"cols": np.asarray(grp["cols"]), "cands": list(grp["cands"]), "mu": np_(cb.mu[g]),
                           "isd": (1.0 / sd).astype(np.float32), "P": P, "B": np_(cb.B[g]),
                           "Bn": (np_(cb.B[g]) ** 2).sum(1), "gam": float(cb.lg[g].exp()),
                           "Z": np_(cb.Z[g]), "ok": (np_(cb.S[g]) >= CR["min_support"]).astype(np.float32),
                           "d99": cb.d99[g]})
        self.cmean = cb.cmean.cpu().numpy()

    def reset(self):
        self.hist.reset()

    def features(self, xrow: np.ndarray, dt: float) -> np.ndarray:
        vals = np.concatenate([xrow[o:o + F] for o, F in self.sl])
        H = self.hist.update(vals, dt)
        out, c, nv = [], 0, self.nv
        for (o, F) in self.sl:
            out += [xrow[o:o + 3 * F], H[c:c + F], H[nv + c:nv + c + F], H[2 * nv + c:2 * nv + c + F]]
            c += F
        return np.concatenate(out)

    def estimate(self, s: np.ndarray):
        ch = np.empty(K, np.float32)
        far = np.zeros(K, bool)
        kk = CR["k"]
        for g in self.g:
            x = np.clip((s[g["cols"]] - g["mu"]) * g["isd"], -10, 10)
            q = g["P"] @ x
            d2 = g["Bn"] - 2.0 * (g["B"] @ q) + q @ q
            idx = np.argpartition(d2, kk)[:kk]
            d2k = np.maximum(d2[idx], 0.0)
            w = np.exp(-g["gam"] * (d2k - d2k.min()))
            w /= w.sum()
            dmin = float(np.sqrt(d2k.min()))
            for j, k in enumerate(g["cands"]):
                wz = w * g["ok"][j, idx]
                den = wz.sum()
                ch[k] = (wz @ g["Z"][j, idx]) / den if den > 1e-9 else self.cmean[k]
                far[k] = dmin > g["d99"]
        return ch, far

    def step(self, xrow, dt, valid):
        s = self.features(xrow, dt)
        ch, far = self.estimate(s)
        lg = np.where(valid, -ch / CR["tau_s"], -np.inf)
        if not valid.any():
            return np.zeros(K, np.float32), False
        e = np.exp(lg - lg.max())
        return e / e.sum(), bool((far & valid).any())


def latency(cb: CaseBase, norm: Normalizer, run: dict, n: int = 3000) -> dict:
    """Per-tick CPU cost of the operation path (single thread, B = 1), split into its parts."""
    rt = Runtime(cb, norm)
    Xp = norm.pack(run).reshape(len(run["t"]), -1).astype(np.float32)
    t = run["t"]
    dt = np.diff(t, prepend=t[0] - 0.05)
    tf, te, tt = [], [], []
    for i in range(min(n, len(t))):
        v = run["valid"][i] if run["valid"][i].any() else np.ones(K, bool)
        t0 = time.perf_counter()
        s = rt.features(Xp[i], float(dt[i]))
        t1 = time.perf_counter()
        ch, far = rt.estimate(s)
        lg = np.where(v, -ch / CR["tau_s"], -np.inf)
        e = np.exp(lg - lg.max())
        _ = (e / e.sum(), bool((far & v).any()))
        t2 = time.perf_counter()
        tf.append(t1 - t0); te.append(t2 - t1); tt.append(t2 - t0)

    def q(a):
        a = np.asarray(a[50:]) * 1000
        return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}
    return {"total": q(tt), "features": q(tf), "projection_search": q(te),
            "stored_floats": cb.n_stored(), "stored_kb": cb.n_stored() * 4 / 1024,
            "dims": [len(g["cols"]) for g in cb.groups]}
