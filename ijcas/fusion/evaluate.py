"""Evaluation of fused navigation outputs and of the non-learned comparison methods.

Accuracy and covariance consistency are reported separately (a learned, data-dependent weighting does
not inherit CI's conservativeness; docs/FUSION_DESIGN_KO.md):
  accuracy      position / velocity / attitude RMSE over output ticks, p95, max, ATE-style global error
                and a relative (local) error over 10 s windows, availability (output with error <= 2 m
                over the whole evaluation time), output fraction, no-output time, discontinuity
  consistency   position NEES = e^T P_f^-1 e (mean, share above chi2_0.95(3) = 7.815)
  fusion        information weights w, effective contribution of each solution to the fused position
                c_k = tr(P_f w_k P_k^-1) / 3 (sums to 1), effective number of contributors 1 / sum c_k^2
Comparison methods (same states, eligibility and covariances): fixed single solutions, equal-weight
mean (covariances ignored), equal-weight CI, conventional CI (weights minimising tr(P_f,pos) on the
simplex, grid), independent-information fusion (sum of P^-1, double counts the shared IMU), and the
ground-truth single-solution oracle (reference only).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .ci import fuse
from .dataset import SOLUTIONS

ALARM_M = 2.0
CHI2_95_3 = 7.814727903251178


def run_tensors(run: dict, dtype=torch.float64):
    """Numpy run arrays -> fusion inputs [T,3,...] (P assembled from the reported blocks)."""
    T = len(run["t"])
    P = np.zeros((T, 3, 9, 9))
    cov = np.nan_to_num(run["cov3"])
    for b in range(3):
        P[:, :, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = cov[:, :, b]
    av = run["avail3"] & run["valid3"][..., None]
    eye = np.eye(9)
    P = np.where(av.any(-1)[..., None, None], P, eye)
    for b in range(3):                                      # absent blocks: identity dummy (excluded by avail)
        idx = ~av[:, :, b]
        P[idx, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = np.eye(3)
    q = np.where(run["valid3"][..., None], run["quat3"], np.array([0, 0, 0, 1.0]))
    f = lambda x: torch.tensor(np.nan_to_num(x), dtype=dtype)  # noqa: E731
    return f(run["pos3"]), f(run["vel3"]), f(q), torch.tensor(P, dtype=dtype), torch.tensor(av)


def _simplex(n, step):
    k = int(round(1 / step))
    return np.array([c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k], float) / k


def weights_for(run: dict, method: str) -> np.ndarray:
    """[T,3] weights of a non-learned method (0 for ineligible solutions)."""
    v = run["valid3"].astype(float)
    T = len(v)
    if method.startswith("fixed_"):
        k = SOLUTIONS.index(method[6:])
        w = np.zeros((T, 3))
        w[:, k] = v[:, k]
        return w
    if method in ("equal_ci", "equal_mean", "independent"):
        return v / np.maximum(v.sum(1, keepdims=True), 1)
    if method == "ci_min_trace":
        av = run["avail3"][:, :, 0] & run["valid3"]
        tr = np.full((T, 3), np.inf)
        info = np.zeros((T, 3, 3, 3))
        for k in range(3):
            ok = av[:, k]
            if ok.any():
                info[ok, k] = np.linalg.inv(run["cov3"][ok, k, 0])
        grid = _simplex(3, 0.02)                                         # [G,3]
        w = np.zeros((T, 3))
        for t in range(T):
            act = np.where(av[t])[0]
            if len(act) == 0:
                continue
            if len(act) == 1:
                w[t, act[0]] = 1.0
                continue
            g = grid[(grid[:, [k for k in range(3) if k not in act]] == 0).all(1)] if len(act) < 3 else grid
            L = np.einsum("gk,kij->gij", g[:, act], info[t, act])
            tr[t, 0] = 0
            best = np.argmin(np.trace(np.linalg.inv(L), axis1=1, axis2=2))
            w[t] = g[best]
        return w
    if method == "oracle_single":
        e = np.where(run["valid3"], run["err3_pos"], np.inf)
        w = np.zeros((T, 3))
        k = e.argmin(1)
        ok = np.isfinite(e.min(1))
        w[ok, k[ok]] = 1.0
        return w
    raise ValueError(method)


def fuse_run(run: dict, w: np.ndarray, method: str = "ci") -> dict:
    pos, vel, quat, P, av = run_tensors(run)
    wt = torch.tensor(w, dtype=torch.float64)
    if method == "equal_mean":                                           # covariances ignored
        P = torch.eye(9, dtype=torch.float64).expand_as(P).clone()
        av = av & torch.tensor(run["valid3"])[..., None]
        av[:] = torch.tensor(run["valid3"])[..., None]
    o = fuse(pos, vel, quat, P, av, wt, independent=(method == "independent"))
    out = {k: o[k].numpy() for k in ("pos", "vel", "quat", "P", "ok")}
    out["w"] = w
    # effective contribution to the fused position: C_k = P_f w_k Lam_k (sum_k C_k = I)
    wp = w[..., 0] if w.ndim == 3 else w                                 # position-block weights
    c = np.zeros_like(wp)
    okp = out["ok"][:, 0]
    if okp.any() and method != "equal_mean":
        Pf = out["P"][okp, 0:3, 0:3]
        for k in range(3):
            a = av[okp, k, 0].numpy()
            if a.any():
                Lk = np.linalg.inv(P[okp, k, 0:3, 0:3].numpy()[a])
                wk = (wp[okp, k] if method != "independent" else np.ones(okp.sum()))[a]
                c_ok = np.zeros(okp.sum())
                c_ok[a] = np.trace(Pf[a] @ (wk[:, None, None] * Lk), axis1=1, axis2=2) / 3
                c[okp, k] = c_ok
    out["contrib"] = c
    return out


def _rte(t, pos, gt, has, win_s=10.0):
    """Relative (local) position error over windows of win_s: |(p(t+d)-p(t)) - (g(t+d)-g(t))|."""
    dt = float(np.median(np.diff(t)))
    lag = int(round(win_s / dt))
    if len(t) <= lag:
        return np.nan
    ok = has[:-lag] & has[lag:]
    d = (pos[lag:] - pos[:-lag]) - (gt[lag:] - gt[:-lag])
    e = np.linalg.norm(d[ok], axis=1)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def metrics(run: dict, out: dict) -> dict:
    t = run["t"]
    m = t >= run["meta"]["normal_s"] if isinstance(run["meta"], dict) else t >= 20.0
    has = out["ok"][:, 0] & np.isfinite(out["pos"]).all(1)
    ep = np.linalg.norm(out["pos"] - run["gt_pos"], axis=1)
    ev = np.linalg.norm(out["vel"] - run["gt_vel"], axis=1)
    ea = np.full(len(t), np.nan)
    oka = out["ok"][:, 2] & np.isfinite(out["quat"]).all(1)
    if oka.any():
        ea[oka] = np.degrees((R.from_quat(run["gt_quat"][oka]).inv() * R.from_quat(out["quat"][oka])).magnitude())
    sel = m & has
    e = ep[sel]
    nees = np.full(len(t), np.nan)
    Pp = out["P"][:, 0:3, 0:3]
    okn = sel & np.isfinite(Pp).all((1, 2))
    if okn.any():
        d = (out["pos"] - run["gt_pos"])[okn]
        nees[okn] = np.einsum("ti,tij,tj->t", d, np.linalg.inv(Pp[okn]), d)
    w = out["w"][sel]
    w = w[..., 0] if w.ndim == 3 else w
    c = out["contrib"][sel]
    jumps = np.linalg.norm(np.diff(out["pos"], axis=0) - out["vel"][:-1] * np.diff(t)[:, None], axis=1)
    jumps = jumps[(has[1:] & has[:-1] & m[1:])]
    rr = lambda x: float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else None  # noqa: E731
    return {"ticks": int(m.sum()), "n_out": int(sel.sum()), "sum_sq_p": float((e ** 2).sum()),
            "pos_rmse_m": rr(e), "vel_rmse_mps": rr(ev[sel]), "att_rmse_deg": rr(ea[sel]),
            "pos_p95_m": float(np.percentile(e, 95)) if len(e) else None,
            "pos_max_m": float(e.max()) if len(e) else None,
            "rte10_m": _rte(t[m], out["pos"][m], run["gt_pos"][m], has[m]),
            "availability": float((has & (ep <= ALARM_M))[m].mean()),
            "output_fraction": float(has[m].mean()), "no_output_s": float((~has[m]).sum() * np.median(np.diff(t))),
            "nees_pos_mean": float(np.nanmean(nees[sel])) if np.isfinite(nees[sel]).any() else None,
            "nees_pos_over95": float(np.mean(nees[okn & m] > CHI2_95_3)) if okn.any() else None,
            "w_mean": w.mean(0).tolist() if len(w) else None,
            "contrib_mean": c.mean(0).tolist() if len(c) else None,
            "enc_mean": float(np.mean(1 / np.maximum((c ** 2).sum(1), 1e-12))) if len(c) else None,
            "single_ratio": float(np.mean((c > 0.99).any(1))) if len(c) else None,
            "max_jump_m": float(jumps.max()) if len(jumps) else 0.0}


def pool(per_run: list[dict]) -> dict:
    n = sum(r["n_out"] for r in per_run)
    ticks = sum(r["ticks"] for r in per_run)
    out = {"runs": len(per_run), "pos_rmse_m": math.sqrt(sum(r["sum_sq_p"] for r in per_run) / n) if n else None,
           "availability": float(np.average([r["availability"] for r in per_run], weights=[r["ticks"] for r in per_run])),
           "output_fraction": n / ticks if ticks else None}
    for k in ("vel_rmse_mps", "att_rmse_deg", "pos_p95_m", "pos_max_m", "rte10_m", "nees_pos_mean",
              "nees_pos_over95", "enc_mean", "single_ratio", "max_jump_m", "no_output_s"):
        v = [r[k] for r in per_run if r.get(k) is not None and np.isfinite(r[k])]
        out[k] = float(np.max(v)) if k in ("pos_max_m", "max_jump_m") else (float(np.mean(v)) if v else None)
    for k in ("w_mean", "contrib_mean"):
        v = [r[k] for r in per_run if r.get(k)]
        out[k] = np.mean(v, 0).round(3).tolist() if v else None
    return out
