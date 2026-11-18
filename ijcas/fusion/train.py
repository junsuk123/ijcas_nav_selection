"""GT-supervised training of the fusion weights through the differentiable CI.

    L_total = L_fusion + lam_S L_weight_smooth + lam_E L_edge
    L_fusion = mean over ticks with a fused output of
               rho(e_p / s_p) + beta_v rho(e_v / s_v) + beta_R rho(e_R / s_R)
    e_p = |p_f - p_GT|, e_v = |v_f - v_GT|, e_R = |Log(R_GT^T R_f)| [deg]; rho = the v1 Huber
    (ijcas/rgat/learn.py COST); terms of blocks without a fused value are skipped.
    L_weight_smooth = 0.5 |w_t - w_{t-1}|_1 on ticks whose eligible set did not change
    L_edge = mean P(z_e != 0) over gated edges (v1)
Ground truth enters only the loss.  Gradients reach the weight head, the R-GAT, the GRU and the input
adapters; the three estimators are recorded outputs and receive none.
Truncated BPTT over whole runs (memory carried across chunks, detached), minibatches of runs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..rgat.learn import COST
from . import geometry as G
from .ci import fuse
from .evaluate import run_tensors
from .weights import FusionNormalizer, masked_weights


def _rho(x, d=COST["huber"]):
    return torch.where(x <= d, x ** 2, d * (2 * x - d))


def fusion_loss(o, gp, gv, gR):
    """o: fuse() output for one tick batch; gp, gv [B,3], gR [B,3,3].  -> per-sample loss [B], mask [B]."""
    ok = o["ok"]
    zero = torch.zeros_like(gp[:, 0])
    ep = torch.where(ok[:, 0], (torch.nan_to_num(o["pos"]) - gp).norm(dim=-1), zero)
    ev = torch.where(ok[:, 1], (torch.nan_to_num(o["vel"]) - gv).norm(dim=-1), zero)
    Rf = G.quat_to_rot(torch.nan_to_num(o["quat"], nan=0.0) + torch.tensor([0, 0, 0, 1e-12], dtype=gp.dtype,
                                                                            device=gp.device))
    ea = torch.where(ok[:, 2], torch.rad2deg(G.log(gR.transpose(-1, -2) @ Rf).norm(dim=-1)), zero)
    L = (_rho(ep / COST["s_p"]) * ok[:, 0] + COST["beta_v"] * _rho(ev / COST["s_v"]) * ok[:, 1]
         + COST["beta_R"] * _rho(ea / COST["s_R"]) * ok[:, 2])
    return L, ok[:, 0]


@dataclass
class FitCfg:
    epochs: int = 40
    gate_mode: str = "soft"
    lam_E: float = 0.1
    lam_S: float = 0.1
    lr: float = 2e-3
    chunk: int = 200
    batch: int = 40


def stack_runs(runs, norm: FusionNormalizer):
    """-> dict of tensors [B,T,...] padded to the longest run (valid False on padding)."""
    T = max(len(r["t"]) for r in runs)
    B = len(runs)
    out = {"X": np.zeros((B, T, *norm.pack(runs[0]).shape[1:]), np.float32),
           "valid": np.zeros((B, T, 3), bool), "avail": np.zeros((B, T, 3, 3), bool),
           "pos": np.zeros((B, T, 3, 3)), "vel": np.zeros((B, T, 3, 3)), "quat": np.zeros((B, T, 3, 4)),
           "P": np.tile(np.eye(9), (B, T, 3, 1, 1)), "gp": np.zeros((B, T, 3)), "gv": np.zeros((B, T, 3)),
           "gR": np.tile(np.eye(3), (B, T, 1, 1))}
    out["quat"][..., 3] = 1.0
    from scipy.spatial.transform import Rotation as R
    for b, r in enumerate(runs):
        n = len(r["t"])
        out["X"][b, :n] = norm.pack(r)
        pos, vel, quat, P, av = run_tensors(r)
        out["pos"][b, :n], out["vel"][b, :n], out["quat"][b, :n] = pos.numpy(), vel.numpy(), quat.numpy()
        out["P"][b, :n], out["avail"][b, :n] = P.numpy(), av.numpy()
        out["valid"][b, :n] = r["valid3"]
        out["gp"][b, :n], out["gv"][b, :n] = r["gt_pos"], r["gt_vel"]
        out["gR"][b, :n] = R.from_quat(r["gt_quat"]).as_matrix()
    return {k: torch.from_numpy(v) for k, v in out.items()}


def fit(model, data: dict, cfg: FitCfg, device="cuda", seed: int = 0, log=print, phase_name="A"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    B, T = data["X"].shape[:2]
    block = getattr(model, "head_mode", "scalar") == "block"
    hist = []
    for ep in range(cfg.epochs):
        t0 = time.time()
        tot = {"fusion": 0.0, "smooth": 0.0, "edge": 0.0, "n": 0}
        perm = torch.randperm(B)
        for bs in range(0, B, cfg.batch):
            idx = perm[bs:bs + cfg.batch]
            st = model.init_state(len(idx), device)
            prev_w = prev_v = None
            for s in range(0, T, cfg.chunk):
                sl = slice(s, s + cfg.chunk)
                g = {k: v[idx, sl].to(device) for k, v in data.items()}
                Ls, Ms, ws, ps = [], [], [], []
                for k in range(g["X"].shape[1]):
                    v = g["valid"][:, k]
                    lg, st, aux = model.step(g["X"][:, k], st, v[..., None] if block else v, cfg.gate_mode)
                    w = masked_weights(lg, v).double()
                    o = fuse(g["pos"][:, k], g["vel"][:, k], g["quat"][:, k], g["P"][:, k], g["avail"][:, k], w)
                    L, M = fusion_loss(o, g["gp"][:, k], g["gv"][:, k], g["gR"][:, k])
                    Ls.append(L), Ms.append(M), ws.append(w)
                    if "p_active" in aux:
                        ps.append(aux["p_active"])
                Ls, Ms, W = torch.stack(Ls, 1), torch.stack(Ms, 1), torch.stack(ws, 1)
                l_f = (Ls * Ms).sum() / Ms.sum().clamp_min(1)
                V = g["valid"]
                if prev_w is not None:
                    W2, V2 = torch.cat([prev_w, W], 1), torch.cat([prev_v, V], 1)
                else:
                    W2, V2 = W, V
                same = (V2[:, 1:] == V2[:, :-1]).all(-1) & V2[:, 1:].any(-1)
                dW = 0.5 * (W2[:, 1:] - W2[:, :-1]).abs().flatten(2).sum(-1)
                l_s = (dW * same).sum() / same.sum().clamp_min(1)
                l_e = torch.stack(ps, 1).mean() if ps else torch.zeros((), device=device)
                loss = l_f + cfg.lam_S * l_s + cfg.lam_E * l_e
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(loss) and torch.isfinite(gn):
                    opt.step()
                st = {k: v.detach() for k, v in st.items()}
                prev_w, prev_v = W[:, -1:].detach(), V[:, -1:]
                tot["fusion"] += l_f.item()
                tot["smooth"] += l_s.item()
                tot["edge"] += float(l_e)
                tot["n"] += 1
        rec = {k: round(v / max(tot["n"], 1), 5) for k, v in tot.items() if k != "n"}
        rec.update(epoch=ep, phase=phase_name, gate_mode=cfg.gate_mode, sec=round(time.time() - t0, 1))
        hist.append(rec)
        log(json.dumps(rec))
    return hist


@torch.no_grad()
def infer_weights(model, data: dict, device="cuda") -> np.ndarray:
    """Operation-mode weights [B,T,3(,3)] (binary gates with hysteresis)."""
    model.to(device).eval()
    B, T = data["X"].shape[:2]
    block = getattr(model, "head_mode", "scalar") == "block"
    st = model.init_state(B, device)
    W = []
    for k in range(T):
        v = data["valid"][:, k].to(device)
        lg, st, _ = model.step(data["X"][:, k].to(device), st, v[..., None] if block else v, "eval")
        W.append(masked_weights(lg, v).double().cpu())
    return torch.stack(W, 1).numpy()
