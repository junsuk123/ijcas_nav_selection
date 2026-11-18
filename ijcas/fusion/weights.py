"""Fusion information weights from the ontology-constrained recurrent R-GAT (and comparison models).

The network reuses ijcas/rgat/model.py (node input adapters, node-type GRU, dynamic edge gates,
relational attention) on the 12-node fusion graph (ijcas/fusion/graph.py) and replaces the selection
score head by a fusion-weight head:

    s_k = MLP([h_k ; h_F]) + c_k              (h_k: solution node, h_F: fusion node)
    w_k = masked softmax_k(s_k)               (0 for ineligible solutions, sum 1)

head="block" gives one weight per solution and state block (position, velocity, attitude): [B,3,3].
Besides the weights, the head outputs u [B,3,3] for the learned covariance inflation gamma >= 1 of each
solution and block (user decision B; ijcas/fusion/ci.py inflate).  ``split`` separates the two.
w is an *information weight* of the covariance intersection (ijcas/fusion/ci.py); the contribution of
a solution to the fused state also depends on its covariance and is reported separately
(ijcas/fusion/evaluate.py).  Attention coefficients are message-aggregation weights, not w.

Inputs per node: [y_norm, m, dt_log] as in v1.  Solution nodes get, besides their v1 features, the
log-traces of their reported covariance blocks (NaN = not reported -> m = 0, never filled).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..rgat.features import SPEC
from ..rgat.model import FlatGru, RecurrentRelGNN
from . import graph as FG

DT_MAX = 600.0
V1_NODE = {"N_gi": 0, "N_vio": 1, "N_lio": 2}


def node_features(run: dict) -> dict:
    """{node: (y [T,F], m [T,F], dt [T,F])} for the fusion graph nodes."""
    T = len(run["t"])
    out = {}
    for n in FG.NODES:
        if n == "F_nav":
            y = run["y_F_nav"].astype(np.float32)
            out[n] = (y, np.isfinite(y).astype(np.float32), np.zeros_like(y))
            continue
        y, m, dt = run[f"y_{n}"], run[f"m_{n}"], run[f"dt_{n}"]
        if n in V1_NODE:
            k = V1_NODE[n]
            yc = run["y_cov"][:, k, :].astype(np.float32)
            age = np.nan_to_num(run["age3"][:, k:k + 1], nan=DT_MAX).astype(np.float32)
            y = np.concatenate([y, yc], 1)
            m = np.concatenate([m, np.isfinite(yc).astype(np.float32)], 1)
            dt = np.concatenate([dt, np.where(np.isfinite(yc), np.repeat(age, 3, 1), DT_MAX)], 1)
        out[n] = (y, m, dt)
    assert all(v[0].shape[0] == T for v in out.values())
    return out


def feature_dims() -> list[int]:
    d = {n: len(SPEC[n]) for n in SPEC}
    return [3 * (3 if n == "F_nav" else d[n] + (3 if n in V1_NODE else 0)) for n in FG.NODES]


class FusionNormalizer:
    """Mean / std of present values per node and feature over the training runs only."""

    def __init__(self, stats: dict | None = None):
        self.stats = stats or {}

    def fit(self, runs):
        feats = [node_features(r) for r in runs]
        for n in FG.NODES:
            y = np.concatenate([f[n][0] for f in feats])
            mu = np.nanmean(np.where(np.isfinite(y), y, np.nan), 0)
            sd = np.nanstd(np.where(np.isfinite(y), y, np.nan), 0)
            self.stats[n] = (np.where(np.isfinite(mu), mu, 0.0).tolist(),
                             np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).tolist())
        return self

    def pack(self, run) -> np.ndarray:
        f = node_features(run)
        T = len(run["t"])
        dims = feature_dims()
        X = np.zeros((T, len(FG.NODES), max(dims)), np.float32)
        for i, n in enumerate(FG.NODES):
            y, m, dt = f[n]
            mu, sd = (np.asarray(a, np.float32) for a in self.stats[n])
            yn = np.where(np.isfinite(y), np.clip((y - mu) / sd, -10, 10), 0.0)
            dtl = np.log1p(np.clip(dt, 0, DT_MAX)) / math.log1p(DT_MAX)
            F = y.shape[1]
            X[:, i, :3 * F] = np.concatenate([yn, m, dtl], 1)
        return X


class FusionRelGNN(RecurrentRelGNN):
    def __init__(self, in_dims, edges, head: str = "scalar", ontology_constraints: bool = True, **kw):
        super().__init__(in_dims, edges, spec=FG.spec(ontology_constraints), **kw)
        d = self.d
        self.head_mode = head
        self.n_w = 3 if head == "block" else 1
        del self.head
        self.fhead = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(), nn.Linear(d, self.n_w + 3))
        self.cand_bias = nn.Parameter(torch.zeros(3, self.n_w + 3))
        self.register_buffer("fus_idx", torch.tensor([FG.NODES.index("F_nav")]))

    def scores(self, v):
        hk = v[:, self.nav_idx]                                         # [B,3,d]
        hf = v[:, self.fus_idx].expand_as(hk)
        return self.fhead(torch.cat([hk, hf], -1)) + self.cand_bias     # [B,3,n_w+3]


class FusionGru(FlatGru):
    """Non-graph comparison: concatenated node inputs -> GRU -> weights (same heads)."""

    def __init__(self, in_dims, hidden: int = 96, head: str = "scalar"):
        self.head_mode = head
        self.n_w = 3 if head == "block" else 1
        super().__init__(in_dims, hidden, n_out=3 * (self.n_w + 3))

    def step(self, x, state, valid, gate_mode: str = "soft"):
        flat = x[:, self.in_mask]
        h = self.gru(torch.nn.functional.elu(self.inp(flat)), state["h"])
        s = self.out(h).view(-1, 3, self.n_w + 3)
        return s.masked_fill(~valid, -1e9), {"h": h}, {}


def split(out: torch.Tensor, n_w: int):
    """Head output [B,3,n_w+3] -> (weight logits [B,3] or [B,3,3], inflation u [B,3,3]).  Ineligible
    rows were masked with -1e9 by the model; u is reset to 0 there (those solutions are excluded)."""
    s = out[..., :n_w]
    u = out[..., n_w:]
    u = torch.where(u < -1e8, torch.zeros_like(u), u)
    return (s.squeeze(-1) if n_w == 1 else s), u


def masked_weights(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over the eligible solutions (dim 1); all-zero where none is eligible.
    logits [B,3] or [B,3,3]; valid [B,3] -> w of the same shape as logits."""
    v = valid if logits.dim() == 2 else valid[..., None].expand_as(logits)
    w = torch.softmax(logits.masked_fill(~v, -1e9), 1)
    return w * v.any(1, keepdim=True)
