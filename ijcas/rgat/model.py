"""Recurrent relational GNN selector: one cell per tick (design section 3).

    memory update  u_i = GRU_kind(i)(x_i, h_i)                    (kind-shared weights, per-node state)
    edge gate      l_e = (U_r u_j)^T (V_r u_i) / sqrt(d_g) + b_r    for every gated candidate edge
                   z_e ~ HardConcrete(l_e) (train) | hysteresis(P(z_e != 0)) in {0, 1} (operation)
    aggregation    L layers of gated R-GAT (or gated R-GCN) over the active edges + self loops
    memory store   h_i <- tanh(v_i^(L))     (bounded, as a GRU hidden state)
    scores         s_k = MLP(v_nav_k) + c_k  -> masked softmax over the eligible candidates

Inputs are packed per node as [y_norm, m, dt_log] padded to a common width; each node has its own
input adapter because the nodes of one kind expose different diagnostics (a shared map over padded
heterogeneous slots would give one weight several meanings).  The GRU, gate, aggregation and score
weights are shared as stated in the design.

Variants (same code): agg='gat'|'gcn'; gated=False (all candidate edges always on); memory=False
(h reset to 0 every tick).  ``FlatGru`` is the non-graph baseline with the same inputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import CANDIDATES, V1, GraphSpec, edge_index

HC_BETA, HC_GAMMA, HC_ZETA = 2.0 / 3.0, -0.1, 1.1          # Louizos et al. 2018 defaults
HC_SHIFT = HC_BETA * math.log(-HC_GAMMA / HC_ZETA)          # P(z != 0) = sigmoid(l - shift)


def p_active(logit):
    return torch.sigmoid(logit - HC_SHIFT)


class RecurrentRelGNN(nn.Module):
    def __init__(self, in_dims: list[int], edges, d: int = 32, d_g: int = 16, layers: int = 2,
                 heads: int = 2, agg: str = "gat", gated: bool = True, memory: bool = True,
                 th_on: float = 0.6, th_off: float = 0.4, spec: GraphSpec | None = None):
        super().__init__()
        self.spec = spec = spec or V1
        self.N, self.d, self.d_g, self.L, self.H = len(spec.nodes), d, d_g, layers, heads
        self.agg, self.gated, self.memory = agg, gated, memory
        self.th_on, self.th_off = th_on, th_off
        self.in_dims = list(in_dims)
        fmax = max(in_dims)
        self.fmax = fmax
        # node-specific input adapters (packed as one padded tensor; padded inputs are masked)
        self.w_in = nn.Parameter(torch.randn(self.N, fmax, d) / math.sqrt(fmax))
        self.b_in = nn.Parameter(torch.zeros(self.N, d))
        mask = torch.zeros(self.N, fmax)
        for i, f in enumerate(in_dims):
            mask[i, :f] = 1
        self.register_buffer("in_mask", mask)
        kind = torch.tensor([spec.kinds.index(spec.node_kind[n]) for n in spec.nodes])
        self.register_buffer("kind", kind)
        nk = len(spec.kinds)
        self.gru_wi = nn.Parameter(torch.randn(nk, 3 * d, d) / math.sqrt(d))
        self.gru_wh = nn.Parameter(torch.randn(nk, 3 * d, d) / math.sqrt(d))
        self.gru_bi = nn.Parameter(torch.zeros(nk, 3 * d))
        self.gru_bh = nn.Parameter(torch.zeros(nk, 3 * d))
        self.set_edges(edges)
        R = len(spec.relations)
        self.U = nn.Parameter(torch.randn(R, d_g, d) / math.sqrt(d))
        self.V = nn.Parameter(torch.randn(R, d_g, d) / math.sqrt(d))
        self.gate_b = nn.Parameter(torch.full((R,), 2.0))    # start with edges on (P ~ 0.97)
        self.W = nn.Parameter(torch.randn(layers, R, d, d) / math.sqrt(d))
        dh = d // heads
        self.att_src = nn.Parameter(torch.randn(layers, R, heads, dh) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(layers, R, heads, dh) * 0.1)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, 1))
        self.cand_bias = nn.Parameter(torch.zeros(len(spec.candidates)))
        self.register_buffer("nav_idx", torch.tensor([spec.nodes.index(n) for n in spec.out_nodes]))

    # -- structure ----------------------------------------------------------------------
    def set_edges(self, edges):
        """Candidate structure (E_full in training, E_* after pruning).  Removed edges are not
        computed at all: they are absent from the index tensors."""
        self.edges = list(edges)
        ei = edge_index(self.edges, self.spec)
        dev = self.gru_wi.device
        self.register_buffer("e_src", torch.tensor(ei["src"], device=dev), persistent=False)
        self.register_buffer("e_dst", torch.tensor(ei["dst"], device=dev), persistent=False)
        self.register_buffer("e_rel", torch.tensor(ei["rel"], device=dev), persistent=False)
        gated = torch.tensor(ei["gated"], device=dev)
        self.register_buffer("e_gated", gated, persistent=False)
        self.register_buffer("g_idx", torch.nonzero(gated).flatten(), persistent=False)
        E = len(self.edges)
        dst1h = torch.zeros(self.N, E, device=dev)
        dst1h[torch.tensor(ei["dst"], device=dev), torch.arange(E, device=dev)] = 1.0
        self.register_buffer("dst1h", dst1h, persistent=False)
        R = len(self.spec.relations)
        nr = torch.zeros(self.N * R, E, device=dev)              # (dst, rel) groups for R-GCN
        nr[torch.tensor(ei["dst"] * R + ei["rel"], device=dev), torch.arange(E, device=dev)] = 1.0
        self.register_buffer("dstrel1h", nr, persistent=False)
        self.edge_names = ei["names"]

    @property
    def gated_names(self):
        return [self.edge_names[i] for i in self.g_idx.tolist()]

    def init_state(self, B: int, device=None):
        device = device or self.w_in.device
        return {"h": torch.zeros(B, self.N, self.d, device=device),
                "z": torch.ones(B, len(self.g_idx), device=device)}   # gate hysteresis state

    # -- one tick -------------------------------------------------------------------------
    def step(self, x, state, valid, gate_mode: str = "soft"):
        """x [B, N, fmax] packed inputs, valid [B, K] bool.  gate_mode: soft (Hard Concrete
        sample), hard_st (binary with hysteresis, straight-through gradient), eval (binary with
        hysteresis, deterministic).  Returns logits [B, K], new state, aux dict."""
        B = x.shape[0]
        h = state["h"] if self.memory else torch.zeros_like(state["h"])
        xin = torch.einsum("bnf,nfd->bnd", x * self.in_mask, self.w_in) + self.b_in
        wi, wh = self.gru_wi[self.kind], self.gru_wh[self.kind]
        gi = torch.einsum("bnd,nkd->bnk", xin, wi) + self.gru_bi[self.kind]
        gh = torch.einsum("bnd,nkd->bnk", h, wh) + self.gru_bh[self.kind]
        ir, iz, inn = gi.chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r, zg = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        u = (1 - zg) * torch.tanh(inn + r * hn) + zg * h
        # edge gates on the gated candidate edges
        E = len(self.edges)
        z_all = torch.ones(B, E, device=x.device)
        aux = {}
        new_z = state["z"]
        if self.gated and len(self.g_idx):
            gs, gd, gr = self.e_src[self.g_idx], self.e_dst[self.g_idx], self.e_rel[self.g_idx]
            us = torch.einsum("bed,egd->beg", u[:, gs], self.U[gr])
            ud = torch.einsum("bed,egd->beg", u[:, gd], self.V[gr])
            logit = (us * ud).sum(-1) / math.sqrt(self.d_g) + self.gate_b[gr]
            p = p_active(logit)
            aux["p_active"] = p
            if gate_mode == "soft":
                eps = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
                s = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + logit) / HC_BETA)
                z = (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0, 1)
            else:
                prev = state["z"]
                hard = torch.where(p >= self.th_on, torch.ones_like(p),
                                   torch.where(p <= self.th_off, torch.zeros_like(p), prev))
                new_z = hard.detach()
                z = hard + (p - p.detach()) if gate_mode == "hard_st" else hard
            z_all = z_all.index_copy(1, self.g_idx, z)
            aux["z"] = z
        # aggregation over active edges.  Operation (eval, one vehicle): only the active edges are
        # gathered and computed, so inactive edges cost nothing beyond their gate score; batched
        # training/evaluation masks them instead (same result, z = 0 removes the term exactly).
        src, dst, rel, d1h, dr1h, za = (self.e_src, self.e_dst, self.e_rel, self.dst1h,
                                        self.dstrel1h, z_all)
        sparse = gate_mode == "eval" and B == 1 and self.gated
        if sparse:
            act = torch.nonzero(z_all[0] > 0).flatten()
            src, dst, rel, d1h, dr1h, za = (src[act], dst[act], rel[act], d1h[:, act],
                                            dr1h[:, act], z_all[:, act])
            aux["active_idx"] = act
        Ea = len(src)
        v = u
        R = len(self.spec.relations)
        for l in range(self.L):
            Wl = self.W[l]
            msg = torch.einsum("bed,ekd->bek", v[:, src], Wl[rel])               # [B,Ea,d]
            if self.agg == "gat":
                dh = self.d // self.H
                ms = msg.view(B, Ea, self.H, dh)
                md = torch.einsum("bed,ekd->bek", v[:, dst], Wl[rel]).view(B, Ea, self.H, dh)
                a = F.leaky_relu((ms * self.att_src[l, rel]).sum(-1)
                                 + (md * self.att_dst[l, rel]).sum(-1), 0.2)    # [B,Ea,H]
                # z-weighted softmax over the incoming edges of each destination:
                #   omega_e = z_e exp(a_e) / sum z exp(a).  The shift is the max of a + log z over
                #   the *active* edges (the self loop always is one), so the denominator is >= 1;
                #   shifting by inactive edges let every active term underflow (NaN in training).
                on = za > 0                                                     # [B,Ea]
                s_ = a + torch.log(za.clamp_min(1e-30))[..., None]
                big = torch.where((d1h.bool()[None, :, :] & on[:, None, :])[..., None],
                                  s_[:, None], torch.full_like(s_[:, None], -1e30))
                shift = big.max(2).values.detach()[:, dst]                      # [B,Ea,H]
                w = torch.where(on[..., None], torch.exp((s_ - shift).clamp(max=0.0)),
                                za[..., None] * torch.exp((a - shift).clamp(max=0.0)))
                den = torch.einsum("ne,beh->bnh", d1h, w)
                om = w / den[:, dst]
                if l == self.L - 1:
                    aux["attn"] = om.mean(-1)                                  # [B,Ea]
                agg = torch.einsum("ne,behk->bnhk", d1h, om[..., None] * ms).reshape(B, self.N, self.d)
            else:
                # gated mean per (destination, relation); the count is floored at 1 so a group
                # whose gates are (nearly) closed fades out instead of dividing by ~0
                cnt = torch.einsum("me,be->bm", dr1h, za)                       # [B,N*R]
                norm = za / cnt[:, dst * R + rel].clamp_min(1.0)
                agg = torch.einsum("ne,bed->bnd", d1h, norm[..., None] * msg)
                if l == self.L - 1:
                    aux["attn"] = norm
            v = v + torch.tanh(agg)
        h_new = torch.tanh(v)
        logits = self.scores(v).masked_fill(~valid, -1e9)
        return logits, {"h": h_new, "z": new_z}, aux

    def scores(self, v):
        """Per-candidate logits from the final node embeddings v [B, N, d] (overridden by the fusion
        weight head, ijcas/fusion/weights.py)."""
        return self.head(v[:, self.nav_idx]).squeeze(-1) + self.cand_bias


class FlatGru(nn.Module):
    """Non-graph baseline: the same packed inputs concatenated -> GRU -> per-candidate scores."""

    def __init__(self, in_dims: list[int], hidden: int = 96, n_out: int | None = None):
        super().__init__()
        self.in_dims = list(in_dims)
        self.fmax = max(in_dims)
        mask = torch.zeros(len(in_dims), self.fmax, dtype=torch.bool)
        for i, f in enumerate(in_dims):
            mask[i, :f] = True
        self.register_buffer("in_mask", mask)
        self.inp = nn.Linear(sum(in_dims), hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_out or len(CANDIDATES)))
        self.hidden = hidden
        self.memory = True

    def init_state(self, B: int, device=None):
        return {"h": torch.zeros(B, self.hidden, device=device or self.inp.weight.device)}

    def step(self, x, state, valid, gate_mode: str = "soft"):
        flat = x[:, self.in_mask]                                              # [B, sum F]
        h = self.gru(F.elu(self.inp(flat)), state["h"])
        logits = self.out(h).masked_fill(~valid, -1e9)
        return logits, {"h": h}, {}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
