"""Training, inference, discrete selection and metrics for the graph selector (design 5-9).

Loss  L = L_nav + lam_E L_edge + lam_S L_switch
  L_nav    = mean over ticks with >= 1 eligible candidate of  sum_k pi_k (c_k - min_j c_j)
             (regret form: same gradient as sum_k pi_k c_k because sum_k pi_k = 1)
  c_k      = rho(|dp|/s_p) + b_v rho(|dv|/s_v) + b_R rho(dR/s_R), rho(x) = x^2 for x <= 3 and
             3 (2x - 3) above (Huber: the design's squared error inside 3 scale units, linear
             beyond, so a 50 m divergence still ranks worse than 15 m but does not own the
             gradient).  From recorded candidate outputs and ground truth; no gradient enters the
             engines.
  L_edge   = mean P(z_e != 0) over gated edges and ticks (expected active-edge fraction)
  L_switch = 0.5 |pi_t - pi_{t-1}|_1 on ticks whose eligible set did not change (a candidate that
             becomes ineligible is never held back by this term)
Truncated BPTT over whole runs: the memory is carried across chunks (detached), as in operation.

Discrete selection (policies, not learned): ineligible current -> immediate switch to the best
eligible score (confirmed candidates first); otherwise switch only after the minimum dwell, to a
candidate eligible for >= confirm_s, whose probability exceeds the current one by > margin.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .features import SPEC
from .graph import CANDIDATES, NODES

# Per-term robust cost: quadratic up to delta, then linear (Huber).  A logarithmic tail ("tail": "log")
# was tried on 2026-11-17 and did not change the fixed-priority plateau, so the registered Huber stays.
COST = {"s_p": 1.0, "s_v": 0.5, "s_R": 5.0, "beta_v": 0.5, "beta_R": 0.5, "huber": 3.0, "tail": "linear"}
# Auxiliary cost-sensitive soft-target term (2026-11-17): KL(q || pi), q_k ~ exp(-(c_k - c_min)/tau) over the
# eligible candidates.  The registered expected-cost term sum_k pi_k (c_k - c_min) has gradient
# pi_k (c_k - c_bar), which vanishes once pi saturates on one candidate; every variant could stall in the
# eligibility-only fixed priority (gru_flat 6/6 trainings, rgcn_dyn 1/3).  Gradient of this term: pi - q.
AUX = {"tau": 1.0, "lam_ce": 1.0}
ALARM_M = 2.0
EVAL_REGIMES = ("normal", "single", "compound", "recovery")
REGIMES = ("normal", "single", "compound", "recovery", "warmup")
DT_MAX = 600.0


def masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over the eligible candidates only (masked in logit space, so the result never
    underflows when an ineligible candidate has the largest logit); all-zero rows when none is eligible."""
    v = valid.bool()
    pi = torch.softmax(logits.masked_fill(~v, -1e9), -1)
    return pi * v.any(-1, keepdim=True)


# ------------------------------------------------------------------------------- data
def load_run(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def costs(run: dict, cost=COST) -> tuple[np.ndarray, np.ndarray]:
    ep, ev, ea = run["err_pos"], run["err_vel"], run["err_att"]
    d = cost["huber"]

    def rho(x):
        if cost.get("tail", "linear") == "log":
            return np.where(x <= d, x ** 2, d * d * (1.0 + 2.0 * np.log(np.maximum(x, d) / d)))
        return np.where(x <= d, x ** 2, d * (2 * x - d))
    c = (rho(ep / cost["s_p"]) + cost["beta_v"] * rho(ev / cost["s_v"])
         + cost["beta_R"] * rho(ea / cost["s_R"]))
    valid = run["valid"] & np.isfinite(c)
    return np.where(valid, c, 0.0).astype(np.float32), valid


class Normalizer:
    """Per node / feature mean and std of present values over the training runs only."""

    def __init__(self, stats: dict | None = None):
        self.stats = stats or {}

    def fit(self, runs):
        for n in NODES:
            y = np.concatenate([r[f"y_{n}"] for r in runs])
            mu = np.nanmean(np.where(np.isfinite(y), y, np.nan), 0)
            sd = np.nanstd(np.where(np.isfinite(y), y, np.nan), 0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
            self.stats[n] = (mu.tolist(), sd.tolist())
        return self

    @property
    def in_dims(self):
        return [3 * len(SPEC[n]) for n in NODES]

    def pack(self, run) -> np.ndarray:
        T = len(run["t"])
        fmax = max(self.in_dims)
        X = np.zeros((T, len(NODES), fmax), np.float32)
        for i, n in enumerate(NODES):
            mu, sd = (np.asarray(a, np.float32) for a in self.stats[n])
            y = run[f"y_{n}"]
            yn = np.where(np.isfinite(y), np.clip((y - mu) / sd, -10, 10), 0.0)
            dtl = np.log1p(np.clip(run[f"dt_{n}"], 0, DT_MAX)) / math.log1p(DT_MAX)
            F = y.shape[1]
            X[:, i, :3 * F] = np.concatenate([yn, run[f"m_{n}"], dtl], 1)
        return X


def stack(runs, norm: Normalizer):
    """Pad runs to a common length -> X [B,T,N,F], V [B,T,K] bool, C [B,T,K]."""
    T = max(len(r["t"]) for r in runs)
    B, K = len(runs), len(CANDIDATES)
    fmax = max(norm.in_dims)
    X = np.zeros((B, T, len(NODES), fmax), np.float32)
    V = np.zeros((B, T, K), bool)
    C = np.zeros((B, T, K), np.float32)
    for b, r in enumerate(runs):
        n = len(r["t"])
        X[b, :n] = norm.pack(r)
        C[b, :n], V[b, :n] = costs(r)
    return torch.from_numpy(X), torch.from_numpy(V), torch.from_numpy(C)


# ------------------------------------------------------------------------------- training
@dataclass
class Phase:
    epochs: int
    gate_mode: str = "soft"          # soft | hard_st | none
    lam_E: tuple = (0.0, 0.0)        # (start, end) linear over the phase (after warmup_frac)
    warmup_frac: float = 0.0
    lam_S: float = 0.1
    lr: float = 2e-3


def fit(model, X, V, C, phases: list[Phase], device="cuda", chunk: int = 200,
        batch: int | None = None, seed: int = 0, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device).train()
    B, T = X.shape[:2]
    batch = batch or B
    hist = []
    for ph in phases:
        opt = torch.optim.Adam(model.parameters(), lr=ph.lr)
        for ep in range(ph.epochs):
            frac = ep / max(ph.epochs - 1, 1)
            if frac < ph.warmup_frac:
                lam_E = 0.0
            else:
                w = (frac - ph.warmup_frac) / max(1 - ph.warmup_frac, 1e-9)
                lam_E = ph.lam_E[0] + (ph.lam_E[1] - ph.lam_E[0]) * w
            perm = torch.randperm(B)
            t0 = time.time()
            tot = {"nav": 0.0, "edge": 0.0, "switch": 0.0, "n": 0}
            for bs in range(0, B, batch):
                idx = perm[bs:bs + batch]
                st = model.init_state(len(idx), device)
                prev_pi, prev_v = None, None
                for s in range(0, T, chunk):
                    xb = X[idx, s:s + chunk].to(device, non_blocking=True)
                    vb = V[idx, s:s + chunk].to(device)
                    cb = C[idx, s:s + chunk].to(device)
                    logits, pacts = [], []
                    for k in range(xb.shape[1]):
                        lg, st, aux = model.step(xb[:, k], st, vb[:, k], ph.gate_mode)
                        logits.append(lg)
                        if "p_active" in aux:
                            pacts.append(aux["p_active"])
                    lg = torch.stack(logits, 1)
                    anyv = vb.any(-1)
                    pi = masked_softmax(lg, vb)
                    cmin = torch.where(vb, cb, torch.full_like(cb, 1e9)).min(-1, keepdim=True).values
                    regret = ((cb - cmin).clamp_min(0) * pi).sum(-1)
                    l_nav = (regret * anyv).sum() / anyv.sum().clamp_min(1)
                    q = masked_softmax(-(cb - cmin).clamp_min(0) / AUX["tau"], vb).detach()
                    logp = torch.log_softmax(lg.masked_fill(~vb.bool(), -1e9), -1)
                    kl = (torch.special.xlogy(q, q) - q * logp.masked_fill(~vb.bool(), 0.0)).sum(-1)
                    l_ce = (kl * anyv).sum() / anyv.sum().clamp_min(1)
                    l_edge = torch.stack(pacts, 1).mean() if pacts else torch.zeros((), device=device)
                    pis = torch.cat([prev_pi, pi], 1) if prev_pi is not None else pi
                    vs = torch.cat([prev_v, vb], 1) if prev_v is not None else vb
                    same = (vs[:, 1:] == vs[:, :-1]).all(-1) & vs[:, 1:].any(-1)
                    dpi = 0.5 * (pis[:, 1:] - pis[:, :-1]).abs().sum(-1)
                    l_sw = (dpi * same).sum() / same.sum().clamp_min(1)
                    loss = l_nav + AUX["lam_ce"] * l_ce + lam_E * l_edge + ph.lam_S * l_sw
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(loss) and torch.isfinite(gn):
                        opt.step()
                    else:                                    # never let one bad chunk poison weights
                        tot["skipped"] = tot.get("skipped", 0) + 1
                        bad = [n for n, p in model.named_parameters()
                               if p.grad is not None and not torch.isfinite(p.grad).all()]
                        log(json.dumps({"skipped_update": {"chunk_start": s, "phase": ph.gate_mode,
                                        "loss_finite": bool(torch.isfinite(loss)),
                                        "terms": [float(l_nav), float(l_ce), float(l_edge), float(l_sw)],
                                        "nonfinite_grads": bad[:8]}}))
                    st = {k: v.detach() for k, v in st.items()}
                    prev_pi, prev_v = pi[:, -1:].detach(), vb[:, -1:]
                    tot["nav"] += l_nav.item()
                    tot["ce"] = tot.get("ce", 0.0) + l_ce.item()
                    tot["edge"] += l_edge.item()
                    tot["switch"] += l_sw.item()
                    tot["n"] += 1
            rec = {k: v / max(tot["n"], 1) for k, v in tot.items() if k not in ("n", "skipped")}
            rec["skipped_updates"] = tot.get("skipped", 0)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite parameters after epoch")
            rec.update(epoch=len(hist), gate_mode=ph.gate_mode, lam_E=lam_E, sec=time.time() - t0)
            hist.append(rec)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
    return hist


@torch.no_grad()
def infer(model, X, V, device="cuda", gate_mode="eval", keep_gates=True):
    """Whole runs, operation mode (binary gates with hysteresis).  Returns numpy arrays."""
    model.to(device).eval()
    B, T = X.shape[:2]
    st = model.init_state(B, device)
    P, Z, A = [], [], []
    for s in range(0, T, 400):
        xb = X[:, s:s + 400].to(device)
        vb = V[:, s:s + 400].to(device)
        for k in range(xb.shape[1]):
            lg, st, aux = model.step(xb[:, k], st, vb[:, k], gate_mode)
            P.append(masked_softmax(lg, vb[:, k]).cpu())
            if keep_gates and "z" in aux:
                Z.append(aux["z"].cpu())
            if keep_gates and "attn" in aux:
                A.append(aux["attn"].cpu())
    out = {"pi": torch.stack(P, 1).numpy()}
    if Z:
        out["z"] = torch.stack(Z, 1).numpy()
    if A:
        out["attn"] = torch.stack(A, 1).numpy()
    return out


# ------------------------------------------------------------------------------- selection
@dataclass
class SelectCfg:
    margin: float = 0.1
    dwell_s: float = 2.0
    confirm_s: float = 1.0


def select(pi: np.ndarray, valid: np.ndarray, t: np.ndarray, cfg: SelectCfg, hold=None):
    """-> sel [T] (-1 = no valid navigation), events [(t, from, to, reason)].
    hold [T] bool (optional, retrieval selector): no learned switch at that tick; policy switches
    (start, current ineligible, none eligible) are unaffected."""
    T, K = pi.shape
    sel = np.full(T, -1, int)
    cur, since = -1, -np.inf
    vsince = np.full(K, np.nan)
    events = []
    for i in range(T):
        v = valid[i]
        vsince = np.where(v, np.where(np.isnan(vsince), t[i], vsince), np.nan)
        if not v.any():
            if cur != -1:
                events.append((float(t[i]), cur, -1, "policy:no_eligible"))
            cur = -1
            continue
        conf = v & ((t[i] - np.nan_to_num(vsince, nan=np.inf) >= cfg.confirm_s - 1e-9)
                    | (np.arange(K) == cur))
        if cur == -1 or not v[cur]:
            pool = conf if conf.any() else v
            new = int(np.argmax(np.where(pool, pi[i], -1.0)))
            events.append((float(t[i]), cur, new, "policy:current_ineligible" if cur != -1
                           else "policy:start"))
            cur, since = new, t[i]
        elif t[i] - since >= cfg.dwell_s - 1e-9 and conf.any() and not (hold is not None and hold[i]):
            best = int(np.argmax(np.where(conf, pi[i], -1.0)))
            if best != cur and pi[i, best] - pi[i, cur] > cfg.margin:
                events.append((float(t[i]), cur, best, "learned:score_margin"))
                cur, since = best, t[i]
        sel[i] = cur
    return sel, events


# ------------------------------------------------------------------------------- metrics
def _yaw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trace_metrics(t, sel, ep, ev, ea, pos, vel, quat, mode, regime, best_ep, faults, normal_s,
                  limited_label=3) -> dict:
    """Generic metrics of one output trace (any selector, any rate).
    sel: candidate label per tick (-1 none); mode: 0 normal 1 limited 2 none."""
    dt = float(np.median(np.diff(t)))
    m = t >= normal_s
    has = (sel >= 0) & np.isfinite(ep)
    e = ep[m & has]
    out = {"ticks": int(m.sum()), "dt": dt,
           "sum_sq_p": float((e ** 2).sum()), "sum_sq_v": float((ev[m & has] ** 2).sum()),
           "sum_sq_att": float(np.nansum(ea[m & has] ** 2)), "n_out": int(len(e)),
           "pos_rmse_m": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
           "vel_rmse_mps": float(np.sqrt(np.mean(ev[m & has] ** 2))) if len(e) else None,
           "att_rmse_deg": float(np.sqrt(np.nanmean(ea[m & has] ** 2))) if len(e) else None,
           "pos_p95_m": float(np.percentile(e, 95)) if len(e) else None,
           "output_fraction": float(has[m].mean()),
           "availability": float((has & (ep <= ALARM_M))[m].mean()),
           "false_normal_s": float((has & (mode == 0) & (ep > ALARM_M))[m].sum() * dt),
           "limited_s": float((mode == 1)[m].sum() * dt),
           "no_valid_nav_s": float((~has)[m].sum() * dt)}
    sw = [i for i in range(1, len(t)) if m[i] and sel[i] != sel[i - 1]]
    pj, vj, yj = [], [], []
    for i in sw:
        if has[i] and has[i - 1]:
            pj.append(float(np.linalg.norm(pos[i] - pos[i - 1] - vel[i - 1] * (t[i] - t[i - 1]))))
            vj.append(float(np.linalg.norm(vel[i] - vel[i - 1])))
            yj.append(float(abs(np.degrees(np.angle(np.exp(1j * (_yaw(quat[i]) - _yaw(quat[i - 1]))))))))
    out.update(switches=len(sw), max_pos_jump_m=max(pj) if pj else 0.0,
               median_pos_jump_m=float(np.median(pj)) if pj else 0.0,
               max_vel_jump_mps=max(vj) if vj else 0.0, max_yaw_jump_deg=max(yj) if yj else 0.0,
               share={c: float(np.mean(sel[m] == k)) for k, c in enumerate(CANDIDATES)})
    out["regimes"] = {}
    for r, name in enumerate(REGIMES[:4]):
        mm = m & (regime == r)
        if mm.any():
            er = ep[mm & has]
            out["regimes"][name] = {"ticks": int(mm.sum()), "sum_sq_p": float((er ** 2).sum()),
                                    "n_out": int(len(er)),
                                    "avail_ticks": int((has & (ep <= ALARM_M))[mm].sum())}
    # fault response (vs the best eligible candidate at the same tick)
    # bad: an eligible candidate within the alarm limit existed but no output was given, or the
    # output is > 1 m worse than the best eligible candidate
    bad = m & ((~has & (best_ep <= ALARM_M)) | (has & (ep - best_ep > 1.0)))
    resp, rec = [], []
    for f in faults:
        a, b = f["start_s"], f["end_s"]
        idx = np.where((t >= a) & (t < b) & bad)[0]
        if len(idx):
            j = idx[0]
            k = j
            while k < len(t) and bad[k]:
                k += 1
            resp.append(float(t[min(k, len(t) - 1)] - t[j]))
        idx = np.where(t >= b)[0]
        if len(idx):
            k = idx[0]
            while k < len(t) and bad[k]:
                k += 1
            rec.append(float(t[min(k, len(t) - 1)] - t[idx[0]]))
    out["response_s"] = resp
    out["recovery_s"] = rec
    return out


def candidate_trace(run, sel):
    """Arrays of the selected candidate's state for a 20 Hz selection."""
    T = len(run["t"])
    i = np.arange(T)
    k = np.clip(sel, 0, None)
    on = sel >= 0
    ep, ev, ea = (np.where(on, run[x][i, k], np.nan) for x in ("err_pos", "err_vel", "err_att"))
    pos, vel, quat = (np.where(on[:, None], run[x][i, k], np.nan)
                      for x in ("cand_pos", "cand_vel", "cand_quat"))
    mode = np.where(sel < 0, 2, np.where(sel == CANDIDATES.index("ins"), 1, 0))
    return ep, ev, ea, pos, vel, quat, mode


def best_err(run):
    ep = np.where(run["valid"], run["err_pos"], np.inf)
    b = ep.min(1)
    return np.where(np.isfinite(b), b, np.nan)


def oracle_sel(run):
    c, v = costs(run)
    s = np.where(v, c, np.inf).argmin(1)
    return np.where(v.any(1), s, -1)


def metrics_for_selection(run, sel):
    ep, ev, ea, pos, vel, quat, mode = candidate_trace(run, sel)
    m = run["meta"]
    return trace_metrics(run["t"], sel, ep, ev, ea, pos, vel, quat, mode, run["regime"],
                         best_err(run), m["faults"], m["normal_s"])


def metrics_rule(run, key):
    """Ontology-K ('ont') / H-FSM ('hfsm') at their own 10 Hz, compared with the best candidate
    of the 20 Hz bank at the same instants."""
    t = run["rule_t"]
    idx = np.clip(np.searchsorted(run["t"], t - 1e-6), 0, len(run["t"]) - 1)
    sel = run[f"rule_{key}_sel"]
    mode = run[f"rule_{key}_mode"]
    sel = np.where(mode == 2, -1, sel)
    m = run["meta"]
    return trace_metrics(t, sel, run[f"rule_{key}_err_pos"], run[f"rule_{key}_err_vel"],
                         run[f"rule_{key}_err_att"], run[f"rule_{key}_pos"], run[f"rule_{key}_vel"],
                         run[f"rule_{key}_quat"], mode, run["rule_regime"], best_err(run)[idx],
                         m["faults"], m["normal_s"])


def pool(per_run: list[dict]) -> dict:
    """Pooled metrics over runs (RMSE over all output ticks of all runs)."""
    s = lambda k: sum(r[k] for r in per_run)  # noqa
    n_out, ticks = s("n_out"), s("ticks")
    out = {"runs": len(per_run),
           "pos_rmse_m": math.sqrt(s("sum_sq_p") / n_out) if n_out else None,
           "vel_rmse_mps": math.sqrt(s("sum_sq_v") / n_out) if n_out else None,
           "att_rmse_deg": math.sqrt(s("sum_sq_att") / n_out) if n_out else None,
           "output_fraction": n_out / ticks if ticks else None,
           "availability": float(np.average([r["availability"] for r in per_run],
                                             weights=[r["ticks"] for r in per_run])),
           "false_normal_s_per_run": s("false_normal_s") / len(per_run),
           "no_valid_nav_s_per_run": s("no_valid_nav_s") / len(per_run),
           "limited_s_per_run": s("limited_s") / len(per_run),
           "switches_per_run": s("switches") / len(per_run),
           "max_pos_jump_m": max(r["max_pos_jump_m"] for r in per_run),
           "max_vel_jump_mps": max(r["max_vel_jump_mps"] for r in per_run),
           "max_yaw_jump_deg": max(r["max_yaw_jump_deg"] for r in per_run)}
    resp = [x for r in per_run for x in r["response_s"]]
    rec = [x for r in per_run for x in r["recovery_s"]]
    out.update(response_mean_s=float(np.mean(resp)) if resp else 0.0,
               response_max_s=float(np.max(resp)) if resp else 0.0,
               recovery_mean_s=float(np.mean(rec)) if rec else 0.0,
               recovery_max_s=float(np.max(rec)) if rec else 0.0)
    reg = {}
    for name in EVAL_REGIMES:
        rows = [r["regimes"][name] for r in per_run if name in r["regimes"]]
        if rows:
            n = sum(x["n_out"] for x in rows)
            reg[name] = {"pos_rmse_m": math.sqrt(sum(x["sum_sq_p"] for x in rows) / n) if n else None,
                         "availability": sum(x["avail_ticks"] for x in rows) / sum(x["ticks"] for x in rows)}
    out["regimes"] = reg
    return out
