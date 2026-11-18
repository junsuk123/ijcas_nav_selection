"""Differentiable weighted covariance intersection (CI) of navigation solutions.

State of solution k: x_k = (p_k, v_k, R_k), error state [dp, dv, dtheta] with a right (body) attitude
perturbation.  Fusion is done per block b in (position, velocity, attitude):

    Lambda_f,b = sum_k w_k,b P_k,b^-1          (w_k,b = w_k renormalised over the solutions that
    P_f,b      = Lambda_f,b^-1                   reported a covariance for block b)
    dx_f,b     = P_f,b sum_k w_k,b P_k,b^-1 dx_k,b

CI is used because all three estimators share the IMU (ontology :sharesMeasurementSource), so their
errors are correlated by an unknown amount; adding the information matrices as if independent would
double count the common part.  With weights on the simplex, CI is consistent for any cross-correlation
*provided the input covariances are consistent* - which this module does not establish.

Blocks are fused separately: PX4 reports no cross-block terms and position/attitude and velocity come in
different messages, so cross-block covariances are not used (they are not assumed to be zero inside a
joint 9x9 inversion either).  Attitude: reference R_ref (largest weight), dtheta_k = Log(R_ref^T R_k),
each attitude covariance is moved to the reference tangent space with J_r^-1(dtheta_k), and the
reference is re-linearised ``iters`` times.  Inverses go through closed-form 3x3 Cholesky factors
(_chol3); no general matrix inverse is used.  Blocks without any reporting solution are returned as NaN (no substitute).
"""
from __future__ import annotations

import torch

from . import geometry as G

BLOCKS = ("pos", "vel", "att")


def _chol3(A):
    """Cholesky factor of SPD 3x3 matrices [...,3,3] in closed form (elementwise ops; cheaper than a
    batched LAPACK/cuSOLVER call for tiny matrices).  Returns the six entries of L."""
    l11 = A[..., 0, 0].clamp_min(1e-300).sqrt()
    l21 = A[..., 1, 0] / l11
    l31 = A[..., 2, 0] / l11
    l22 = (A[..., 1, 1] - l21 * l21).clamp_min(1e-300).sqrt()
    l32 = (A[..., 2, 1] - l31 * l21) / l22
    l33 = (A[..., 2, 2] - l31 * l31 - l32 * l32).clamp_min(1e-300).sqrt()
    return l11, l21, l31, l22, l32, l33


def _chol3_inv(L):
    """A^-1 = L^-T L^-1 from the closed-form factor."""
    l11, l21, l31, l22, l32, l33 = L
    m11, m22, m33 = 1 / l11, 1 / l22, 1 / l33                    # M = L^-1 (lower triangular)
    m21 = -l21 * m11 * m22
    m32 = -l32 * m22 * m33
    m31 = -(l31 * m11 + l32 * m21) * m33
    z = torch.zeros_like(m11)
    M = torch.stack([torch.stack([m11, z, z], -1), torch.stack([m21, m22, z], -1),
                     torch.stack([m31, m32, m33], -1)], -2)
    return M.transpose(-1, -2) @ M


def _block_ci(dx, P, w, part, independent=False):
    """dx [B,K,3], P [B,K,3,3], w [B,K] (sums to 1 over the valid set), part [B,K] bool.
    independent=True: comparison baseline Lambda_f = sum_k P_k^-1 over the participants (no weights,
    treats the solutions as independent).  -> dx_f [B,3], P_f [B,3,3], ok [B]."""
    wb = torch.where(part, w, torch.zeros_like(w))
    s = wb.sum(1, keepdim=True)                                            # over the solutions (dim 1)
    ok = s.squeeze(1) > 0
    wb = part.to(w.dtype) if independent else wb / s.clamp_min(1e-12)
    eye = torch.eye(3, dtype=P.dtype, device=P.device)
    Ps = torch.where(part[..., None, None], P, eye.expand_as(P))          # dummy for absent blocks
    Lam = _chol3_inv(_chol3(Ps))                                           # P_k^-1 via Cholesky
    Lf = (wb[..., None, None] * Lam).sum(1)
    Lf = torch.where(ok[..., None, None], Lf, eye.expand_as(Lf))
    rhs = (wb[..., None] * (Lam @ dx[..., None]).squeeze(-1)).sum(1)
    Pf = _chol3_inv(_chol3(Lf))                                            # P_f = Lambda_f^-1 via Cholesky
    dxf = (Pf @ rhs[..., None]).squeeze(-1)
    nan = torch.full_like(dxf, float("nan"))
    return torch.where(ok[..., None], dxf, nan), torch.where(ok[..., None, None], Pf, nan[..., None].expand_as(Pf)), ok


def fuse(pos, vel, quat, P, avail, w, iters: int = 2, independent: bool = False):
    """Weighted CI of K solutions.

    pos, vel [B,K,3]; quat [B,K,4] xyzw; P [B,K,9,9] (blocks pos 0:3, vel 3:6, att 6:9);
    avail [B,K,3] bool: solution k reported a covariance for block (pos, vel, att) and is eligible;
    w [B,K] fusion weights (>= 0, sum 1 over eligible solutions, 0 for ineligible), or [B,K,3] one
    weight per block (position, velocity, attitude).
    Returns dict: pos, vel, quat [B,..], P [B,9,9] (block diagonal; NaN where undefined),
    ok [B,3] per block.  Gradients flow to w (and to the inputs).  independent=True gives the
    'covariance-based (independent information) fusion' comparison (w only marks eligibility).
    """
    B, K = w.shape[:2]
    dt = pos.dtype
    wb = (lambda b: w[..., b]) if w.dim() == 3 else (lambda b: w)  # noqa: E731
    wm = w.mean(-1) if w.dim() == 3 else w
    eligible = avail.any(-1) & (wm > 0)
    has = eligible.any(-1)
    k_ref = torch.where(eligible, wm, torch.full_like(wm, -1.0)).argmax(-1)     # largest weight
    gi = k_ref[:, None, None]
    p_ref = torch.gather(pos, 1, gi.expand(B, 1, 3)).squeeze(1).detach()
    v_ref = torch.gather(vel, 1, gi.expand(B, 1, 3)).squeeze(1).detach()
    out = {}
    dpv, Ppv, okpv = _block_ci(torch.stack([pos - p_ref[:, None], vel - v_ref[:, None]], 2),
                               torch.stack([P[..., 0:3, 0:3], P[..., 3:6, 3:6]], 2),
                               torch.stack([wb(0), wb(1)], -1), avail[..., 0:2], independent)
    dp, dv, Pp, Pv, okp, okv = dpv[:, 0], dpv[:, 1], Ppv[:, 0], Ppv[:, 1], okpv[:, 0], okpv[:, 1]
    out["pos"], out["vel"] = p_ref + dp, v_ref + dv
    R = G.quat_to_rot(quat)                                                      # [B,K,3,3]
    R_ref = torch.gather(R, 1, gi[..., None].expand(B, 1, 3, 3)).squeeze(1).detach()
    Pa = P[..., 6:9, 6:9]
    for _ in range(max(1, iters)):
        phi = G.log(R_ref[:, None].transpose(-1, -2) @ R)                       # [B,K,3]
        J = G.jr_inv(phi)
        Pa_ref = J @ Pa @ J.transpose(-1, -2)
        Pa_ref = 0.5 * (Pa_ref + Pa_ref.transpose(-1, -2))
        da, PA, oka = _block_ci(phi, Pa_ref, wb(2), avail[..., 2], independent)
        R_new = R_ref @ G.exp(torch.nan_to_num(da))
        R_ref = torch.where(oka[:, None, None], R_new, R_ref)
    out["quat"] = torch.where(oka[:, None], G.rot_to_quat(R_ref), torch.full((B, 4), float("nan"), dtype=dt,
                                                                             device=w.device))
    Pf = torch.zeros(B, 9, 9, dtype=dt, device=w.device)
    Pf[:, 0:3, 0:3], Pf[:, 3:6, 3:6], Pf[:, 6:9, 6:9] = Pp, Pv, PA
    out["P"] = Pf
    out["ok"] = torch.stack([okp, okv, oka], -1) & has[:, None]
    out["pos"] = torch.where(out["ok"][:, 0:1], out["pos"], torch.full_like(out["pos"], float("nan")))
    out["vel"] = torch.where(out["ok"][:, 1:2], out["vel"], torch.full_like(out["vel"], float("nan")))
    return out



def inflate(P, avail, u):
    """Learned covariance inflation (user decision B, 2026-11-18): P~_k,b = gamma_k,b P_k,b, gamma >= 1.

    log gamma_k,b = softplus(log gamma_norm_k,b + u_k,b) with
      gamma_norm_k,b = max_j tr(P_j,b) / tr(P_k,b) over the solutions reporting block b at this tick
    (gamma_norm equalises the reported scales to the most conservative one; it is computed from the
    reported values only, nothing is fitted).  u [B,K,3] is the network output; u = 0 is the
    'trace-normalised CI' comparison.  The reported covariance keeps its shape (axis ratios, cross
    terms); only its scale changes.  Returns (P~ [B,K,9,9], log_gamma [B,K,3])."""
    tr = torch.stack([torch.diagonal(P[..., 3 * b:3 * b + 3, 3 * b:3 * b + 3], dim1=-2, dim2=-1).sum(-1)
                      for b in range(3)], -1)                                   # [B,K,3]
    tr = torch.where(avail, tr, torch.zeros_like(tr))
    ref = tr.max(1, keepdim=True).values
    lg_norm = torch.where(avail, torch.log(ref.clamp_min(1e-300) / tr.clamp_min(1e-300)), torch.zeros_like(tr))
    lg = torch.nn.functional.softplus(lg_norm + u)
    lg = torch.where(avail, lg, torch.zeros_like(lg))
    scale = torch.exp(lg)
    Pt = P.clone()
    for b in range(3):
        Pt[..., 3 * b:3 * b + 3, 3 * b:3 * b + 3] = P[..., 3 * b:3 * b + 3, 3 * b:3 * b + 3] * scale[..., b, None, None]
    return Pt, lg
