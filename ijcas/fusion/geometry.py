"""SO(3) helpers in torch (batched, differentiable) for the navigation-solution fusion.

Conventions: quaternions are [x, y, z, w] (ROS / scipy); a rotation R maps body to world; an attitude
error is a right (body-axis) perturbation R_true = R Exp(eps), which is the convention of FAST-LIO2's
boxplus and is assumed for the reported orientation covariances of the three estimators (assumption,
docs/FUSION_DESIGN_KO.md).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def hat(w: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] skew matrix."""
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] xyzw (normalised here) -> [..., 3, 3]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 4] xyzw with w >= 0 (branch-free Shepperd selection)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = torch.stack([
        torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + t], -1),
        torch.stack([1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0],
                     m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
        torch.stack([m[..., 0, 1] + m[..., 1, 0], 1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2],
                     m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1],
                     1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], m[..., 1, 0] - m[..., 0, 1]], -1)], -2)
    k = torch.stack([t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1).argmax(-1)
    q = torch.gather(cands, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.where(q[..., 3:4] < 0, -q, q)


def exp(w: torch.Tensor) -> torch.Tensor:
    """so(3) -> SO(3), [..., 3] -> [..., 3, 3] (Rodrigues; series near 0)."""
    th2 = (w * w).sum(-1)
    th = th2.clamp_min(EPS * EPS).sqrt()
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(EPS * EPS))
    W = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    return eye + a[..., None, None] * W + b[..., None, None] * (W @ W)


def log(R: torch.Tensor) -> torch.Tensor:
    """SO(3) -> so(3) via the quaternion (stable at 0 and near pi), [..., 3, 3] -> [..., 3]."""
    q = rot_to_quat(R)
    v, w = q[..., :3], q[..., 3]
    n = v.norm(dim=-1)
    ang = 2 * torch.atan2(n, w)
    scale = torch.where(n < 1e-7, 2.0 / w.clamp_min(EPS), ang / n.clamp_min(EPS))
    return v * scale[..., None]


def jr_inv(phi: torch.Tensor) -> torch.Tensor:
    """Inverse right Jacobian of SO(3), [..., 3] -> [..., 3, 3]."""
    th2 = (phi * phi).sum(-1)
    th = th2.clamp_min(EPS * EPS).sqrt()
    small = th2 < 1e-8
    c = torch.where(small, torch.full_like(th, 1.0 / 12.0),
                    1 / th2.clamp_min(EPS * EPS) - (1 + torch.cos(th)) / (2 * th * torch.sin(th).clamp_min(EPS)))
    P = hat(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(P.shape)
    return eye + 0.5 * P + c[..., None, None] * (P @ P)
