"""Mathematical properties of the fusion operator (ijcas/fusion/geometry.py, ci.py)."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ijcas.fusion import geometry as G
from ijcas.fusion.ci import fuse

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only (a module-level default would leak into other test files)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _rand_spd(rng, n=3, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


def _P9(rng, sp=1.0, sv=0.1, sa=1e-3):
    P = np.zeros((9, 9))
    P[0:3, 0:3], P[3:6, 3:6], P[6:9, 6:9] = _rand_spd(rng, 3, sp), _rand_spd(rng, 3, sv), _rand_spd(rng, 3, sa)
    return P


def test_quaternion_and_so3():
    rng = np.random.default_rng(0)
    q = Rotation.random(200, random_state=1).as_quat()
    R = G.quat_to_rot(torch.tensor(q))
    assert np.allclose(R.numpy(), Rotation.from_quat(q).as_matrix(), atol=1e-12)
    q2 = G.rot_to_quat(R).numpy()
    assert np.allclose(np.abs((q2 * q).sum(-1)), 1.0, atol=1e-12)          # same rotation (sign)
    d = rng.standard_normal((200, 3))
    w = torch.tensor(d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0, 3.1, (200, 1)))  # |w| < pi
    assert torch.allclose(G.log(G.exp(w)), w, atol=1e-9)
    assert np.allclose(G.exp(w).numpy(), Rotation.from_rotvec(w.numpy()).as_matrix(), atol=1e-12)
    small = torch.tensor([[1e-9, -2e-9, 3e-10]])
    assert torch.allclose(G.log(G.exp(small)), small, atol=1e-15)


def test_jr_inverse_matches_numeric():
    rng = np.random.default_rng(1)
    for _ in range(20):
        phi = torch.tensor(rng.standard_normal(3) * 0.8)
        eps = torch.tensor(rng.standard_normal(3) * 1e-6)
        lhs = G.log(G.exp(phi) @ G.exp(eps))                            # Log(Exp(phi) Exp(eps))
        rhs = phi + G.jr_inv(phi) @ eps
        assert torch.allclose(lhs, rhs, atol=1e-10)


def _state(rng, K):
    pos = torch.tensor(rng.standard_normal((1, K, 3)))
    vel = torch.tensor(rng.standard_normal((1, K, 3)))
    quat = torch.tensor(Rotation.random(K, random_state=int(rng.integers(1e6))).as_quat()[None])
    P = torch.tensor(np.stack([_P9(rng) for _ in range(K)])[None])
    return pos, vel, quat, P


def test_single_solution_is_reproduced():
    rng = np.random.default_rng(2)
    pos, vel, quat, P = _state(rng, 3)
    avail = torch.tensor([[[False] * 3, [True] * 3, [False] * 3]])
    w = torch.tensor([[0.0, 1.0, 0.0]])
    o = fuse(pos, vel, quat, P, avail, w)
    assert torch.allclose(o["pos"], pos[:, 1]) and torch.allclose(o["vel"], vel[:, 1])
    assert np.allclose(np.abs((o["quat"][0].numpy() * quat[0, 1].numpy()).sum()), 1.0, atol=1e-12)
    for a, b in ((0, 3), (3, 6), (6, 9)):
        assert torch.allclose(o["P"][0, a:b, a:b], P[0, 1, a:b, a:b], atol=1e-12)


def test_identical_solutions_are_not_double_counted():
    rng = np.random.default_rng(3)
    pos, vel, quat, P = _state(rng, 1)
    pos, vel, quat, P = (x.repeat(1, 2, *([1] * (x.dim() - 2))) for x in (pos, vel, quat, P))
    avail = torch.ones(1, 2, 3, dtype=torch.bool)
    w = torch.tensor([[0.3, 0.7]])
    o = fuse(pos, vel, quat, P, avail, w)
    assert torch.allclose(o["P"][0, 0:3, 0:3], P[0, 0, 0:3, 0:3], atol=1e-10)   # CI: P_f = P
    oi = fuse(pos, vel, quat, P, avail, torch.tensor([[0.5, 0.5]]), independent=True)
    assert torch.allclose(oi["P"][0, 0:3, 0:3], P[0, 0, 0:3, 0:3] / 2, atol=1e-10)  # independent: P/2


def test_ineligible_and_empty():
    rng = np.random.default_rng(4)
    pos, vel, quat, P = _state(rng, 3)
    avail = torch.ones(1, 3, 3, dtype=torch.bool)
    o1 = fuse(pos, vel, quat, P, avail, torch.tensor([[0.5, 0.5, 0.0]]))
    pos2 = pos.clone()
    pos2[0, 2] += 1000.0                                                  # w = 0: no influence
    o2 = fuse(pos2, vel, quat, P, avail, torch.tensor([[0.5, 0.5, 0.0]]))
    assert torch.allclose(o1["pos"], o2["pos"])
    o3 = fuse(pos, vel, quat, P, torch.zeros(1, 3, 3, dtype=torch.bool), torch.zeros(1, 3))
    assert not o3["ok"].any() and torch.isnan(o3["pos"]).all() and torch.isnan(o3["quat"]).all()


def test_fused_covariance_spd_and_gradient():
    rng = np.random.default_rng(5)
    pos, vel, quat, P = _state(rng, 3)
    logits = torch.zeros(1, 3, requires_grad=True)
    w = torch.softmax(logits, -1)
    o = fuse(pos, vel, quat, P, torch.ones(1, 3, 3, dtype=torch.bool), w)
    Pf = o["P"][0]
    assert torch.allclose(Pf, Pf.T, atol=1e-12) and (torch.linalg.eigvalsh(Pf) > 0).all()
    gt_p, gt_v = torch.zeros(3), torch.zeros(3)
    gt_R = torch.eye(3)
    loss = (o["pos"][0] - gt_p).norm() + (o["vel"][0] - gt_v).norm() + \
        G.log(gt_R.T @ G.quat_to_rot(o["quat"][0])).norm()
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_ci_consistent_under_unknown_correlation_independent_is_not():
    """Two estimates with correlation 0.9 (shared IMU analogue): CI keeps NEES <= dim on average,
    independent information fusion is over-confident."""
    rng = np.random.default_rng(6)
    n = 4000
    Pk = np.diag([1.0, 2.0, 0.5])
    rho = 0.9
    Lk = np.linalg.cholesky(Pk)
    z0 = rng.standard_normal((n, 3))
    e1 = (z0 @ Lk.T)
    e2 = rho * e1 + np.sqrt(1 - rho ** 2) * (rng.standard_normal((n, 3)) @ Lk.T)
    pos = torch.tensor(np.stack([e1, e2], 1))                              # truth = 0
    vel = torch.zeros(n, 2, 3)
    quat = torch.tensor([0, 0, 0, 1.0]).expand(n, 2, 4).clone()
    P = torch.zeros(n, 2, 9, 9)
    P[:, :, 0:3, 0:3] = torch.tensor(Pk)
    P[:, :, 3:6, 3:6] = torch.eye(3)
    P[:, :, 6:9, 6:9] = torch.eye(3)
    avail = torch.ones(n, 2, 3, dtype=torch.bool)
    w = torch.full((n, 2), 0.5)

    def nees(o):
        e = o["pos"]
        S = torch.linalg.inv(o["P"][:, 0:3, 0:3])
        return float((e[:, None] @ S @ e[..., None]).mean())
    assert nees(fuse(pos, vel, quat, P, avail, w)) <= 3.2
    assert nees(fuse(pos, vel, quat, P, avail, w, independent=True)) > 4.5


def test_attitude_fusion_on_manifold():
    """Two attitudes +-a about z with equal covariance and weights -> the midpoint, also across +-pi."""
    for base in (0.0, np.pi - 0.05):
        a = 0.4
        q = Rotation.from_euler("z", [base - a, base + a]).as_quat()
        quat = torch.tensor(q[None])
        P = torch.eye(9).expand(1, 2, 9, 9).clone() * 0.01
        o = fuse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), quat, P, torch.ones(1, 2, 3, dtype=torch.bool),
                 torch.tensor([[0.5, 0.5]]))
        yaw = Rotation.from_quat(o["quat"][0].numpy()).as_euler("xyz")[2]
        assert abs(np.angle(np.exp(1j * (yaw - base)))) < 1e-9


def test_inflation_gamma_ge_1_shape_kept_and_trainable():
    from ijcas.fusion.ci import inflate
    rng = np.random.default_rng(7)
    pos, vel, quat, P = _state(rng, 3)
    P[0, 2] *= 1e-6                                                       # an over-confident solution
    avail = torch.ones(1, 3, 3, dtype=torch.bool)
    u = torch.zeros(1, 3, 3, requires_grad=True)
    Pt, lg = inflate(P, avail, u)
    assert (lg >= 0).all()                                                # gamma >= 1
    for k in range(3):
        for b in range(3):
            A, B = P[0, k, 3 * b:3 * b + 3, 3 * b:3 * b + 3], Pt[0, k, 3 * b:3 * b + 3, 3 * b:3 * b + 3]
            assert torch.allclose(B / B.trace(), A / A.trace(), atol=1e-12)  # shape kept
    o = fuse(pos, vel, quat, Pt, avail, torch.full((1, 3), 1 / 3))
    o["pos"].sum().backward()
    assert (u.grad[0, :, 0].abs() > 1e-6).all()                           # every solution's u matters
