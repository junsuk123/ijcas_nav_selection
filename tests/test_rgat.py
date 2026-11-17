"""Graph selector invariants: ontology-protected edges, sparse operation == masked batch,
pruned edges are not computed, selection policies, no ground truth in the inputs."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from ijcas.rgat.features import SPEC
from ijcas.rgat.graph import (DEFINITIONAL, NODES, ONTOLOGY_NAME, candidate_edges, restrict)
from ijcas.rgat.learn import SelectCfg, select

DATA = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data"))
from ijcas.rgat.model import RecurrentRelGNN

ROOT = Path(__file__).resolve().parents[1]
DIMS = [3 * len(SPEC[n]) for n in NODES]
# capability -> sensor-quality node that carries it
CAP_NODE = {"Inertial": "S_imu", "MagHeading": "S_imu", "BaroHeight": "S_imu",
            "GnssPosition": "S_gnss", "GnssVelocity": "S_gnss", "VisualFeatures": "S_cam",
            "LidarStructure": "S_lidar"}


def test_protected_edges_follow_the_ontology():
    import rdflib
    g = rdflib.Graph().parse(ROOT / "ijcas/knowledge/ijcas_nav.ttl")
    ns = dict(g.namespaces())
    base = rdflib.Namespace(str(ns[""]))
    est = {v: k for k, v in ONTOLOGY_NAME.items() if k.startswith("N_")}
    derived = set()
    for prop in ("requiresEssential", "usesAiding"):
        for s, _, o in g.triples((None, base[prop], None)):
            e, c = str(s).split("#")[-1].split("/")[-1], str(o).split("#")[-1].split("/")[-1]
            if e in est and c in CAP_NODE:
                derived.add((CAP_NODE[c], est[e]))
    assert derived == DEFINITIONAL
    prot = {(e.src, e.dst) for e in candidate_edges() if e.protected and e.rel != "self"}
    assert prot == DEFINITIONAL


def _model(agg):
    torch.manual_seed(0)
    m = RecurrentRelGNN(DIMS, candidate_edges(), agg=agg).eval()
    with torch.no_grad():   # switch one relation's gates off, keep the others on
        m.gate_b[1] = -20.0
        m.U.zero_()
    return m


@pytest.mark.parametrize("agg", ["gat", "gcn"])
def test_sparse_operation_equals_masked_batch(agg):
    m = _model(agg)
    x = torch.randn(1, len(NODES), max(DIMS))
    v = torch.tensor([[True, False, True, True]])
    with torch.no_grad():
        l1, s1, a1 = m.step(x, m.init_state(1), v, "eval")
        l2, s2, a2 = m.step(x.repeat(3, 1, 1), m.init_state(3), v.repeat(3, 1), "eval")
    assert 0 < len(a1["active_idx"]) < len(m.edges)          # some edges really skipped
    assert torch.allclose(l1, l2[:1], atol=1e-5)
    assert torch.allclose(s1["h"], s2["h"][:1], atol=1e-5)
    assert l1[0, 1] < -1e8                                     # ineligible candidate masked


def test_pruned_edges_are_not_computed():
    full = candidate_edges()
    gated = [e.name for e in full if not e.protected]
    keep = set(gated[::3])
    m = RecurrentRelGNN(DIMS, restrict(full, keep)).eval()
    assert set(m.gated_names) == keep
    assert all(e.protected or e.name in keep for e in m.edges)
    st = m.init_state(1)
    assert st["z"].shape[1] == len(keep)
    with torch.no_grad():
        m.step(torch.randn(1, len(NODES), max(DIMS)), st, torch.ones(1, 4, dtype=torch.bool), "eval")


def test_selection_policies():
    t = np.arange(0, 10, 0.05)
    T = len(t)
    pi = np.tile([0.6, 0.4, 0.0, 0.0], (T, 1))
    valid = np.ones((T, 4), bool)
    valid[:, 2:] = False
    cfg = SelectCfg(margin=0.1, dwell_s=2.0, confirm_s=1.0)
    # candidate 0 becomes ineligible at t=3 -> immediate switch (no dwell)
    v = valid.copy()
    v[t >= 3, 0] = False
    sel, ev = select(pi, v, t, cfg)
    i = np.searchsorted(t, 3.0)
    assert sel[i] == 1 and ev[-1][3] == "policy:current_ineligible"
    # a better score only wins after the dwell and by more than the margin
    pi2 = pi.copy()
    pi2[t >= 0.5] = [0.3, 0.7, 0.0, 0.0]
    sel, ev = select(pi2, valid, t, cfg)
    assert sel[np.searchsorted(t, 1.0)] == 0
    assert sel[np.searchsorted(t, 2.05)] == 1
    # a candidate that just became eligible must be confirmed first
    v3 = valid.copy()
    v3[t < 4, 1] = False
    sel, _ = select(pi2, v3, t, cfg)
    assert sel[np.searchsorted(t, 4.5)] == 0 and sel[np.searchsorted(t, 5.05)] == 1


def _any_run():
    data = Path(os.environ.get("IJCAS_DATA", Path.home() / "ijcas_data")) / "runs" / "validation"
    if data.exists():
        for r in sorted(data.iterdir()):
            if (r / "DONE").exists():
                return r
    return None


def test_inputs_do_not_depend_on_ground_truth(tmp_path):
    run = _any_run()
    if run is None:
        pytest.skip("no recorded run available")
    from ijcas.rgat.dataset import build_run
    from ijcas.selector.common import load_config
    cfg = load_config(ROOT / "config/selector/thresholds.yaml")
    a = build_run(run, cfg)
    cp = tmp_path / run.name
    shutil.copytree(run, cp)
    gt = np.load(cp / "gt.npz")["gt"].copy()
    gt[:, 1:4] += 37.0                                         # shift the true trajectory
    np.savez(cp / "gt.npz", gt=gt)
    b = build_run(cp, cfg)
    for k in a:
        if k.startswith(("y_", "m_", "dt_")) or k == "valid":
            assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert not np.allclose(a["err_pos"], b["err_pos"], equal_nan=True)


def test_masked_softmax_no_underflow():
    import torch
    from ijcas.rgat.learn import masked_softmax
    lg = torch.tensor([[0.0, 1.0, 500.0, -3.0], [5.0, 5.0, 5.0, 5.0]])
    v = torch.tensor([[True, True, False, True], [False, False, False, False]])
    pi = masked_softmax(lg, v)
    assert torch.allclose(pi[0].sum(), torch.tensor(1.0)) and pi[0, 2] == 0 and pi[0, 1] > pi[0, 0]
    assert torch.all(pi[1] == 0)


def test_cost_continuous_and_monotone():
    import numpy as np
    from ijcas.rgat.learn import costs
    e = np.array([0.5, 2.9, 3.0, 3.1, 15.0, 50.0, 1e4])
    run = {"err_pos": np.stack([e] * 4, 1), "err_vel": np.zeros((len(e), 4)), "err_att": np.zeros((len(e), 4)),
           "valid": np.ones((len(e), 4), bool)}
    c, _ = costs(run)
    assert np.all(np.diff(c[:, 0]) > 0) and abs(c[2, 0] - 9.0) < 1e-4


def test_retrieval_runtime_matches_batch_and_hold():
    """The numpy operation path (no torch, B = 1) gives the same pi / hold as batch inference."""
    import glob
    import numpy as np
    import pytest
    import ijcas.rgat.retrieval as R
    from ijcas.rgat.learn import Normalizer, SelectCfg, load_run, select
    paths = sorted(glob.glob(str(DATA / "rgat" / "dataset" / "validation" / "*.npz")))
    if len(paths) < 2:
        pytest.skip("no validation dataset")
    old = dict(R.CR)
    R.CR.update(epochs=1, M=16, M_plain=32, init_sub=3000, k=8)
    try:
        tr, va = [load_run(paths[0])], [load_run(paths[1])]
        norm = Normalizer().fit(tr)
        S, C, V = R._ticks(tr, norm)
        cb = R.CaseBase(R.groups_for("cr_e0"), "cpu")
        R.fit_cr(cb, S, C, V, None, 0, supervised=True, log=lambda m: None)
        pi_b, hold_b, _ = R.predict(cb, va, norm)[0]
        rt, r = R.Runtime(cb, norm), va[0]
        Xp = norm.pack(r).reshape(len(r["t"]), -1).astype(np.float32)
        dt = np.diff(r["t"], prepend=r["t"][0] - 0.05)
        out = [rt.step(Xp[i], float(dt[i]), r["valid"][i]) for i in range(len(r["t"]))]
        m = r["valid"].any(1)
        assert np.abs(np.array([o[0] for o in out])[m] - pi_b[m]).max() < 1e-4
        assert np.mean(np.array([o[1] for o in out]) == hold_b) > 0.999
        # hold blocks learned switches only
        sel_free, _ = select(pi_b, r["valid"], r["t"], SelectCfg(0.0, 0.5, 1.0))
        sel_hold, ev = select(pi_b, r["valid"], r["t"], SelectCfg(0.0, 0.5, 1.0), hold=np.ones(len(m), bool))
        assert all(e[3].startswith("policy:") for e in ev)
        assert len(sel_hold) == len(sel_free)
    finally:
        R.CR.clear(); R.CR.update(old)
