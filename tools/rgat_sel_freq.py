"""Selection frequency on the val split (validation seeds 8-10, 30 runs) with the frozen models.

    python3 tools/rgat_sel_freq.py [validation|test]
        -> results/rgat/val_selection_freq.json | results/rgat/test/test_selection_freq.json
    (plot: tools/rgat_plot_selfreq.py [validation|test])

Per run and method: tick counts of each selected candidate [gnss_ins, vio, lio, ins, none] in the eval
window (t >= normal_s), overall ("all") and per regime ("reg0".."reg3": normal, single, compound,
recovery), plus switches.  Learned variants: every seed; rule comparators at their own 10 Hz.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ijcas.rgat import experiment as X  # noqa: E402
from ijcas.rgat.learn import load_run, stack, infer, select, oracle_sel  # noqa: E402
from ijcas.rgat.retrieval import load_cr, predict  # noqa: E402

RES = ROOT / "results/rgat"
SET = sys.argv[1] if len(sys.argv) > 1 else "validation"          # "validation" (val split) or "test" (all)
DS = Path(os.environ.get("IJCAS_DATA", "/home/j/ijcas_data")) / "rgat/dataset" / SET
_all = sorted(str(p) for p in DS.glob("*.npz"))
va_p = X.split(_all)[1] if SET == "validation" else _all
if SET == "test":
    import importlib.util
    _spec = importlib.util.spec_from_file_location("rgat_run", ROOT / "tools/rgat_run.py")
    _rr = importlib.util.module_from_spec(_spec)
    sys.argv = sys.argv[:1]
    _spec.loader.exec_module(_rr)
    _rr.check_frozen()                                                # frozen models only, after the test eval
runs = [load_run(p) for p in va_p]
dev = "cuda" if torch.cuda.is_available() else "cpu"
LAB = ["gnss_ins", "vio", "lio", "ins", "none"]


def share(sel, t, regime, normal_s, rt=None):
    m = t >= normal_s
    s = np.where(sel < 0, 4, sel)[m]
    return np.bincount(s, minlength=5).astype(float), regime[m]


def per_run_counts(sel_list, t_list, reg_list):
    out = []
    for sel, t, reg, r in zip(sel_list, t_list, reg_list, runs):
        m = t >= r["meta"]["normal_s"]
        s = np.where(sel < 0, 4, sel)[m]
        rg = reg[m]
        out.append({"all": np.bincount(s, minlength=5).tolist(),
                    **{f"reg{g}": np.bincount(s[rg == g], minlength=5).tolist() for g in range(4)},
                    "switches": int(np.sum(sel[1:] != sel[:-1]))})
    return out


res = {"runs": [r["meta"]["id"] for r in runs], "scenario": [r["meta"]["scenario"] for r in runs],
       "scene": [r["meta"]["scene"] for r in runs], "methods": {}}
T = [r["t"] for r in runs]
R = [r["regime"] for r in runs]
# eligibility share (how often each candidate is eligible) and oracle
res["eligible"] = [{"all": r["valid"][r["t"] >= r["meta"]["normal_s"]].sum(0).tolist(),
                    "n": int((r["t"] >= r["meta"]["normal_s"]).sum())} for r in runs]
res["methods"]["Oracle"] = [per_run_counts([oracle_sel(r) for r in runs], T, R)]
for key, name in (("ont", "Ontology-K"), ("hfsm", "H-FSM")):
    sels = [np.where(r[f"rule_{key}_mode"] == 2, -1, r[f"rule_{key}_sel"]) for r in runs]
    res["methods"][name] = [per_run_counts(sels, [r["rule_t"] for r in runs], [r["rule_regime"] for r in runs])]
for v in ("rgat_dyn", "rgcn_dyn", "rgat_static", "rgat_dense", "rgat_nomem", "gru_flat",
          "cr_plain", "cr_e0", "cr_gstar", "cr_gstar_kd"):
    seeds = []
    for d in sorted((RES / "runs" / v).glob("seed*")):
        if (d / "model.npz").exists():
            cb, norm, cfg, rep = load_cr(d, dev)
            pr = predict(cb, runs, norm)
            pis, holds = [p for p, _, _ in pr], [h for _, h, _ in pr]
        else:
            model, norm, cfg, rep = X.load_model(d, dev)
            Xv, Vv, _ = stack(runs, norm)
            pa = infer(model, Xv, Vv, dev, keep_gates=False)["pi"]
            pis, holds = [pa[b, :len(r["t"])] for b, r in enumerate(runs)], [None] * len(runs)
        sels = [select(pis[b], r["valid"], r["t"], cfg, hold=holds[b])[0] for b, r in enumerate(runs)]
        seeds.append(per_run_counts(sels, T, R))
        print(v, d.name, flush=True)
    res["methods"][v] = seeds
out = RES / ("val_selection_freq.json" if SET == "validation" else "test/test_selection_freq.json")
out.write_text(json.dumps(res))
print("->", out)
