"""Test-set figures from the one-time frozen evaluation (results/rgat/test/results.json).

    python3 tools/rgat_plot_test.py  -> results/rgat/figures/test_*.png, results/rgat/test/per_scenario.md

Reuses the validation figure functions.  The retrieval variant shown in the regime / scenario panels
is cr_gstar_kd, the one compared with rgat_dyn in the pre-registered comparison (design 12.2 ①) and
the lowest CR on val; it is not chosen from test numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import rgat_plot_val as PV  # noqa: E402
from ijcas.rgat.learn import pool  # noqa: E402

RES = ROOT / "results/rgat"
SHOW = ["rgat_dyn", "gru_flat", "cr_gstar_kd", "H-FSM", "Ontology-K", "F-vio", "Oracle"]
SC_NAME = {"S00": "정상", "S01": "GNSS 단절", "S02": "GNSS 위치 편향", "S03": "카메라 열화", "S04": "LiDAR 열화",
           "S05": "GNSS 단절+카메라", "S06": "GNSS 단절+LiDAR", "S07": "센서 지연·손실",
           "S08": "외부 센서 전부 상실", "S09": "미지 장면(office) 조합", "S10": "IMU 결함"}


def per_scenario(res):
    """{method: {scenario: (rmse mean over seeds, availability mean)}}"""
    out = {}
    scen = sorted({m["scenario"] for m in res["baselines"]["Oracle"]["per_run"]})
    for name, seeds in res["learned"].items():
        out[name] = {}
        for s in scen:
            P = [pool([m for m in sd["per_run"] if m["scenario"] == s]) for sd in seeds]
            out[name][s] = (float(np.mean([p["pos_rmse_m"] for p in P])), float(np.mean([p["availability"] for p in P])),
                            float(np.std([p["pos_rmse_m"] for p in P])))
    for name, b in res["baselines"].items():
        out[name] = {}
        for s in scen:
            p = pool([m for m in b["per_run"] if m["scenario"] == s])
            out[name][s] = (p["pos_rmse_m"], p["availability"], 0.0)
    return scen, out


def fig_scenarios(scen, ps):
    fig, axs = plt.subplots(3, 4, figsize=(17, 11.5), sharey=True)
    y = np.arange(len(SHOW))[::-1]
    for ax, s in zip(axs.flat, scen):
        vmax = max(ps[n][s][0] or 0 for n in SHOW)
        for yi, n in zip(y, SHOW):
            v, a, _ = ps[n][s]
            if v is None:
                continue
            c = PV.INK2 if n == "Oracle" else PV.color(n)
            ax.barh(yi, v, height=0.6, color=c, alpha=0.55 if n == "Oracle" else 1.0)
            ax.text(v + vmax * 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color=PV.INK2)
        ax.set_xlim(0, vmax * 1.3)
        ax.set_title(f"{s} {SC_NAME.get(s, '')}", loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    for ax in axs.flat[len(scen):]:
        ax.axis("off")
    for row in axs:
        row[0].set_yticks(y, SHOW)
    for ax in axs[-1]:
        ax.set_xlabel("위치 RMSE [m]")
    PV._legend(fig)
    fig.suptitle("시나리오별 위치 RMSE (시험 세트, 시나리오당 30회; 학습 변형은 3시드 평균)", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def main():
    PV._font()
    PV.LABELS.update(full="시험 세트(시드 101–130, 330회)", short="시험 세트", best="사전 비교 대상(val 최저)")
    res = json.loads((RES / "test/results.json").read_text())
    per_seed = {v: [{"val": s["pooled"]} for s in seeds] for v, seeds in res["learned"].items()}
    base = {k: v["pooled"] for k, v in res["baselines"].items()}
    scen, ps = per_scenario(res)
    out = RES / "figures"
    for name, f in (("test_rmse", lambda: PV.fig_rmse(per_seed, base)),
                    ("test_tradeoff", lambda: PV.fig_tradeoff(per_seed, base)),
                    ("test_regimes", lambda: PV.fig_regimes(per_seed, base, best_cr="cr_gstar_kd")),
                    ("test_scenarios", lambda: fig_scenarios(scen, ps))):
        fig = f()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        print(out / f"{name}.png")
    names = list(res["learned"]) + list(res["baselines"])
    L = ["| method | " + " | ".join(scen) + " |", "|---" * (len(scen) + 1) + "|"]
    for n in names:
        L.append(f"| {n} | " + " | ".join("-" if ps[n][s][0] is None else f"{ps[n][s][0]:.2f} ({ps[n][s][1]:.2f})"
                                          for s in scen) + " |")
    md = "# Test per-scenario position RMSE [m] (availability)\n\n" + "\n".join(L) + "\n"
    (RES / "test/per_scenario.md").write_text(md)
    (RES / "test/per_scenario.json").write_text(json.dumps(ps, indent=1))
    print(md)


if __name__ == "__main__":
    main()
