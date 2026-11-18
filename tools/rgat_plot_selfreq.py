"""Selection share per scenario (val split) from results/rgat/val_selection_freq.json.

    python3 tools/rgat_plot_selfreq.py  -> results/rgat/figures/val_selection_freq.png

The JSON holds per-run tick counts of each selected candidate (eval window t >= 20 s) for the frozen
models (all seeds), the rule comparators and the oracle; written by tools/rgat_sel_freq.py.
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
sys.path.insert(0, str(ROOT / "tools"))
from rgat_plot_val import INK, INK2, SURF, _font  # noqa: E402

RES = ROOT / "results/rgat"
LAB = ["GNSS/INS (PX4 EKF2)", "VIO (OpenVINS)", "LIO (FAST-LIO2)", "INS 전파", "유효한 항법 없음"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c9c8c2"]
METHODS = [("rgat_dyn", "rgat_dyn (3시드)"), ("cr_gstar_kd", "cr_gstar_kd (3시드)"),
           ("Ontology-K", "Ontology-K"), ("H-FSM", "H-FSM"), ("Oracle", "Oracle (참값, 매 시점 최소 비용)")]
SC_NAME = {"S00": "정상", "S01": "GNSS 단절", "S02": "GNSS 위치 편향", "S03": "카메라 열화", "S04": "LiDAR 열화",
           "S05": "GNSS 단절+카메라", "S06": "GNSS 단절+LiDAR", "S07": "센서 지연·손실", "S08": "외부 센서 전부 상실",
           "S09": "미지 장면 조합", "S10": "IMU 결함"}


def share(seeds, mask):
    S = []
    for s in seeds:
        c = np.array([r["all"] for i, r in enumerate(s) if mask[i]], float).sum(0)
        S.append(c / c.sum())
    return np.mean(S, 0)


def main(path=RES / "val_selection_freq.json", out=RES / "figures/val_selection_freq.png", title="검증 세트"):
    _font()
    d = json.loads(Path(path).read_text())
    sc = np.array(d["scenario"])
    rows = [("전체", np.ones(len(sc), bool))] + [(f"{s} {SC_NAME.get(s, '')}", sc == s) for s in sorted(set(sc))]
    fig, axs = plt.subplots(1, len(METHODS), figsize=(17, 6.8), sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    y[0] += 0.4                                                       # gap under the overall row
    for ax, (m, mt) in zip(axs, METHODS):
        for yi, (_, mask) in zip(y, rows):
            p = share(d["methods"][m], mask) * 100
            left = 0.0
            for k in range(5):
                if p[k] <= 0:
                    continue
                ax.barh(yi, p[k], left=left, height=0.62, color=COL[k], edgecolor=SURF, linewidth=1.5)
                if p[k] >= 14:
                    ax.text(left + p[k] / 2, yi, f"{p[k]:.0f}", ha="center", va="center", fontsize=8.5,
                            color="#ffffff" if k in (0, 1) else INK)
                left += p[k]
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0", "50", "100 %"])
        ax.set_title(mt, loc="left", fontsize=11)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    axs[0].set_yticks(y, [r for r, _ in rows])
    axs[0].get_yticklabels()[0].set_fontweight("bold")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COL, LAB)], loc="lower center", ncol=5,
               frameon=False, fontsize=10)
    n = len(sc)
    fig.suptitle(f"선택된 항법 알고리즘의 시간 비율 ({title} {n}회, 평가 구간 t ≥ 20 s; 막대 안 숫자 = %)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.925, f"각 시나리오 {n // len(set(sc))}회 · 학습 변형은 동결 모델 3시드 평균 · 규칙 기반은 자체 10 Hz, 학습 선택기는 20 Hz",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        main(RES / "test/test_selection_freq.json", RES / "figures/test_selection_freq.png", "시험 세트")
    else:
        main()
