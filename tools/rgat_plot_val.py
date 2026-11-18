"""Validation-set figures for the graph / retrieval selectors (no test data is read).

    python3 tools/rgat_plot_val.py   -> results/rgat/figures/val_*.png

Reads results/rgat/runs/<variant>/seed<k>/report.json (per-seed values) and the baselines in
results/rgat/val_summary.json (written by `tools/rgat_run.py val-summary`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/rgat"
OUT = RES / "figures"

# reference palette (light), validated: blue / orange / aqua, all-pairs CVD >= 9.2
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C_NN, C_CR, C_RULE, C_FIX = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
NN = ["rgat_dyn", "rgcn_dyn", "rgat_static", "rgat_dense", "rgat_nomem", "gru_flat"]
CR = ["cr_plain", "cr_e0", "cr_gstar", "cr_gstar_kd"]
RULE = ["H-FSM", "Ontology-K"]
FIX = ["F-vio", "F-ins", "F-gnss_ins", "F-lio"]
LABELS = {"full": "검증 세트(validation 시드 8–10, 30회)", "short": "검증 세트", "best": "val 최저인"}
REG = [("normal", "정상"), ("single", "단일 열화"), ("compound", "복합 열화"), ("recovery", "복구(종료 후 15 s)")]


def _font():
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f and ("Regular" in f or f.endswith("NotoSansCJK-Regular.ttc")):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False})


def color(n):
    return C_NN if n in NN else C_CR if n in CR else C_RULE if n in RULE else C_FIX


def load():
    per_seed = {}
    for v in NN + CR:
        reps = [json.loads((d / "report.json").read_text()) for d in sorted((RES / "runs" / v).glob("seed*"))
                if (d / "report.json").exists()]
        if reps:
            per_seed[v] = reps
    vs = RES / "val_summary.json"
    if not vs.exists():   # baselines do not depend on training; the discarded run 1 computed them on the same val set
        vs = ROOT / "results/rgat_archive/run1_linear_tail_20261117/val_summary.json"
        print("baselines from", vs)
    base = json.loads(vs.read_text())["baselines"]
    return per_seed, base


def fig_rmse(per_seed, base):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) all methods, means
    rows = [(n, np.mean([r["val"]["pos_rmse_m"] for r in per_seed[n]])) for n in NN + CR if n in per_seed]
    rows += [(n, base[n]["pos_rmse_m"]) for n in RULE + FIX]
    rows.sort(key=lambda x: x[1])
    y = np.arange(len(rows))[::-1]
    vs = sorted(v for _, v in rows)
    cap = min(vs[-1], 2.0 * vs[-2]) * 1.0                 # one runaway baseline (e.g. a diverged engine)
    for yi, (n, v) in zip(y, rows):
        a1.barh(yi, min(v, cap), height=0.55, color=color(n), edgecolor=SURF, linewidth=0)
        a1.text(min(v, cap) + cap * 0.012, yi, f"{v:.3f}" + (" (축 밖)" if v > cap else ""), va="center",
                fontsize=9, color=INK2)
    a1.axvline(base["Oracle"]["pos_rmse_m"], color=INK2, lw=1.2, ls=(0, (4, 3)))
    a1.text(base["Oracle"]["pos_rmse_m"] + 0.03, -0.95, f"Oracle {base['Oracle']['pos_rmse_m']:.3f}",
            fontsize=9, color=INK2, va="center")
    a1.set_yticks(y, [n for n, _ in rows])
    a1.set_xlim(0, cap * 1.22)
    a1.set_ylim(-1.3, len(rows) - 0.4)
    a1.set_xlabel("위치 RMSE [m] (낮을수록 좋음)")
    a1.set_title("(a) 전체 방법 (학습 변형은 3시드 평균)", loc="left", fontsize=11.5)
    a1.grid(axis="y", visible=False)
    # (b) zoom: learned variants, per seed
    names = [n for n in NN + CR if n in per_seed]
    yb = np.arange(len(names))[::-1]
    for yi, n in zip(yb, names):
        vals = [r["val"]["pos_rmse_m"] for r in per_seed[n]]
        m = np.mean(vals)
        a2.plot([min(vals), max(vals)], [yi, yi], color=color(n), lw=2, alpha=0.45, solid_capstyle="round")
        a2.scatter(vals, [yi] * len(vals), s=42, color=color(n), edgecolor=SURF, linewidth=1.5, zorder=3)
        a2.plot([m, m], [yi - 0.28, yi + 0.28], color=INK, lw=2)
        a2.text(0.995, yi, f"{m:.3f} ± {np.std(vals):.3f}", va="center", ha="right", fontsize=9, color=INK2,
                transform=a2.get_yaxis_transform())
    for ref, lab in ((base["Oracle"]["pos_rmse_m"], "Oracle"), (base["F-vio"]["pos_rmse_m"], "F-vio")):
        a2.axvline(ref, color=INK2, lw=1.2, ls=(0, (4, 3)))
        a2.text(ref + 0.003, -0.95, lab, fontsize=9, color=INK2, va="center")
    a2.set_yticks(yb, names)
    allv = [r["val"]["pos_rmse_m"] for n in names for r in per_seed[n]]
    a2.set_xlim(min(0.70, base["Oracle"]["pos_rmse_m"] - 0.03), max(1.08, max(allv) * 1.12))
    a2.set_ylim(-1.3, len(names) - 0.4)
    a2.set_xlabel("위치 RMSE [m]  (점 = 시드, 세로 막대 = 평균)")
    av = [r["val"]["availability"] for n in names for r in per_seed[n]]
    a2.set_title(f"(b) 학습 선택기 확대 (가용성은 모두 {min(av):.3f}–{max(av):.3f})", loc="left", fontsize=11.5)
    a2.grid(axis="y", visible=False)
    _legend(fig)
    fig.suptitle(f"{LABELS['full']} 위치 RMSE", x=0.01, ha="left", fontsize=13.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def _legend(fig, extra=(), n=4):
    from matplotlib.patches import Patch
    h = [Patch(color=C_NN, label="온라인 신경망 선택기 (R-GAT 계열, GRU)"),
         Patch(color=C_CR, label="검색 기반 선택 (CR)"), Patch(color=C_RULE, label="규칙 기반 (1차 구현)"),
         Patch(color=C_FIX, label="단일 항법기 고정")][:n] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=9.5)


def fig_tradeoff(per_seed, base):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    pts = [(n, np.mean([r["val"]["availability"] for r in per_seed[n]]),
            np.mean([r["val"]["pos_rmse_m"] for r in per_seed[n]])) for n in NN + CR if n in per_seed]
    pts += [(n, base[n]["availability"], base[n]["pos_rmse_m"]) for n in RULE + FIX]
    for n, a, e in pts:
        ax.scatter(a, e, s=70, color=color(n), edgecolor=SURF, linewidth=2, zorder=3)
    o = base["Oracle"]
    ax.scatter(o["availability"], o["pos_rmse_m"], s=90, marker="*", color=INK, zorder=4)
    lab = {"Oracle": (o["availability"], o["pos_rmse_m"])}
    for n, a, e in pts:
        if n in RULE + FIX:
            lab[n] = (a, e)
    for n, (a, e) in lab.items():
        ax.annotate(n, (a, e), xytext=(7, 4), textcoords="offset points", fontsize=9.5, color=INK2)
    ax.annotate("학습 선택기 10종\n(R-GAT 계열 6 + CR 4)", (0.984, 0.84), xytext=(0.925, 0.25), fontsize=9.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("가용성 (오차 ≤ 2 m 출력 비율, 높을수록 좋음)")
    ax.set_ylabel("위치 RMSE [m] (낮을수록 좋음)")
    ax.set_xlim(0.79, 1.0)
    ax.set_ylim(0, 3.6)
    ax.set_title("오른쪽 아래일수록 좋음 (가용성 높고 오차 작음)", loc="left", fontsize=11.5)
    _legend(fig)
    fig.suptitle(f"{LABELS['short']} 정확도와 가용성", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def fig_regimes(per_seed, base, best_cr=None):
    best_cr = best_cr or min((n for n in CR if n in per_seed),
                  key=lambda n: np.mean([r["val"]["pos_rmse_m"] for r in per_seed[n]]))
    methods = ["rgat_dyn", "gru_flat", best_cr, "H-FSM", "Ontology-K", "F-vio", "Oracle"]

    def val(n, g):
        if n in per_seed:
            v = [r["val"]["regimes"].get(g, {}).get("pos_rmse_m") for r in per_seed[n]]
            return np.mean([x for x in v if x is not None])
        return base[n]["regimes"].get(g, {}).get("pos_rmse_m", np.nan)

    fig, axs = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)
    y = np.arange(len(methods))[::-1]
    vmax = max(val(n, g) for n in methods for g, _ in REG)
    for ax, (g, title) in zip(axs, REG):
        for yi, n in zip(y, methods):
            v = val(n, g)
            c = INK2 if n == "Oracle" else color(n)
            ax.barh(yi, v, height=0.55, color=c, alpha=0.55 if n == "Oracle" else 1.0)
            ax.text(v + vmax * 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_xlim(0, vmax * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("위치 RMSE [m]")
    axs[0].set_yticks(y, methods)
    _legend(fig)
    fig.suptitle(f"체제별 위치 RMSE ({LABELS['short']}; 학습 변형은 3시드 평균, CR은 {LABELS['best']} {best_cr})", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def fig_edges():
    d = RES / "runs/rgat_dyn/seed0"
    rep = json.loads((d / "report.json").read_text())
    kept = set(json.loads((d / "structure.json").read_text())["kept_gated"])
    u = rep["prune"]["usage"]
    names = u["names"]
    M = np.array([u["rate"][g] for g, _ in REG]).T                           # [edges, regimes]
    order = np.argsort(-M.max(1))
    M, names = M[order], [names[i] for i in order]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                                                      "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.8, 13))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(4), [t for _, t in REG], fontsize=9.5)
    ax.set_yticks(range(len(names)), [("" if n in kept else "✕ ") + n.replace("->", " → ") for n in names],
                  fontsize=8.5)
    for t, n in zip(ax.get_yticklabels(), names):
        t.set_color(INK if n in kept else MUTED)
    ax.grid(False)
    ax.axvline(-0.5, color=SURF)
    for x in range(1, 4):
        ax.axvline(x - 0.5, color=SURF, lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("운용 모드 게이트 활성률", color=INK2)
    cb.outline.set_visible(False)
    removed = len(names) - len(kept & set(names))
    ax.set_title(f"rgat_dyn 시드 0: 게이트 간선 53개의 체제별 활성률 (val, 프루닝 판단 시점·미세조정 전)\n"
                 f"✕ = 프루닝으로 영구 제거 {removed}개", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def fig_latency(per_seed):
    lat = json.loads((RES / "latency.json").read_text())
    names = [n for n in NN + CR if n in lat and n in per_seed]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for n in names:
        L = lat[n]
        p50, p99 = (L["total"]["p50_ms"], L["total"]["p99_ms"]) if "total" in L else (L["p50_ms"], L["p99_ms"])
        e = np.mean([r["val"]["pos_rmse_m"] for r in per_seed[n]])
        ax.plot([p50, p99], [e, e], color=color(n), lw=2, alpha=0.45, solid_capstyle="round")
        ax.scatter([p99], [e], s=70, color=color(n), edgecolor=SURF, linewidth=2, zorder=3)
        ax.scatter([p50], [e], s=30, color=SURF, edgecolor=color(n), linewidth=1.5, zorder=3)
        pos = {"rgat_dense": (2.7, 0.838), "rgat_static": (2.7, 0.832), "rgat_dyn": (2.7, 0.826),
               "rgat_nomem": (2.7, 0.820), "rgcn_dyn": (0.75, 0.812), "gru_flat": (0.21, 0.814),
               "cr_e0": (0.45, 0.899), "cr_gstar": (0.48, 0.866), "cr_gstar_kd": (0.48, 0.849),
               "cr_plain": (0.105, 0.869)}.get(n, (p99 * 1.2, e))
        ax.annotate(n, (p99, e), xytext=pos, textcoords="data", fontsize=9.5, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=4))
    ax.axvline(5.0, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(4.6, 0.900, "판정 기준 p99 ≤ 5 ms\n(20 Hz 주기의 10 %)", ha="right", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.06, 8)
    ax.set_ylim(0.805, 0.91)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xlabel("선택기 1 tick 지연 [ms] (CPU 1스레드, 차량 1대; 빈 원 = p50, 채운 원 = p99, 로그 축)")
    ax.set_ylabel("val 위치 RMSE [m] (3시드 평균)")
    ax.set_title("R-GAT은 모델 1 tick, CR은 특징 생성 + 투영 + 검색 전체. 시험 세트 수집 중 측정(상한)",
                 loc="left", fontsize=10.5, color=INK2)
    _legend(fig, n=2)
    fig.suptitle("정확도–운용 지연 (시드 0 모델)", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def main():
    _font()
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed, base = load()
    for name, f in (("val_rmse", lambda: fig_rmse(per_seed, base)),
                    ("val_tradeoff", lambda: fig_tradeoff(per_seed, base)),
                    ("val_regimes", lambda: fig_regimes(per_seed, base)),
                    ("val_latency", lambda: fig_latency(per_seed)),
                    ("val_edges", fig_edges)):
        fig = f()
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
