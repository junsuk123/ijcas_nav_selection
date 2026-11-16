"""G9: paper tables and figures from a G6 aggregate.

    python3 tools/g9_report.py --set test        (reads results/g6/test/{aggregate,per_run}.json)
    -> results/g9/<set>/report.md, fig_availability.png, fig_false_normal.png

Tables: per scenario and selector the run means of availability, false-normal time, no-valid-nav
time, switches; the paired difference comparator - Ontology-K with its run-level bootstrap 95 % CI
(from g6_aggregate).  Flat-K is reported as identical to Ontology-K when it is (G4: same knowledge,
same inputs) instead of as a separate column.
Figures (static, light): availability heatmap (scenario x selector, one-hue blue ramp, values
printed in every cell) and false-normal time of F-G / H-FSM / Ontology-K (grouped bars; the
table is the accessible view).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELECTORS = ["F-G", "F-V", "F-L", "H-FSM", "Ontology-K", "Oracle"]
NAMES = {"S00": "S00 nominal", "S01": "S01 GNSS outage", "S02": "S02 GNSS bias",
         "S03": "S03 visual degr.", "S04": "S04 LiDAR degr.", "S05": "S05 GNSS out + visual",
         "S06": "S06 GNSS out (corridor)", "S07": "S07 delay/drop", "S08": "S08 all aiding lost",
         "S09": "S09 unseen office", "S10": "S10 IMU fault"}
# reference palette (dataviz skill, light): sequential blue 100..700, categorical slots 1-3
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES = {"F-G": "#2a78d6", "H-FSM": "#eb6834", "Ontology-K": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fmt_ci(d):
    if not d or d.get("mean") is None:
        return "-"
    ci = d.get("ci95")
    return f"{d['mean']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def tables(agg, per_run):
    flat_same = all(
        r["selectors"]["Flat-K"][k] == r["selectors"]["Ontology-K"][k]
        for r in per_run if "error" not in r
        for k in ("availability", "false_normal_s", "switches"))
    out = [f"Flat-K identical to Ontology-K on every run and metric: **{flat_same}**\n"]
    for title, key, f in (("Availability (output with error <= 2 m)", "availability", "{:.3f}"),
                          ("False-normal output time per run [s]", "false_normal_s", "{:.1f}"),
                          ("No-valid-nav time per run [s]", "no_valid_nav_s", "{:.1f}"),
                          ("Switches per run", "switches", "{:.1f}")):
        out.append(f"\n### {title}\n")
        out.append("| scenario | runs | " + " | ".join(SELECTORS) + " |")
        out.append("|---|---|" + "---|" * len(SELECTORS))
        for sid, s in agg.items():
            row = [NAMES.get(sid, sid), str(s["runs"])]
            for name in SELECTORS:
                v = s["selectors"].get(name, {}).get(key)
                row.append("-" if v is None else f.format(v))
            out.append("| " + " | ".join(row) + " |")
    out.append("\n### Paired difference comparator - Ontology-K, availability (mean, run-level bootstrap 95 % CI)\n")
    comps = [s for s in SELECTORS if s != "Ontology-K"]
    out.append("| scenario | " + " | ".join(comps) + " |")
    out.append("|---|" + "---|" * len(comps))
    for sid, s in agg.items():
        out.append("| " + " | ".join([NAMES.get(sid, sid)] + [
            fmt_ci(s["selectors"][c].get("availability_minus_ontology")) for c in comps]) + " |")
    out.append("\n### Paired difference comparator - Ontology-K, false-normal time [s]\n")
    out.append("| scenario | " + " | ".join(comps) + " |")
    out.append("|---|" + "---|" * len(comps))
    for sid, s in agg.items():
        out.append("| " + " | ".join([NAMES.get(sid, sid)] + [
            fmt_ci(s["selectors"][c].get("false_normal_s_minus_ontology")) for c in comps]) + " |")
    return "\n".join(out) + "\n"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def fig_availability(agg, path):
    sids = list(agg)
    m = np.array([[agg[s]["selectors"][n]["availability"] for n in SELECTORS] for s in sids])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    lo = 0.7
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(sids) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(np.clip(m, lo, 1.0), cmap=cmap, vmin=lo, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            dark = (np.clip(m[i, j], lo, 1) - lo) / (1 - lo) > 0.55
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)
    # 2 px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(SELECTORS)), SELECTORS)
    ax.set_yticks(range(len(sids)), [NAMES.get(s, s) for s in sids])
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title("Availability (share of time with output error <= 2 m); colour clipped at 0.7",
                 fontsize=8.5, color=INK, pad=22, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_false_normal(agg, path):
    sids = list(agg)
    names = list(SERIES)
    x = np.arange(len(sids))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 2.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, n in enumerate(names):
        v = [agg[s]["selectors"][n]["false_normal_s"] for s in sids]
        ax.bar(x + (k - 1) * w, v, w * 0.9, color=SERIES[n], label=n, linewidth=0, zorder=2)
    ax.set_xticks(x, [s for s in sids], fontsize=8)
    ax.set_ylabel("false-normal time per run [s]", fontsize=8, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    _style(ax)
    top = max(agg[s]["selectors"][n]["false_normal_s"] for s in sids for n in names)
    ax.set_ylim(0, max(1.0, top) * 1.3)          # headroom so the legend never covers a bar
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", labelcolor=INK)
    ax.set_title("Time the output was declared valid while its error exceeded 2 m (run mean)",
                 fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--g6", default=None, help="results/g6/<dir> (default: results/g6/<set>)")
    a = ap.parse_args()
    src = (Path(a.g6).resolve() if a.g6 else ROOT / "results/g6" / a.set)
    agg = json.loads((src / "aggregate.json").read_text())
    per_run = json.loads((src / "per_run.json").read_text())
    out = ROOT / "results/g9" / src.name
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# G9 report: {src.name}\n", f"source: `{src.relative_to(ROOT)}`; failed runs: "
          f"{len(agg.get('failed_runs', []))}\n", tables(agg["scenarios"], per_run)]
    md.append("\n![availability](fig_availability.png)\n\n![false normal](fig_false_normal.png)\n")
    (out / "report.md").write_text("".join(md))
    fig_availability(agg["scenarios"], out / "fig_availability.png")
    fig_false_normal(agg["scenarios"], out / "fig_false_normal.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
