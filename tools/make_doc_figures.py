"""Documentation figures for the repository (docs/figures/*.png, *.svg; white background, English labels).

    python3 tools/make_doc_figures.py all [--captures DIR] [--renders DIR]
    python3 tools/make_doc_figures.py architecture | model | graph | ontology | sensors | scenes

architecture  system pipeline: simulation -> estimators -> ontology/policies -> fusion, offline loop
model         GRU-R-GAT fusion model at one tick (heads, inflation, weighted CI, loss)
graph         the 12-node fusion graph, edges taken from ijcas/fusion/graph.py
ontology      definitional part of ijcas/knowledge/ijcas_{nav,fusion}.ttl (graphviz dot)
sensors       camera / LiDAR frames grabbed from a running simulation (tools/capture_sensor_views.py)
scenes        third-person renders of the scenes (isaac/render_overview.py)

The diagrams describe what the code does; items that are not implemented yet are drawn dashed and
marked "planned".  No result numbers are drawn here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

# Okabe-Ito based: light fill, dark edge (one hue per role)
C = {"sim": ("#EAF2FB", "#0B5E9C"), "est": ("#FDF0DA", "#A86A00"), "know": ("#E3F4EC", "#00704F"),
     "fus": ("#F6E8F0", "#8E3F6C"), "off": ("#F2F2F2", "#4D4D4D"), "gt": ("#FDECE3", "#B5470A")}
INK, MUTED = "#1F1F1F", "#5A5A5A"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight",
                    pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def box(ax, x, y, w, h, title, lines=(), role="sim", dashed=False, tsize=10, bsize=8.6):
    """Rounded box; title on top, body lines below it (keep titles <= ~26 and lines <= ~34 characters
    per 27 units of width)."""
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec,
                                lw=1.2, ls=(0, (4, 3)) if dashed else "-", zorder=3))
    if lines:
        n = len(lines)
        ty = y + h / 2 + (n * 1.55) / 2 + 0.25
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=4)
        ax.text(x + w / 2, ty - 1.55, "\n".join(lines), ha="center", va="top", fontsize=bsize,
                color=MUTED, zorder=4, linespacing=1.32)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=4)
    if dashed:
        ax.text(x + w - 0.8, y + h - 0.7, "planned", ha="right", va="top", fontsize=7.8, style="italic",
                color=ec, zorder=4)


def panel(ax, x, y, w, h, title, role):
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=ec,
                                lw=1.5, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 0.25, y + h - 4.0), w - 0.5, 3.75,
                                boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none", zorder=1.5))
    ax.text(x + 1.3, y + h - 2.1, title, ha="left", va="center", fontsize=11, weight="bold", color=ec,
            zorder=2)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", rad=0.0, z=2, size=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=size, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1,
                                 shrinkB=1))


def route(ax, pts, color=INK, lw=1.3, ls="-", label=None, lseg=1, lsize=8.2):
    """Orthogonal poly-line with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)
    if label:
        (x0, y0), (x1, y1) = pts[lseg], pts[lseg + 1]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=lsize, color=color,
                zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))


def canvas(w, h):
    fig = plt.figure(figsize=(w / 10, h / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------------------------------------
def architecture():
    fig, ax = canvas(140, 84)
    ax.text(1, 81.6, "Ontology-constrained adaptive fusion of navigation solutions: system pipeline",
            fontsize=13.5, weight="bold", color=INK, va="center")
    PW, GAP, X0 = 29.5, 6.0, 1.0
    PX = [X0 + i * (PW + GAP) for i in range(4)]
    py, ph = 26.0, 52.5
    rows = [62.8, 51.2, 39.6, 28.0]                      # box bottoms, top row first
    bh = 9.8
    content = {
        0: ("1  Isaac Sim + Pegasus", "sim", [
            ("Scenes", ["warehouse · open field · corridor"]),
            ("Quadrotor + PX4 SITL", ["lockstep · offboard trajectory"]),
            ("Sensors", ["IMU 200 Hz · GNSS 5 Hz · baro, mag", "stereo cam 20 Hz · LiDAR 10 Hz"]),
            ("Fault injection  S00-S10", ["GNSS · camera · LiDAR · IMU", "topic delay and drop"])]),
        1: ("2  Estimators  (ROS 2)", "est", [
            ("PX4 EKF2  (GNSS/INS)", ["MAVLink ODOMETRY", "state + variances"]),
            ("OpenVINS  (stereo VIO)", ["state + pose / twist", "covariance (6x6)"]),
            ("FAST-LIO2  (LiDAR-inertial)", ["state + pose covariance", "+ velocity block (patch)"]),
            ("Context monitor", ["sensor diagnostics", "/ijcas/diag/*"])]),
        2: ("3  Knowledge & policies", "know", [
            ("Ontology  (OWL / TTL)", ["SOSA/SSN · PROV-O alignment", "definitional edges, CI rule"]),
            ("Node features", ["E conditions · S sensor quality", "N solution health"]),
            ("Eligibility policies", ["rules, not learned: IMU, align-", "ment, freshness, covariance"]),
            ("Causal frame alignment", ["VIO / LIO local frame", "to common ENU frame"])]),
        3: ("4  Adaptive fusion", "fus", [
            ("GRU-R-GAT", ["12 nodes · gated edges", "protected ontology edges"]),
            ("Weights and inflation", [r"$w_k$: softmax over eligible", r"$\gamma_{k,b} \geq 1$ per solution, block"]),
            ("Weighted CI", ["per block: position, velocity,", "attitude (SO(3) tangent space)"]),
            ("Fused state", [r"$x_f,\ P_f$, contributions $c_k$", "effective no. of contributors"])]),
    }
    for i, (title, role, items) in content.items():
        panel(ax, PX[i], py, PW, ph, title, role)
        for (t, ls), y in zip(items, rows):
            box(ax, PX[i] + 1.4, y, PW - 2.8, bh, t, ls, role)
    note_c = C["est"][1]
    for k in range(3):                                   # vertical flow inside the fusion panel
        arrow(ax, (PX[3] + PW / 2, rows[k]), (PX[3] + PW / 2, rows[k + 1] + bh), color=C["fus"][1])
    ym = (rows[1] + rows[2] + bh) / 2
    arrow(ax, (PX[0] + PW, ym), (PX[1], ym), color=C["sim"][1], lw=2.2, size=13)
    arrow(ax, (PX[1] + PW, ym), (PX[2], ym), color=note_c, lw=2.2, size=13)
    kx, fx = PX[2] + PW - 1.4, PX[3] + 1.4
    c = C["know"][1]
    arrow(ax, (kx, rows[0] + bh / 2), (fx, rows[0] + bh / 2), color=c, lw=1.6)
    arrow(ax, (kx, rows[1] + bh / 2), (fx, rows[0] + 2.0), color=c, lw=1.6)
    arrow(ax, (kx, rows[2] + bh / 2), (fx, rows[1] + bh / 2 - 1.0), color=c, lw=1.6)
    arrow(ax, (kx, rows[3] + bh / 2), (fx, rows[2] + bh / 2 - 1.0), color=c, lw=1.6)
    # offline lane (title at the bottom so that the arrows from above cross no text)
    LX, LW, LY, LH = 1.0, 103.0, 0.6, 22.0
    ax.add_patch(FancyBboxPatch((LX, LY), LW, LH, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=C["off"][1], lw=1.5, zorder=1))
    ax.text(LX + 1.4, LY + 2.0, "Offline experiment  ·  MATLAB R2025a starts and monitors every process "
            "(PIDs, exit codes, logs)", fontsize=10, weight="bold", color=C["off"][1], va="center")
    steps = [("Ground truth", ["from Isaac Sim;", "loss and metrics only"], "gt"),
             ("Recorder", ["states, covariances,", "diagnostics"], "off"),
             ("Dataset", ["20 Hz tensors", "per run"], "off"),
             ("Training", ["loss on the FUSED", "state; backprop via CI"], "off"),
             ("Evaluation", ["RMSE, RTE · NEES", "(separately) · figures"], "off")]
    SW = 15.5
    SX = [2.6, 31.5, 50.0, 68.5, 87.0]
    sy, sh = 8.0, 11.0
    for (t, ls, r), x in zip(steps, SX):
        box(ax, x, sy, SW, sh, t, ls, r, tsize=9.6, bsize=8.2)
    for i in (1, 2, 3):
        arrow(ax, (SX[i] + SW, sy + sh / 2), (SX[i + 1], sy + sh / 2), color=C["off"][1])
    gc = C["gt"][1]
    gy = 5.6
    route(ax, [(SX[0] + SW / 2, sy), (SX[0] + SW / 2, gy), (SX[3] + SW / 2, gy), (SX[3] + SW / 2, sy)],
          color=gc, ls=(0, (4, 2)))
    route(ax, [(SX[3] + SW / 2, gy), (SX[4] + SW / 2, gy), (SX[4] + SW / 2, sy)], color=gc, ls=(0, (4, 2)))
    arrow(ax, (SX[0] + SW / 2, py), (SX[0] + SW / 2, sy + sh), color=C["sim"][1], ls=(0, (3, 2)))
    arrow(ax, (SX[1] + SW / 2, py), (SX[1] + SW / 2, sy + sh), color=note_c, ls=(0, (3, 2)))
    fc = C["fus"][1]
    route(ax, [(SX[3] + SW / 2, sy + sh), (SX[3] + SW / 2, 24.3), (PX[3] + 6, 24.3), (PX[3] + 6, py)],
          color=fc, ls=(0, (4, 2)), label="learned parameters", lseg=1)
    box(ax, PX[3] + 1.4, 3.0, PW - 2.8, 15.5, "ROS 2 fusion node",
        ["/ijcas/fusion/{odometry,", "weights, health, diagnostics}", "online runtime, latency"], "fus",
        dashed=True)
    arrow(ax, (PX[3] + PW - 7, py), (PX[3] + PW - 7, 18.5), color=fc, ls=(0, (4, 2)))
    save(fig, "system_architecture")


# ------------------------------------------------------------------------------------------------
def model():
    fig, ax = canvas(140, 80)
    ax.text(1, 77.6, "GRU-R-GAT fusion weights, learned inflation and weighted CI at one tick t (20 Hz)",
            fontsize=13.5, weight="bold", color=INK, va="center")
    # A inputs
    panel(ax, 1, 37, 25, 37, "Node inputs", "know")
    for i, (t, ls) in enumerate([("E  x4", ["observation conditions"]), ("S  x4", ["sensor quality"]),
                                 ("N  x3", ["health, log tr $P_{k,b}$"]), ("F_nav", ["fusion node"])]):
        box(ax, 2.4, 61.5 - i * 6.6, 22.2, 5.8, t, ls, "know", tsize=9.2, bsize=8.2)
    ax.text(13.5, 39.3, "each feature: value, mask, age\nmissing: mask 0, never filled", ha="center",
            va="center", fontsize=7.9, color=C["know"][1], linespacing=1.3)
    # B R-GAT
    panel(ax, 30, 37, 40, 37, "Recurrent R-GAT  (12 nodes)", "fus")
    box(ax, 31.6, 57.5, 36.8, 11.0, "Node-type GRU memory",
        [r"$h_t = \mathrm{GRU}_{kind}(\mathrm{adapter}_{node}(x_t),\ h_{t-1})$"], "fus")
    box(ax, 31.6, 39.5, 36.8, 15.5, "Relational attention  x2 layers",
        ["per-relation messages, edge gates $z_{ij}$", "protected (always on): sensor > solution",
         "definitional (6), solution > fusion (3),", "self loops (12)"], "fus")
    arrow(ax, (50, 57.5), (50, 55.0), color=C["fus"][1])
    arrow(ax, (26, 55.5), (30, 55.5), color=C["know"][1], lw=1.8, size=12)
    # C head
    panel(ax, 74, 50, 24, 24, "Head (per solution k)", "fus")
    ax.text(86, 64.0, r"$[h_k\,;\,h_F] \rightarrow \mathrm{MLP} + c_k$", ha="center", va="center",
            fontsize=10, color=INK)
    ax.text(86, 58.6, r"logit $s_k$", ha="center", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(86, 54.0, r"inflation offset $u_{k,b}$", ha="center", va="center", fontsize=10, color=INK,
            weight="bold")
    arrow(ax, (70, 56), (74, 60), color=C["fus"][1], lw=1.8, size=12)
    # D weights, inflation (one panel, one arrow into the CI)
    panel(ax, 101, 35.5, 38, 38.5, "Head outputs", "fus")
    box(ax, 102.4, 57.0, 35.2, 11.5, "Information weights",
        [r"$w_k = \mathrm{softmax}_{k\ \mathrm{eligible}}(s_k)$", r"ineligible (policy):  $w_k = 0$"], "fus")
    box(ax, 102.4, 37.0, 35.2, 18.5, r"Learned inflation  $\gamma \geq 1$",
        [r"$\log\gamma_{k,b} = \mathrm{softplus}(\log\gamma^{norm}_{k,b} + u_{k,b})$",
         r"$\gamma^{norm}_{k,b} = \max_j\,\mathrm{tr}\,P_{j,b}\,/\,\mathrm{tr}\,P_{k,b}$  (reported)",
         r"$\gamma_{k,b}P_{k,b}$: scale changes, shape kept"], "fus", bsize=8.8)
    arrow(ax, (98, 59), (102.4, 62.5), color=C["fus"][1], lw=1.5)
    arrow(ax, (98, 54), (102.4, 50), color=C["fus"][1], lw=1.5)
    # E solutions
    box(ax, 74, 31.0, 24, 14.0, "Solutions k", ["GNSS/INS · VIO · LIO", "aligned $x_k = (p, v, R)$",
                                                 "$P_k$ as reported"], "est")
    arrow(ax, (98, 41), (102.4, 43), color=C["est"][1], lw=1.5)
    # F CI
    box(ax, 74, 4.0, 45, 22.0, "Weighted covariance intersection",
        ["per block b = position, velocity, attitude",
         r"$\Lambda_{k,b} = (\gamma_{k,b}P_{k,b})^{-1}, \quad \Lambda_{f,b} = \sum_k w_k\,\Lambda_{k,b}, \quad P_{f,b} = \Lambda_{f,b}^{-1}$",
         r"$x_{f,b} = P_{f,b} \sum_k w_k\,\Lambda_{k,b}\,x_{k,b}$",
         r"attitude: reference tangent space, $J_r^{-1}$, re-linearised"], "fus", bsize=8.9)
    arrow(ax, (86, 31.0), (88, 26.0), color=C["est"][1], lw=1.6)
    arrow(ax, (114, 35.5), (110, 26.0), color=C["fus"][1], lw=2.0, size=12)
    ax.text(114.2, 30.8, r"$w_k,\ \gamma_{k,b}$", fontsize=9.5, color=C["fus"][1], va="center")
    box(ax, 122, 4.0, 17, 22.0, "Output", [r"$x_f,\ P_f$", "contribution", r"$c_k = \mathrm{tr}(P_f\,w_k\Lambda_k)/3$",
                                          r"ENC $= 1/\sum_k c_k^2$"], "fus", bsize=8.4)
    arrow(ax, (119, 15), (122, 15), color=C["fus"][1], lw=1.6)
    # G training
    panel(ax, 1, 4, 69, 29, "Training  (offline; ground truth enters only here)", "gt")
    ax.text(3.0, 22.0,
            r"$L = \rho(|p_f - p_{GT}|/s_p) + \beta_v\,\rho(|v_f - v_{GT}|/s_v) + \beta_R\,\rho(|\mathrm{Log}(R_{GT}^{T}R_f)|/s_R)$"
            "\n" r"$\quad +\ \lambda_S\,|w_t - w_{t-1}|_1 + \lambda_E\,P(z_{ij} = 1)$       $\rho$: Huber",
            fontsize=10, color=INK, va="center", linespacing=1.9)
    ax.text(3.0, 11.0, "phase 1: 30 epochs, soft gates  ·  phase 2: 10 epochs, hard straight-through gates\n"
            "gradients reach head, R-GAT, GRU and adapters; the estimators are recorded outputs",
            fontsize=8.6, color=MUTED, va="center", linespacing=1.45)
    arrow(ax, (74, 15), (70, 15), color=C["gt"][1], lw=1.6)
    route(ax, [(45, 33), (45, 37)], color=C["gt"][1], ls=(0, (4, 2)))
    ax.text(46, 35.0, "backprop through CI", fontsize=8.2, color=C["gt"][1], va="center")
    save(fig, "fusion_model")


# ------------------------------------------------------------------------------------------------
def graph():
    from ijcas.fusion import graph as FG
    from ijcas.fusion.weights import feature_dims
    edges = FG.edges(True)
    dims = dict(zip(FG.NODES, feature_dims()))
    desc = {"E_gnss": "GNSS reception", "E_vis": "visual conditions", "E_geo": "LiDAR geometry",
            "E_mot": "motion", "S_gnss": "GNSS quality", "S_imu": "IMU quality", "S_cam": "camera quality",
            "S_lidar": "LiDAR quality", "N_gi": "PX4 EKF2 (GNSS/INS)", "N_vio": "OpenVINS (VIO)",
            "N_lio": "FAST-LIO2 (LIO)", "F_nav": "fusion node"}
    pos = {"E_gnss": (0, 4.5), "E_vis": (0, 1.5), "E_geo": (0, -1.5), "E_mot": (0, -4.5),
           "S_gnss": (6, 4.5), "S_imu": (6, 1.5), "S_cam": (6, -1.5), "S_lidar": (6, -4.5),
           "N_gi": (12.5, 3.6), "N_vio": (12.5, 0), "N_lio": (12.5, -3.6), "F_nav": (18.5, 0)}
    role = {"env": "sim", "sensor": "know", "nav": "est", "fusion": "fus"}
    style = {"env>sensor": ("#7B9CC0", 0.8, "-", 0.0, 0.55), "env>nav": ("#9DB5D1", 0.8, "-", -0.12, 0.5),
             "sensor>nav:aux": ("#7FB8A0", 1.0, (0, (3, 2)), 0.0, 0.8),
             "sensor>nav:def": ("#00704F", 2.8, "-", 0.0, 1.0), "nav>nav": ("#C08A2E", 1.0, "-", 0.35, 0.8),
             "nav>fusion": ("#8E3F6C", 3.0, "-", 0.0, 1.0)}
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(-2.2, 23.2)
    ax.set_ylim(-9.0, 7.6)
    ax.axis("off")
    W, H = 3.3, 1.55
    for e in edges:
        if e.rel == "self":
            continue
        col, lw, ls, rad, al = style[e.rel]
        p, q = np.array(pos[e.src]), np.array(pos[e.dst])
        a = FancyArrowPatch(p + [W / 2, 0], q - [W / 2, 0], arrowstyle="-|>", mutation_scale=9, color=col,
                            lw=lw, linestyle=ls, alpha=al, connectionstyle=f"arc3,rad={rad}",
                            zorder=3 if e.rel in ("sensor>nav:def", "nav>fusion") else 1)
        if e.rel == "nav>nav":
            a = FancyArrowPatch(p + [W / 2, 0], q + [W / 2, 0], arrowstyle="-|>", mutation_scale=9,
                                color=col, lw=lw, alpha=al,
                                connectionstyle=f"arc3,rad={-0.55 if p[1] > q[1] else -0.55}", zorder=1)
        ax.add_patch(a)
    for n, (x, y) in pos.items():
        fc, ec = C[role[FG.NODE_KIND[n]]]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=fc, ec=ec, lw=1.4, zorder=4))
        ax.text(x, y + 0.28, n, ha="center", va="center", fontsize=10.5, weight="bold", color=INK, zorder=5)
        ax.text(x, y - 0.33, desc[n], ha="center", va="center", fontsize=8.2, color=MUTED, zorder=5)
        ax.text(x - W / 2 + 0.12, y + H / 2 - 0.05, f"{dims[n] // 3}", ha="left", va="top", fontsize=6.5,
                color=ec, zorder=5)
    for x, t, r in ((0, "observation conditions (E)", "sim"), (6, "sensor quality (S)", "know"),
                    (12.5, "navigation solutions (N)", "est"), (18.5, "fusion (F)", "fus")):
        ax.text(x, 6.0, t, ha="center", fontsize=10.5, weight="bold", color=C[r][1])
    from collections import Counter
    n = Counter(e.rel for e in edges)
    prot = sum(v for k, v in n.items() if k in ("sensor>nav:def", "nav>fusion", "self"))
    items = [("sensor>nav:def", f"sensor > solution, definitional ({n['sensor>nav:def']}, protected)"),
             ("nav>fusion", f"solution > fusion ({n['nav>fusion']}, protected)"),
             ("sensor>nav:aux", f"sensor > solution, auxiliary ({n['sensor>nav:aux']}, gated)"),
             ("env>sensor", f"conditions > sensor ({n['env>sensor']}, gated)"),
             ("env>nav", f"conditions > solution ({n['env>nav']}, gated)"),
             ("nav>nav", f"solution > solution ({n['nav>nav']}, gated)")]
    lx, ly = 17.0, -4.6
    ax.add_patch(FancyBboxPatch((lx - 0.3, ly - 2.45), 6.15, 5.15, boxstyle="round,pad=0,rounding_size=0.25",
                                fc="white", ec="#BBBBBB", lw=0.8, zorder=6))
    for i, (r, t) in enumerate(items):
        col, lw, ls, _, al = style[r]
        yy = ly + 2.25 - i * 0.62
        ax.plot([lx, lx + 0.9], [yy, yy], color=col, lw=lw, ls=ls, alpha=al, zorder=7)
        ax.text(lx + 1.1, yy, t, va="center", fontsize=7.6, color=INK, zorder=7)
    ax.text(lx, ly - 1.85, f"+ self loops on all {n['self']} nodes (protected)\n"
            f"{len(edges)} edges: {prot} protected, {len(edges) - prot} gated",
            fontsize=7.6, color=MUTED, va="center", zorder=7, linespacing=1.35)
    ax.text(-2.1, -8.15, "Edges and protected relations are read from ijcas/fusion/graph.py and the ontology "
            "(definitional sensor > solution pairs). Number in each node corner: input features "
            "(each as value, mask, age).\nE nodes hold sensor-derived observation conditions, not the true "
            "environment. Ablation 'no ontology': definitional and fusion edges become gated like the rest.",
            fontsize=8, color=MUTED, va="center", linespacing=1.4)
    ax.text(-2.1, 7.35, "Fusion graph of the recurrent R-GAT  (12 nodes, candidate edges)", fontsize=13.5,
            weight="bold", color=INK, va="center")
    save(fig, "rgat_graph")


# ------------------------------------------------------------------------------------------------
def ontology():
    from rdflib import RDF, Namespace
    from ijcas.fusion import ontology as O
    g = O.graph()
    NAV = Namespace("https://incsl.sejong.ac.kr/ijcas/nav#")
    FUS = Namespace("https://incsl.sejong.ac.kr/ijcas/fusion#")
    L = lambda u: str(u).split("#")[-1]  # noqa: E731
    sensors = sorted({L(s) for s in g.subjects(RDF.type, NAV.Sensor)})
    caps = sorted({L(s) for s in g.subjects(RDF.type, NAV.MeasurementCapability)})
    envs = sorted({L(s) for s in g.subjects(RDF.type, NAV.EnvironmentContext)})
    ests = ["PX4_EKF2", "OpenVINS", "FAST_LIO", "INS_DR"]
    sols = {L(o): L(s) for s, o in g.subject_objects(FUS.producesEstimate)}          # sol -> est
    est_cls = {e: L(next(g.objects(NAV[e], RDF.type))) for e in ests}
    label = {e: str(next(g.objects(NAV[e], Namespace("http://www.w3.org/2000/01/rdf-schema#").label), e))
             for e in ests}
    font = 'fontname="DejaVu Sans"'
    D = ['digraph G {', f'graph [rankdir=LR, splines=true, nodesep=0.18, ranksep=0.9, bgcolor="white", {font}, '
         'fontsize=11, pad=0.25];', f'node [shape=box, style="rounded,filled", {font}, fontsize=10, penwidth=1.2];',
         f'edge [{font}, fontsize=8.5, color="#555555", arrowsize=0.7];',
         'labelloc=t; labeljust=l; fontsize=15; label=<<b>Operational ontology (ijcas_nav.ttl + ijcas_fusion.ttl): '
         'definitional knowledge behind the fusion graph</b><br/> >;']

    def cluster(name, title, nodes, fc, ec, lab=lambda n: n, loc="t"):
        D.append(f'subgraph cluster_{name} {{ label=<<b>{title}</b>>; fontcolor="{ec}"; color="{ec}"; '
                 f'style="rounded"; penwidth=1.2; labelloc={loc};')
        for n in nodes:
            D.append(f'"{n}" [label=<{lab(n)}>, fillcolor="{fc}", color="{ec}"];')
        D.append("}")
    groups = {}
    for e in g.subjects(RDF.type, NAV.EnvironmentContext):
        key = tuple(sorted(L(c) for c in g.objects(e, NAV.degrades)))
        groups.setdefault(key, []).append(L(e))
    env_nodes = {f"env{i}": (sorted(v), k) for i, (k, v) in enumerate(sorted(groups.items()))}
    for nid, (members, _) in env_nodes.items():            # no cluster: dot places each next to its target
        D.append(f'"{nid}" [label=<<font point-size="8" color="#B5470A">EnvironmentContext</font><br/>'
                 f'{"<br/>".join(members)}>, fillcolor="{C["gt"][0]}", color="{C["gt"][1]}"];')
    for n in sensors:
        D.append(f'"{n}" [label=<<font point-size="8" color="{C["sim"][1]}">Sensor</font><br/>{n}>, '
                 f'fillcolor="{C["sim"][0]}", color="{C["sim"][1]}"];')
    cluster("cap", "MeasurementCapability", caps, *C["know"], loc="b")
    cluster("est", "Estimator", ests, *C["est"],
            lab=lambda e: f'<b>{e}</b><br/><font point-size="8.5" color="#5A5A5A">{label[e]}<br/>:{est_cls[e]}</font>')
    sol_lab = {}
    for s, e in sols.items():
        ali = str(next(g.objects(FUS[s], FUS.requiresAlignment), "false")).lower() == "true"
        covs = [L(o) for o in g.objects(FUS[s], FUS.hasUncertainty)]
        rep = sum(str(next(g.objects(FUS[c], FUS.reportedByEstimator), "false")).lower() == "true" for c in covs)
        sol_lab[s] = (f'<b>{s}</b><br/><font point-size="8.5" color="#5A5A5A">reported covariance blocks: {rep}/3'
                      + ('<br/>requires alignment' if ali else '') + '</font>')
    cluster("sol", "NavigationSolution  (ssn:Output)", list(sols), *C["fus"], lab=lambda s: sol_lab[s])
    meth = str(next(g.objects(FUS.nav_fusion, FUS.requiresMethodUnderSharedSource)))
    D.append(f'"nav_fusion" [label=<<b>nav_fusion</b><br/><font point-size="8.5" color="#5A5A5A">'
             f':FusionProcess (sosa:Procedure)<br/>shared source &#8658; {meth}</font>>, '
             f'fillcolor="{C["fus"][0]}", color="{C["fus"][1]}"];')
    D.append(f'"sol_fused" [label=<<b>sol_fused</b><br/><font point-size="8.5" color="#5A5A5A">'
             f':FusedNavigationSolution</font>>, fillcolor="{C["fus"][0]}", color="{C["fus"][1]}"];')
    for s in g.subjects(RDF.type, NAV.Sensor):
        for c in g.objects(s, NAV.providesMeasurement):
            D.append(f'"{L(s)}" -> "{L(c)}" [color="{C["sim"][1]}"];')
    for nid, (_, targets) in env_nodes.items():
        for c in targets:
            D.append(f'"{nid}" -> "{c}" [color="#B5470A", style=dashed, arrowhead=tee];')
    for e in ests:
        for c in g.objects(NAV[e], NAV.requiresEssential):
            D.append(f'"{L(c)}" -> "{e}" [dir=back, color="{C["est"][1]}", penwidth=1.6];')
        for c in g.objects(NAV[e], NAV.usesAiding):
            D.append(f'"{L(c)}" -> "{e}" [dir=back, color="{C["est"][1]}", style=dashed];')
    for s, e in sols.items():
        D.append(f'"{e}" -> "{s}" [color="{C["fus"][1]}"];')
        D.append(f'"{s}" -> "nav_fusion" [color="{C["fus"][1]}", penwidth=1.8];')
    D.append(f'"nav_fusion" -> "sol_fused" [color="{C["fus"][1]}", penwidth=1.8];')
    shared = O.shared_sources()
    pairs = sorted({tuple(sorted(k)) for k in shared})
    sid = {g.value(FUS[s], FUS.solutionId).toPython(): s for s in sols}
    for a, b in pairs:
        D.append(f'"{sid[a]}" -> "{sid[b]}" [dir=both, arrowhead=none, arrowtail=none, style=dotted, '
                 f'color="#8E3F6C", constraint=false];')
    D.append(f'legend [shape=plaintext, style="", label=<<table border="0" cellborder="0" cellspacing="2">'
             f'<tr><td align="left"><font color="{C["sim"][1]}">&#8594;</font> providesMeasurement</td></tr>'
             f'<tr><td align="left"><font color="#B5470A">- -&#8867;</font> degrades</td></tr>'
             f'<tr><td align="left"><font color="{C["est"][1]}">&#8592;</font> requiresEssential (bold) / usesAiding (dashed)</td></tr>'
             f'<tr><td align="left"><font color="{C["fus"][1]}">&#8594;</font> producesEstimate, contributesToFusion</td></tr>'
             f'<tr><td align="left"><font color="#8E3F6C">&#183;&#183;&#183;</font> sharesMeasurementSource (Inertial, derived)</td></tr>'
             f'<tr><td align="left"><font color="#5A5A5A">INS_DR: separate limited fallback, not fused</font></td></tr>'
             f'</table>>];')
    D.append('{rank=same; "nav_fusion"; legend;}')
    D.append("}")
    src = "\n".join(D)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ontology.dot").write_text(src)
    for ext in ("png", "svg"):
        args = ["dot", f"-T{ext}", "-o", str(OUT / f"ontology.{ext}")] + (["-Gdpi=170"] if ext == "png" else [])
        subprocess.run(args, input=src.encode(), check=True)
    print("wrote", OUT / "ontology.png")


# ------------------------------------------------------------------------------------------------
def _load_caps(d: Path):
    caps = []
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        sc = m.get("scenario", m)
        caps.append({"path": p, "run": p.stem.rsplit("_", 1)[0] if p.stem.count("_") >= 2 else p.stem,
                     "tag": p.stem.split("_", 2)[2] if p.stem.count("_") >= 2 else "",
                     "scene": sc.get("scene", "?"), "sid": sc.get("id", "?"), "z": {k: z[k] for k in z.files}})
    return caps


def _lidar_bev(ax, pts, rmax=25.0):
    xy = pts[:, :2]
    r = np.linalg.norm(xy, axis=1)
    k = r < rmax
    ax.scatter(xy[k, 0], xy[k, 1], c=pts[k, 2], s=0.25, cmap="viridis", vmin=-1.5, vmax=4.0, rasterized=True,
               linewidths=0)
    ax.scatter([0], [0], marker="^", s=40, c="#B5470A", zorder=3)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#BBBBBB")


def sensors(cap_dir: Path, picks: list[str] | None = None):
    caps = _load_caps(cap_dir)
    if not caps:
        print("no captures in", cap_dir)
        return
    if picks:
        caps = [c for p in picks for c in caps if c["path"].stem == p]
    n = len(caps)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.05 * n), gridspec_kw={"width_ratios": [1.33, 1.33, 1.0]})
    axes = np.atleast_2d(axes)
    for i, c in enumerate(caps):
        z = c["z"]
        for j, key in enumerate(("cam0", "cam1")):
            ax = axes[i, j]
            if key in z:
                ax.imshow(z[key], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"stereo camera {'left' if j == 0 else 'right'}  (640x480, 20 Hz)", fontsize=10)
        ax = axes[i, 2]
        if "lidar" in z:
            _lidar_bev(ax, z["lidar"])
        if i == 0:
            ax.set_title("3D LiDAR scan, top view  (colour = height)", fontsize=10)
        tag = c["tag"].replace("_", " ")
        axes[i, 0].set_ylabel(f"{c['sid']}  {c['scene'].replace('_', ' ')}\n{tag}", fontsize=10, weight="bold")
    fig.suptitle("Isaac Sim sensor outputs as seen by the estimators  (nominal flight and injected faults)",
                 fontsize=13, weight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "isaac_sensor_views")


def scenes(render_dir: Path):
    imgs = sorted(render_dir.glob("*.png"))
    if not imgs:
        print("no renders in", render_dir)
        return
    n = len(imgs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, imgs):
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("_", " "), fontsize=11, weight="bold")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Isaac Sim scenes used in the experiments", fontsize=13, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "isaac_scenes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("all", "architecture", "model", "graph", "ontology", "sensors", "scenes"))
    ap.add_argument("--captures", default="")
    ap.add_argument("--picks", default="")
    ap.add_argument("--renders", default="")
    a = ap.parse_args()
    if a.what in ("all", "architecture"):
        architecture()
    if a.what in ("all", "model"):
        model()
    if a.what in ("all", "graph"):
        graph()
    if a.what in ("all", "ontology"):
        ontology()
    if a.what in ("all", "sensors") and a.captures:
        sensors(Path(a.captures), [p for p in a.picks.split(",") if p] or None)
    if a.what in ("all", "scenes") and a.renders:
        scenes(Path(a.renders))


if __name__ == "__main__":
    main()
