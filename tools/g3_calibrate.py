"""G3 step 2: fix indicator thresholds and the quality model on the validation dataset.

    python3 tools/g3_calibrate.py results/g3/dataset.npz
    -> config/selector/thresholds_validated.yaml, results/g3/calibration_report.json

Indicator thresholds (only signals that carry information about their outcome label, AUC >= 0.6;
otherwise the indicator's thresholds are disabled and it is reported as uninformative):
  degraded threshold  maximises Youden's J (TPR - FPR) for "bad" on the side the indicator flags
  invalid threshold   most permissive value whose flagged set is >= 90 % bad (>= 50 samples);
                      if none exists the design value is kept
Outcome labels: VisualFeatures / OpenVINS <- rel_err_openvins > 0.3 m per 2 s, LidarStructure <-
rel_err_glim > 0.3 m per 2 s, GnssPosition <- GNSS-only error > alarm (2 m).
Inertial (shared IMU, R10) has two labels because its two levels mean different things:
  degraded  the injected IMU fault of the indicator's own kind (saturation for imu_saturation,
            time_offset for imu_clock_offset); ticks of other IMU fault kinds are excluded
            (scenario label; calibration only)
  invalid   that fault AND every IMU-dependent candidate's global error > alarm: 'Unusable' makes
            all of them unusable at once (R01/R10), which is only right if none of them survives.
            Without such evidence the invalid threshold is removed (the shared IMU is then at most
            'Limited' and the quality model chooses among the downgraded candidates).
Quality model: per candidate a logistic regression for P(global error > 1 m) on the candidate's
own signals (+ alignment age, fit residual and velocity disagreement with PX4 for local engines;
+ the GNSS-local consistency for PX4); missing values -> median + missing flag (never 0).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REL_BAD = 0.3
ALARM = 2.0
LABELS = {"VisualFeatures": ("rel_err_openvins", REL_BAD), "OpenVINS": ("rel_err_openvins", REL_BAD),
          "LidarStructure": ("rel_err_glim", REL_BAD), "GnssPosition": ("gnss_pos_err", ALARM),
          "Inertial": ("imu_fault", 0.5)}
CALIBRATED = {"gnss_local_consistency", "vis_features", "vis_coverage", "vis_brightness", "vis_stereo", "ov_tracking",
              "lidar_points", "lidar_geometry", "gnss_pos_vel_residual", "px4_pos_ratio",
              "imu_clock_offset", "imu_saturation"}
IMU_KIND = {"imu_saturation": "saturation", "imu_clock_offset": "time_offset"}
IMU_DEPENDENT = ("px4", "openvins", "glim")
QUALITY_FEATURES = {
    "px4": ["gnss.fix", "gnss.pos_vel_residual_m", "px4.pos_horiz_ratio", "px4.vel_ratio",
            "consistency.gnss_local_z"],
    "openvins": ["vision.features", "vision.grid_cells_ge10", "vision.mean",
                 "vision.stereo_matches", "openvins.tracked_features", "align_age_openvins",
                 "align.openvins.residual_m", "align.openvins.vel_disagree_mps"],
    "glim": ["lidar.points", "lidar.lambda_ratio", "align_age_glim", "align.glim.residual_m",
             "align.glim.vel_disagree_mps"],
}


def calibrate_indicator(ind, x, bad, bad_invalid=None):
    """bad: label for the degraded level; bad_invalid: separate label for the invalid level
    (None -> the same label, and the design invalid value is kept when nothing qualifies)."""
    ok = np.isfinite(x) & np.isfinite(bad)
    if bad_invalid is not None:
        bad_invalid = bad_invalid[ok].astype(bool)
    x, bad = x[ok], bad[ok].astype(bool)
    rep = {"id": ind["id"], "samples": int(len(x)), "bad_share": float(bad.mean()) if len(x) else None}
    if len(x) < 200 or bad.sum() < 30 or (~bad).sum() < 30:
        rep["status"] = "insufficient data: design value kept"
        return ind, rep
    below = "degraded_below" in ind or "invalid_below" in ind
    score = -x if below else x                      # larger score = more suspicious
    auc = roc_auc_score(bad, score)
    rep["auc"] = float(auc)
    if auc < 0.6:
        # the environment cue does not predict the engine's outcome on validation data:
        # its thresholds are disabled (it would only raise false alarms), freshness kept
        new = {k: v for k, v in ind.items() if not k.startswith(("degraded_", "invalid_"))}
        rep["status"] = "uninformative (AUC<0.6): thresholds disabled"
        return new, rep
    # candidates from each class separately: a rare fault class (1 % of the ticks) must not fall
    # outside a pooled quantile grid; rounded so float noise is not a distinct value
    grid = np.linspace(0.0, 1.0, 201)
    qs = np.unique(np.round(np.concatenate([np.quantile(x[bad], grid), np.quantile(x[~bad], grid)]), 6))
    # candidate thresholds halfway between observed values: a threshold equal to an observed value
    # (e.g. exactly 0 for a signal that is 0 when healthy) would flip on rounding noise
    qs = (qs[:-1] + qs[1:]) / 2 if len(qs) > 1 else qs
    best_j, best_t = -1, None
    inv_t = None
    for t in qs:
        flag = x < t if below else x > t
        tpr = (flag & bad).sum() / bad.sum()
        fpr = (flag & ~bad).sum() / (~bad).sum()
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
        bi = bad if bad_invalid is None else bad_invalid
        if flag.sum() >= 50 and (flag & bi).sum() / flag.sum() >= 0.9:
            if inv_t is None or (t > inv_t if below else t < inv_t):
                inv_t = t
    new = copy.deepcopy(ind)
    side = "below" if below else "above"
    new[f"degraded_{side}"] = float(best_t)
    if inv_t is None and bad_invalid is not None:
        new.pop(f"invalid_{side}", None)       # no evidence that nothing survives: at most Limited
        rep["invalid_removed"] = True
    if inv_t is not None:
        new[f"invalid_{side}"] = float(inv_t)
        if (below and new["invalid_below"] > new["degraded_below"]) or \
                (not below and new["invalid_above"] < new["degraded_above"]):
            new[f"degraded_{side}"] = new[f"invalid_{side}"]
    inv_key = f"invalid_{side}"
    if inv_key in new:                         # a degraded band never lies beyond the invalid one
        new[f"degraded_{side}"] = (max if below else min)(new[f"degraded_{side}"], new[inv_key])
    rep.update({"status": "calibrated", "youden_j": float(best_j),
                f"degraded_{side}": new[f"degraded_{side}"],
                f"invalid_{side}": new.get(f"invalid_{side}"),
                "design": {k: ind[k] for k in ind if k.startswith(("degraded", "invalid"))}})
    return new, rep


def fit_quality(d, cand):
    feats = QUALITY_FEATURES[cand]
    y_raw = d[f"glob_err_{cand}"].astype(float)
    ok = np.isfinite(y_raw)
    y = (y_raw[ok] > 1.0).astype(int)
    cols, med = [], {}
    for f in feats:
        v = d[f].astype(float)[ok] if f in d else np.full(ok.sum(), np.nan)
        v = np.where(np.isinf(v), np.nan, v)
        m = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        med[f] = m
        miss = ~np.isfinite(v)
        cols += [np.where(miss, m, v), miss.astype(float)]
    X = np.column_stack(cols)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    rep = {"candidate": cand, "samples": int(ok.sum()), "bad_share": float(y.mean()) if len(y) else None}
    if len(y) < 200 or y.sum() < 20 or (1 - y).sum() < 20:
        rep["status"] = "insufficient data: rule-based q kept"
        return None, rep
    lr = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)
    p = lr.predict_proba((X - mu) / sd)[:, 1]
    rep.update({"status": "fitted", "auc_train": float(roc_auc_score(y, p))})
    model = {"features": feats, "median": med, "mu": mu.tolist(), "sd": sd.tolist(),
             "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    return model, rep


def main(path):
    d = dict(np.load(path, allow_pickle=True))
    cfg = yaml.safe_load((ROOT / "config/selector/thresholds.yaml").read_text())
    report = {"dataset": path, "rows": int(len(d["t"])), "indicators": [], "quality": []}
    new_inds = []
    for ind in cfg["indicators"]:
        if ind["id"] not in CALIBRATED:
            new_inds.append(ind)
            continue
        target = ind["assesses"][0]
        x = d[ind["signal"]].astype(float) if ind["signal"] in d else np.full(len(d["t"]), np.nan)
        bad_invalid = None
        if target == "Inertial":
            kind = d["imu_fault_kind"].astype(str)
            own = kind == IMU_KIND[ind["id"]]
            bad = np.where(own, 1.0, np.where(kind == "", 0.0, np.nan))
            errs = np.column_stack([d[f"glob_err_{c}"].astype(float) for c in IMU_DEPENDENT])
            none_survives = np.all(~(errs <= ALARM), axis=1)     # NaN (no output) counts as failed
            bad_invalid = (own & none_survives).astype(float)
        else:
            lab_key, thr = LABELS[target]
            lab = d[lab_key].astype(float)
            bad = np.where(np.isfinite(lab), (lab > thr).astype(float), np.nan)
        new, rep = calibrate_indicator(ind, x, bad, bad_invalid)
        new_inds.append(new)
        report["indicators"].append(rep)
    out = copy.deepcopy(cfg)
    out["indicators"] = new_inds
    out["version"] = "validated"
    quality = {}
    for cand in QUALITY_FEATURES:
        model, rep = fit_quality(d, cand)
        report["quality"].append(rep)
        if model is not None:
            quality[cand] = model
    out["quality_model"] = quality
    (ROOT / "config/selector/thresholds_validated.yaml").write_text(
        "# Generated by tools/g3_calibrate.py from the validation set; frozen for the test set.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    (ROOT / "results/g3/calibration_report.json").write_text(json.dumps(report, indent=1))
    for r in report["indicators"]:
        print(f"{r['id']:24s} n={r['samples']:6d} bad={r.get('bad_share') or 0:.2f} "
              f"auc={r.get('auc', float('nan')):.2f} {r['status']} "
              f"{ {k: round(v, 3) for k, v in r.items() if k.startswith(('degraded', 'invalid')) and v is not None} }")
    for r in report["quality"]:
        print(f"quality {r['candidate']:9s} n={r['samples']} bad={r.get('bad_share')} {r['status']} "
              f"auc={r.get('auc_train')}")


if __name__ == "__main__":
    main(sys.argv[1])
