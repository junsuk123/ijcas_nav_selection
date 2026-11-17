"""S00-S10 (Notion 04) resolved deterministically from (scenario id, seed).

Seeds 1-10 form the validation set (thresholds, quality calibration, hysteresis are fixed on it);
seeds 101-130 are the fixed test set.  The seed picks the scene, the degradation level and the
onset time; noise and fault randomness inside a run come from the same seed in the rig.  Every
run starts with >= 20 s of normal observation (hover 8 s + ramp 4 s + cruise), then degrades.

Levels follow the Notion starting values (S01 10/30/60 s, S02 3/5/10 m, S03 brightness
1.0/0.5/0.2, S07 20/50/100 ms); S10 magnitudes are pilot values to be confirmed on validation.
S04 levels were re-set for FAST-LIO2 on 2026-11-16 from engine-only pilots (see the S04 branch).
"""
from __future__ import annotations

import copy

import numpy as np

VALIDATION_SEEDS = list(range(1, 11))
TEST_SEEDS = list(range(101, 131))
DEV_SCENES = ("warehouse", "open_field", "corridor")
DURATION_S = 180.0


def _rng(sid: str, seed: int):
    return np.random.default_rng(int(seed) * 1000 + int(sid[1:]))


def _pick(seq, seed):
    return seq[seed % len(seq)]


def resolve(sid: str, seed: int) -> dict:
    rng = _rng(sid, seed)
    onset = round(float(rng.uniform(40.0, 70.0)), 1)
    base = {"id": sid, "seed": seed, "duration_s": DURATION_S, "hover_s": 8.0, "ramp_s": 4.0,
            "normal_s": 20.0, "gnss_rate_hz": 5.0, "faults": [], "variant": {}}
    s = copy.deepcopy(base)
    F = s["faults"]

    def gnss_outage(t0, dur):
        F.append({"target": "gnss", "kind": "outage", "start_s": t0, "end_s": t0 + dur})

    if sid == "S00":
        s["scene"] = _pick(DEV_SCENES, seed)
    elif sid == "S01":
        dur = _pick([10.0, 30.0, 60.0], seed // 3)
        s["scene"] = _pick(DEV_SCENES, seed)
        gnss_outage(onset, dur)
        s["variant"] = {"outage_s": dur}
    elif sid == "S02":
        mag = _pick([3.0, 5.0, 10.0], seed)
        vel_valid = (seed // 3) % 2 == 0
        s["scene"] = _pick(("open_field", "warehouse"), seed // 2)
        ang = float(rng.uniform(0, 2 * np.pi))
        vec = [round(float(mag * np.cos(ang)), 3), round(float(mag * np.sin(ang)), 3), 0.0]
        F.append({"target": "gnss", "kind": "position_bias", "start_s": onset,
                  "end_s": onset + 60.0, "params": {"vector_m": vec, "ramp_s": 5.0}})
        if not vel_valid:
            F.append({"target": "gnss", "kind": "velocity_invalid", "start_s": onset,
                      "end_s": onset + 60.0,
                      "params": {"sigma_mps": 0.5, "bias_mps": [round(float(0.3 * np.cos(ang)), 3),
                                                                round(float(0.3 * np.sin(ang)), 3), 0.0]}})
        s["variant"] = {"bias_m": mag, "velocity_valid": vel_valid}
    elif sid == "S03":
        kind = _pick(["brightness_0.5", "brightness_0.2", "occlusion", "blur"], seed)
        s["scene"] = _pick(DEV_SCENES, seed // 4)
        t1 = onset + 40.0
        if kind.startswith("brightness"):
            F.append({"target": "camera", "kind": "brightness", "start_s": onset, "end_s": t1,
                      "params": {"scale": float(kind.split("_")[1])}})
        elif kind == "occlusion":
            F.append({"target": "camera", "kind": "occlusion", "start_s": onset, "end_s": t1,
                      "params": {"fraction": 0.85, "side": "both"}})
        else:
            F.append({"target": "camera", "kind": "blur", "start_s": onset, "end_s": t1,
                      "params": {"kernel_px": 31}})
        s["variant"] = {"visual": kind}
    elif sid == "S04":
        # Levels re-set for FAST-LIO2 on 2026-11-16 (user decision: strengthen S04 only).  The
        # GLIM-era levels (corridor scene alone / point dropout 0.9 / range 4 m) left FAST-LIO2 at
        # <= 0.2 m.  Pilots (tools/lio_fault_probe.py, seeds 901-907, $IJCAS_DATA/fastlio_pilot),
        # rule fixed before the pilots: the mildest tested level at which FAST-LIO2 fails (error
        # > 2 m within the window or 30 s after it, or > 1 s without output):
        #   corridor + range 5 m -> 44 m divergence while still publishing (3 m: km-scale)
        #   warehouse range 3 m  -> 24 m and 29 s without output (2 m: 294 m)
        #   point dropout 0.99 / 0.997 / 0.999 -> <= 0.24 m, no failure -> full LiDAR blackout
        kind = _pick(["corridor_geometry", "corridor_geometry", "point_dropout", "range_limit"], seed)
        s["scene"] = "corridor" if kind == "corridor_geometry" else _pick(("warehouse", "open_field"), seed)
        if kind == "corridor_geometry":
            level = 5.0
            F.append({"target": "lidar", "kind": "range_limit", "start_s": onset,
                      "end_s": onset + 40, "params": {"max_m": level}})
        elif kind == "point_dropout":
            level = 1.0
            F.append({"target": "lidar", "kind": "dropout", "start_s": onset, "end_s": onset + 40,
                      "params": {"fraction": level}})
        else:
            level = 3.0
            F.append({"target": "lidar", "kind": "range_limit", "start_s": onset,
                      "end_s": onset + 40, "params": {"max_m": level}})
        s["variant"] = {"lidar": kind, "level": level}
    elif sid == "S05":
        s["scene"] = "warehouse"
        gnss_outage(onset, 50.0)
        vis = _pick(["brightness_0.2", "occlusion"], seed)
        if vis == "occlusion":
            F.append({"target": "camera", "kind": "occlusion", "start_s": onset + 10,
                      "end_s": onset + 45, "params": {"fraction": 0.85, "side": "both"}})
        else:
            F.append({"target": "camera", "kind": "brightness", "start_s": onset + 10,
                      "end_s": onset + 45, "params": {"scale": 0.2}})
        s["variant"] = {"visual": vis, "outage_s": 50.0}
    elif sid == "S06":
        s["scene"] = "corridor"
        gnss_outage(onset, 50.0)
        if seed % 2:
            F.append({"target": "lidar", "kind": "dropout", "start_s": onset + 10,
                      "end_s": onset + 45, "params": {"fraction": 0.9}})
        s["variant"] = {"outage_s": 50.0, "extra_lidar_dropout": bool(seed % 2)}
    elif sid == "S07":
        delay = _pick([0.02, 0.05, 0.10], seed)
        s["scene"] = _pick(DEV_SCENES, seed // 3)
        for tgt in ("camera", "lidar"):
            F.append({"target": tgt, "kind": "delay", "start_s": onset, "end_s": onset + 60,
                      "params": {"mean_s": delay, "jitter_s": delay / 4}})
            F.append({"target": tgt, "kind": "drop", "start_s": onset, "end_s": onset + 60,
                      "params": {"probability": 0.2}})
        F.append({"target": "gnss_ros", "kind": "delay", "start_s": onset, "end_s": onset + 60,
                  "params": {"mean_s": delay, "jitter_s": delay / 4}})
        s["variant"] = {"delay_s": delay}
    elif sid == "S08":
        s["scene"] = _pick(DEV_SCENES, seed)
        dur = 30.0
        gnss_outage(onset, dur)
        F.append({"target": "camera", "kind": "occlusion", "start_s": onset, "end_s": onset + dur,
                  "params": {"fraction": 1.0, "side": "both"}})
        F.append({"target": "lidar", "kind": "dropout", "start_s": onset, "end_s": onset + dur,
                  "params": {"fraction": 1.0}})
        lose_mag_baro = (seed // 3) % 2 == 1
        if lose_mag_baro:
            F.append({"target": "mag", "kind": "loss", "start_s": onset, "end_s": onset + dur})
            F.append({"target": "baro", "kind": "loss", "start_s": onset, "end_s": onset + dur})
        s["variant"] = {"outage_s": dur, "mag_baro_lost": lose_mag_baro}
    elif sid == "S09":
        # unseen geometry (office) and fault combinations never used for calibration
        s["scene"] = "office"
        combo = _pick(["gnss_bias+lidar_dropout", "gnss_outage+blur", "brightness+lidar_range"], seed)
        if combo == "gnss_bias+lidar_dropout":
            F.append({"target": "gnss", "kind": "position_bias", "start_s": onset,
                      "end_s": onset + 50, "params": {"vector_m": [4.0, -4.0, 0.0], "ramp_s": 8.0}})
            F.append({"target": "lidar", "kind": "dropout", "start_s": onset + 15,
                      "end_s": onset + 45, "params": {"fraction": 0.95}})
        elif combo == "gnss_outage+blur":
            gnss_outage(onset, 40.0)
            F.append({"target": "camera", "kind": "blur", "start_s": onset + 10,
                      "end_s": onset + 35, "params": {"kernel_px": 41}})
        else:
            F.append({"target": "camera", "kind": "brightness", "start_s": onset,
                      "end_s": onset + 40, "params": {"scale": 0.15}})
            F.append({"target": "lidar", "kind": "range_limit", "start_s": onset + 10,
                      "end_s": onset + 40, "params": {"max_m": 3.0}})
        s["variant"] = {"combo": combo}
    elif sid == "S10":
        kind = _pick(["gyro_bias_step", "accel_bias_step", "time_offset", "saturation"], seed)
        s["scene"] = _pick(DEV_SCENES, seed // 4)
        p = {"gyro_bias_step": {"rad_s": [0.0, 0.0, 0.05]},
             "accel_bias_step": {"mps2": [0.4, 0.0, 0.0]},
             "time_offset": {"offset_s": 0.03},
             "saturation": {"gyro_rad_s": 0.25, "accel_mps2": 12.0}}[kind]
        F.append({"target": "imu", "kind": kind, "start_s": onset, "end_s": onset + 50,
                  "params": p})
        s["variant"] = {"imu": kind}
    else:
        raise ValueError(sid)
    return s


ALL = [f"S{i:02d}" for i in range(11)]
