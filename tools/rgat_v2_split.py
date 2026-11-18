"""Pre-registered v2 validation split (design 13.2): reproduces VAL2_SEEDS of tools/rgat_run.py.

    python3 tools/rgat_v2_split.py

Pool = validation seeds 1-10 + first test seeds 101-130 (S09 excluded).  Among 40 000 random
8-seed subsets (numpy default_rng(2026)), take the lexicographic best of (fault/level variants not
covered in any scenario, -min count of an S10 IMU fault kind, -min count of a scene).  Uses only the
scenario definitions (ijcas/scenarios.py), never results.
"""
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ijcas.scenarios import TEST_SEEDS, VALIDATION_SEEDS, resolve  # noqa: E402

POOL = VALIDATION_SEEDS + TEST_SEEDS
SC = [f"S{i:02d}" for i in range(11) if i != 9]
KINDS = ("gyro_bias_step", "accel_bias_step", "time_offset", "saturation")


@lru_cache(maxsize=None)
def _r(sid, s):
    return resolve(sid, s)


def key(sid, s):
    r = _r(sid, s)
    v = dict(r["variant"])
    if sid == "S00":
        v["scene"] = r["scene"]
    return json.dumps(v, sort_keys=True)


def main():
    need = {sid: {key(sid, s) for s in POOL} for sid in SC}
    rng = np.random.default_rng(2026)
    best = None
    for _ in range(40000):
        val = sorted(rng.choice(POOL, 8, replace=False).tolist())
        miss = sum(len(need[sid] - {key(sid, s) for s in val}) for sid in SC)
        s10 = Counter(_r("S10", s)["variant"]["imu"] for s in val)
        scenes = Counter(_r(sid, s)["scene"] for sid in SC for s in val)
        sc = (miss, -min(s10.get(k, 0) for k in KINDS), -min(scenes.values()))
        if best is None or sc < best[0]:
            best = (sc, val)
    print("score", best[0], "VAL2_SEEDS", tuple(best[1]))


if __name__ == "__main__":
    main()
