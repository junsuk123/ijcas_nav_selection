"""Ray-traced GNSS model (isaac/ijcas_sim/gnss_sky.py) against analytic box worlds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
from ijcas_sim.gnss_sky import (CHIP_CA, CONSTELLATIONS, GnssErrorModel, SkyModel,  # noqa: E402
                                constellation_ecef, dll_multipath_error, geodetic_to_ecef)

LAT, LON = 37.55, 127.0


class BoxWorld:
    """Closest ray hit against axis-aligned boxes: (hit, distance, point, normal)."""

    def __init__(self, boxes):
        self.boxes = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in boxes]

    def __call__(self, o, d, max_dist):
        o, d = np.asarray(o, float), np.asarray(d, float)
        best = (False, np.inf, None, None)
        for lo, hi in self.boxes:
            tn, tf, axis = -np.inf, np.inf, -1
            ok = True
            for k in range(3):
                if abs(d[k]) < 1e-12:
                    if o[k] < lo[k] or o[k] > hi[k]:
                        ok = False
                        break
                    continue
                t1, t2 = (lo[k] - o[k]) / d[k], (hi[k] - o[k]) / d[k]
                a, b = min(t1, t2), max(t1, t2)
                if a > tn:
                    tn, axis = a, k
                tf = min(tf, b)
            if not ok or tn > tf or tf < 0 or tn < 0 or tn > max_dist or axis < 0:
                continue
            if tn < best[1]:
                nrm = np.zeros(3)
                nrm[axis] = -np.sign(d[axis])
                best = (True, float(tn), o + tn * d, nrm)
        return best


def sky(world, seed=3, **kw):
    return SkyModel(seed=seed, lat_deg=LAT, lon_deg=LON, raycast=world, **kw)


def test_constellation_geometry():
    pos, ids = constellation_ecef(("gps",), 1234.0)
    assert len(ids) == 24
    r = np.linalg.norm(pos, axis=1)
    assert np.allclose(r, CONSTELLATIONS["gps"]["a_m"], rtol=1e-9)
    rng = np.linalg.norm(pos - geodetic_to_ecef(np.radians(LAT), np.radians(LON)), axis=1)
    s = sky(BoxWorld([]), constellations=("gps",), epoch_s=1234.0, extras=False)   # nominal 24 slots
    u, el = s.satellites(0.0)
    above = el > 0
    assert 6 <= above.sum() <= 14                       # typical nominal GPS sky at mid latitude
    assert rng[above].min() > 19.9e6 and rng[above].max() < 26.0e6


@pytest.mark.parametrize("names,lo,hi", [(("gps",), 4, 16), (("gps", "galileo"), 9, 28),
                                         (("gps", "galileo", "beidou"), 15, 40)])
def test_satellite_count_depends_on_constellations(names, lo, hi):
    counts = [sky(BoxWorld([]), constellations=names, epoch_s=e).assess(0.0, [0, 0, 5], [0, 0, 0])["used"]
              for e in np.linspace(0, 80000, 9)]
    assert lo <= min(counts) and max(counts) <= hi


def test_open_sky_solution():
    out = sky(BoxWorld([])).assess(0.0, [0, 0, 5], [1, 0, 0])
    assert out["fix"] and out["n_nlos"] == 0 and out["n_multipath"] == 0
    assert out["used"] == out["n_los"] == out["n_above_mask"]
    assert np.linalg.norm(out["pos_err"]) < 6.0 and out["pdop"] < 4.0


def test_ground_multipath_excess_path():
    world = BoxWorld([([-500, -500, -1], [500, 500, 0])])
    s = sky(world)
    o = np.array([0.0, 0.0, 5.0 + s.model.antenna_up_m])
    u_all, el = s.satellites(0.0)
    i = int(np.argmax(el > np.radians(20)))
    delta, loss, nrm = s._reflection(o, u_all[i])
    assert delta == pytest.approx(2 * o[2] * np.sin(el[i]), rel=1e-6)
    assert loss == s.model.loss_ground_db and nrm[2] == pytest.approx(1.0)
    out = s.assess(0.0, [0, 0, 5], [0, 0, 0])
    assert out["n_multipath"] == out["n_los"] > 0 and out["fix"]


def test_urban_canyon_nlos_and_dop():
    # street along x, 16 m wide, 60 m tall facades on both sides, ground
    world = BoxWorld([([-500, -500, -1], [500, 500, 0]), ([-500, 8, 0], [500, 30, 60]),
                      ([-500, -30, 0], [500, -8, 60])])
    open_ = BoxWorld([([-500, -500, -1], [500, 500, 0])])
    nlos, errs, errs_open, used, used_open = 0, [], [], [], []
    for e in np.linspace(0, 80000, 12):
        a = sky(world, constellations=("gps", "galileo"), epoch_s=e).assess(0.0, [0, 0, 6], [3, 0, 0])
        b = sky(open_, constellations=("gps", "galileo"), epoch_s=e).assess(0.0, [0, 0, 6], [3, 0, 0])
        nlos += a["n_nlos"]
        used.append(a["used"])
        used_open.append(b["used"])
        if a["fix"]:
            errs.append(np.linalg.norm(a["pos_err"][:2]))
        errs_open.append(np.linalg.norm(b["pos_err"][:2]))
    assert nlos > 0
    assert np.mean(used) < np.mean(used_open)
    assert np.median(errs) > np.median(errs_open)


def test_nlos_excess_path_formula():
    # one facade behind the receiver (wall plane y = -10, normal +y): satellite at az 90 deg (north) blocked
    # by a wall in front; its reflection off the back wall has delta = 2 D (n.u)
    world = BoxWorld([([-500, -20, 0], [500, -10, 80])])
    s = sky(world)
    o = np.array([0.0, 0.0, 6.0])
    el = np.radians(30.0)
    u = np.array([0.0, np.cos(el), np.sin(el)])
    delta, loss, nrm = s._reflection(o, u)
    assert np.allclose(nrm, [0, 1, 0])
    assert delta == pytest.approx(2 * 10.0 * np.cos(el), rel=1e-6)
    assert loss == s.model.loss_facade_db


def test_tunnel_has_no_fix():
    world = BoxWorld([([-200, -6, -1], [200, 6, 0]), ([-200, 5.5, 0], [200, 6.5, 7]),
                      ([-200, -6.5, 0], [200, -5.5, 7]), ([-200, -6.5, 7], [200, 6.5, 8])])
    out = sky(world).assess(0.0, [0, 0, 4], [8, 0, 0])
    assert out["used"] == 0 and not out["fix"]


def test_dll_envelope():
    d, chip = 0.1, CHIP_CA
    assert dll_multipath_error(2.0, 0.5, 0.0, d) == pytest.approx(0.5 * 2.0 / 1.5)
    assert dll_multipath_error(2.0, 0.5, np.pi, d) == pytest.approx(-0.5 * 2.0 / 0.5)
    plateau = dll_multipath_error(100.0, 0.5, 0.0, d)
    assert plateau == pytest.approx(0.5 * d * chip / 2 / 1.5)
    assert dll_multipath_error(chip * (1 + d / 2) + 1.0, 0.5, 0.0, d) == 0.0


def test_deterministic_per_seed():
    world = BoxWorld([([-500, -500, -1], [500, 500, 0]), ([-500, 8, 0], [500, 30, 40])])
    a = [sky(world, seed=9).assess(t, [t, 0, 5], [1, 0, 0])["pos_err"] for t in (0.0, 0.2, 0.4)]
    b = [sky(world, seed=9).assess(t, [t, 0, 5], [1, 0, 0])["pos_err"] for t in (0.0, 0.2, 0.4)]
    assert np.allclose(a, b)
    c = sky(world, seed=10)
    assert (c.constellations, c.epoch_s) != (sky(world, seed=9).constellations, sky(world, seed=9).epoch_s)


def test_cn0_pattern_and_nlos_threshold():
    s = sky(BoxWorld([]))
    m = s.model
    assert s.cn0_los(np.radians(90)) == pytest.approx(m.cn0_zenith_dbhz + s.rx_offset_db)
    assert 35.0 < s.cn0_los(np.radians(10)) < 40.5
    # a low satellite reflected off a facade is below the acquisition threshold, a high one is not
    assert s.cn0_los(np.radians(12)) - m.loss_facade_db < m.acquire_dbhz
    assert s.cn0_los(np.radians(60)) - m.loss_facade_db > m.acquire_dbhz


def test_reacquisition_after_tunnel():
    ground = ([-500, -500, -1], [500, 500, 0])
    tunnel = [ground, ([-200, 5.5, 0], [200, 6.5, 7]), ([-200, -6.5, 0], [200, -5.5, 7]),
              ([-200, -6.5, 7], [200, 6.5, 8])]
    world = BoxWorld([ground])
    s = sky(world, seed=5)
    for t in np.arange(0.0, 1.0, 0.2):                       # open sky at power-up: fix immediately
        assert s.assess(t, [0, 0, 4], [6, 0, 0])["fix"]
    world.boxes = BoxWorld(tunnel).boxes                      # 40 s inside the tunnel
    for t in np.arange(1.0, 41.0, 0.2):
        assert not s.assess(t, [0, 0, 4], [6, 0, 0])["fix"]
    world.boxes = BoxWorld([ground]).boxes                    # exit: satellites visible again
    first = s.assess(41.0, [0, 0, 4], [6, 0, 0])
    assert not first["fix"] and first["n_reacquiring"] >= 4
    fixes = [(t, s.assess(t, [0, 0, 4], [6, 0, 0])["fix"]) for t in np.arange(41.2, 47.0, 0.2)]
    t_fix = min(t for t, f in fixes if f)
    assert t_fix >= 41.0 + s.model.reacq_long_s - 1e-9


def test_unhealthy_and_spare_satellites_vary_by_seed():
    counts = {len(sky(BoxWorld([]), seed=k, constellations=("gps",)).ids) for k in range(20)}
    assert min(counts) >= 24 and max(counts) <= 31 and len(counts) > 3
    unhealthy = [int((~sky(BoxWorld([]), seed=k).healthy).sum()) for k in range(20)]
    assert set(unhealthy) <= {0, 1, 2} and len(set(unhealthy)) > 1
