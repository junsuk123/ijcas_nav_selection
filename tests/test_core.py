"""Unit tests of the pure-python parts (no Isaac, no ROS)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac"))

from ijcas.baselines import EARTH_RADIUS, InsDr, InsState, lla_to_enu  # noqa: E402
from ijcas.evaluation import fit_4dof  # noqa: E402
from ijcas_sim.degrade import DelayQueue, degrade_image, degrade_scan  # noqa: E402
from ijcas_sim.faults import FaultEvent, FaultSchedule  # noqa: E402
from ijcas_sim.trajectory import GRAVITY, LoopTrajectory  # noqa: E402


def pegasus_reprojection(position, origin_lat, origin_long):
    """Verbatim formula of pegasus geo_mag_utils.reprojection (radians in/out)."""
    x_rad = position[1] / EARTH_RADIUS
    y_rad = position[0] / EARTH_RADIUS
    c = math.sqrt(x_rad * x_rad + y_rad * y_rad)
    if c == 0.0:
        return origin_lat, origin_long
    lat = math.asin(math.cos(c) * math.sin(origin_lat) + x_rad * math.sin(c) * math.cos(origin_lat) / c)
    lon = origin_long + math.atan2(y_rad * math.sin(c), c * math.cos(origin_lat) * math.cos(c)
                                   - x_rad * math.sin(origin_lat) * math.sin(c))
    return lat, lon


def test_lla_inverse_matches_pegasus_projection():
    origin = {"lat": 38.736832, "lon": -9.137977, "alt": 90.0}
    for p in ([0, 0, 0], [35.2, -12.5, 4.0], [-80.0, 60.0, 10.0]):
        lat, lon = pegasus_reprojection(p, math.radians(origin["lat"]), math.radians(origin["lon"]))
        enu = lla_to_enu(math.degrees(lat), math.degrees(lon), origin["alt"] + p[2], origin)[0]
        assert np.allclose(enu, p, atol=1e-6)


def test_fault_schedule_validation_and_activity():
    s = FaultSchedule.from_config([
        {"target": "gnss", "kind": "outage", "start_s": 20, "end_s": 50},
        {"target": "gnss", "kind": "position_bias", "start_s": 60,
         "params": {"vector_m": [5, 0, 0], "ramp_s": 10}}])
    assert s.first("gnss", "outage", 19.9) is None and s.first("gnss", "outage", 20.0)
    assert s.first("gnss", "outage", 50.0) is None
    e = s.first("gnss", "position_bias", 65.0)
    assert e.ramp(65.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        FaultEvent("gnss", "outage", 10, 5)
    with pytest.raises(ValueError):
        FaultEvent("camera", "brightness", 0, 1, {})
    with pytest.raises(ValueError):
        FaultEvent("radar", "outage", 0, 1)


def test_trajectory_is_smooth_and_flat_outputs_consistent():
    traj = LoopTrajectory([[0, 0, 3], [20, 0, 3], [20, 20, 5], [0, 20, 3]], speed=3.0,
                          hover_s=2.0, ramp_s=3.0, corner_radius_m=3.0)
    tab = traj.table(0.0, 6000, 0.005)
    v = np.linalg.norm(tab["velocity"], axis=1)
    assert v[:300].max() < 1e-9 and abs(v[-1] - 3.0) < 1e-6
    # finite-difference velocity of the position matches the analytic velocity
    fd = np.gradient(tab["position"], 0.005, axis=0)
    assert np.abs(fd[10:-10] - tab["velocity"][10:-10]).max() < 1e-3
    # thrust axis = normalised (a + g e_z)
    zb = R.from_quat(tab["quat_xyzw"]).apply([0, 0, 1])
    f = tab["acceleration"] + [0, 0, GRAVITY]
    assert np.abs(zb - f / np.linalg.norm(f, axis=1, keepdims=True)).max() < 1e-9
    assert np.abs(tab["body_rates"]).max() < 2.0


def test_ins_dr_exact_on_perfect_imu():
    traj = LoopTrajectory([[0, 0, 3], [20, 0, 3], [20, 20, 5], [0, 20, 3]], speed=3.0,
                          hover_s=0.0, ramp_s=1e-6, corner_radius_m=4.0)
    tab = traj.table(1.0, 2000, 0.005)
    rot = R.from_quat(tab["quat_xyzw"])
    f = rot.inv().apply(tab["acceleration"] + [0, 0, 9.80665])
    imu = np.column_stack([tab["t"], tab["body_rates"], f])
    ins = InsDr(InsState(tab["t"][0], tab["position"][0], tab["velocity"][0], rot[0],
                         np.zeros(3), np.zeros(3)))
    out = ins.run(imu, tab["t"][-1])
    err = np.linalg.norm(out[-1, 1:4] - tab["position"][-1])
    assert err < 0.05      # 10 s of perfect IMU at 200 Hz: integration error only


def test_fit_4dof_recovers_yaw_and_translation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3)) * 10
    rot = R.from_euler("z", 0.7)
    dst = rot.apply(src) + [3.0, -2.0, 1.0]
    r, t = fit_4dof(src, dst)
    assert np.allclose(r.apply(src) + t, dst, atol=1e-9)


def test_image_and_scan_degradations():
    rng = np.random.default_rng(0)
    img = np.full((48, 64), 200, np.uint8)
    s = FaultSchedule.from_config([
        {"target": "camera", "kind": "brightness", "start_s": 1, "end_s": 2,
         "params": {"scale": 0.5}},
        {"target": "camera", "kind": "occlusion", "start_s": 3, "end_s": 4,
         "params": {"fraction": 0.25, "side": "left"}},
        {"target": "lidar", "kind": "range_limit", "start_s": 0, "end_s": 9,
         "params": {"max_m": 5.0}}])
    assert degrade_image(img, "cam0", 0.5, s, rng)[0].mean() == 200
    assert degrade_image(img, "cam0", 1.5, s, rng)[0].mean() == 100
    occ = degrade_image(img, "cam0", 3.5, s, rng)[0]
    assert (occ[:, :16] == 0).all() and (occ[:, 16:] == 200).all()
    assert (degrade_image(img, "cam1", 3.5, s, rng)[0] == 200).all()
    xyz = np.array([[1.0, 0, 0], [10.0, 0, 0]], np.float32)
    (kept, _, _), _ = degrade_scan(xyz, np.zeros(2, np.float32), np.zeros(2, np.float32), 1.0, s,
                                   rng)
    assert len(kept) == 1
    q = DelayQueue()
    q.push(2.0, "a")
    q.push(1.0, "b")
    assert q.pop_due(1.5) == ["b"] and q.pop_due(2.0) == ["a"]


def test_outer_loop_tracks_reference_on_point_mass():
    from ijcas_sim.closed_loop import ENU2NED, FLU2FRD, G, OuterLoop
    traj = LoopTrajectory([[0, 0, 3], [20, 0, 3], [20, 20, 4], [0, 20, 3]], speed=2.0,
                          hover_s=8.0, ramp_s=4.0, corner_radius_m=4.0)
    ol = OuterLoop(traj, hover_thrust=0.5)
    p = np.array([1.0, -1.0, 0.15])           # starts off the reference, on the ground
    v = np.zeros(3)
    dt, errs = 0.01, []
    for k in range(int(60 / dt)):
        t = k * dt
        q_wxyz, thrust, info = ol.command(t, (p, None, v))
        r_ned = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        r_enu = ENU2NED.inv() * r_ned * FLU2FRD.inv()
        acc = r_enu.apply([0, 0, 1]) * thrust / 0.5 * G - [0, 0, G]
        v = v + acc * dt
        p = p + v * dt
        if t > 20:
            errs.append(np.linalg.norm(p - ol.reference(t)[0]))
    assert max(errs) < 0.3, max(errs)


def test_outer_loop_holds_then_aborts_without_navigation():
    from ijcas_sim.closed_loop import OuterLoop
    traj = LoopTrajectory([[0, 0, 3], [20, 0, 3], [20, 20, 4]], speed=2.0, hover_s=8.0, ramp_s=4.0)
    ol = OuterLoop(traj, hover_thrust=0.5, hold_s=3.0)
    _, th, info = ol.command(10.0, None)
    assert info["mode"] == "no_nav_hold" and th == 0.5
    _, th, info = ol.command(14.0, None)
    assert info["mode"] == "abort_descent" and th < 0.5 and ol.aborted


def test_alignment_maturity_bound_and_invalidation():
    from ijcas.handover import Alignment
    al = Alignment(window_s=30.0, min_path_m=10.0)
    yaw = 0.4
    rot = R.from_euler("z", yaw)
    for k in range(400):                       # 40 s of straight-ish flight, exact reference
        t = 0.1 * k
        p = np.array([2.0 * t, 0.5 * np.sin(0.2 * t), 3.0])
        al.observe_local(t, p, np.array([2.0, 0.1 * np.cos(0.2 * t), 0.0]))
        al.add_pair(t, p, rot.apply(p) + [5.0, -2.0, 0.0])
    assert al.valid and abs(al.yaw - yaw) < 1e-6 and al.residual_m < 1e-6
    assert al.mature(39.9) and not al.mature(al.established_at + 5.0)
    b0 = al.error_bound(0.02)
    for k in range(400, 500):                  # reference lost: frozen alignment, 20 m flown
        t = 0.1 * k
        al.observe_local(t, np.array([2.0 * t, 0.0, 3.0]), np.array([2.0, 0.0, 0.0]))
    assert al.path_since_update_m > 19.0 and al.error_bound(0.02) > b0 + 0.35
    al.invalidate(50.0, "VisualFeatures=Unusable")
    assert not al.valid and not al.mature(60.0) and al.invalidations
    # a jump that the engine's own velocity does not explain is a reset
    al2 = Alignment()
    al2.observe_local(0.0, np.zeros(3), np.zeros(3))
    assert al2.observe_local(0.05, np.array([3.0, 0, 0]), np.zeros(3))
