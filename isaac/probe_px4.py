"""G0/G1 probe: PX4 SITL v1.14.3 as an estimator on a forced trajectory.

Checks (1) the forced trajectory is followed (tracking error), (2) PX4 boots with the IJCAS
parameters and accepts OFFBOARD attitude + ARM, so its land detector reports in-air while the
body moves without rotor thrust, (3) uXRCE-DDS exports /fmu/out/* on our ROS domain (checked by
tools/px4_dds_listen.py running beside this probe).  Writes ground truth for the comparison.

    isaac/run_isaac.sh isaac/probe_px4.py --seconds 90 --out results/g0/px4_probe
"""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=90.0)
parser.add_argument("--env", default="Default Environment")
parser.add_argument("--out", required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import (  # noqa: E402
    PX4MavlinkBackendConfig)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402

from ijcas_sim.faults import FaultSchedule  # noqa: E402
from ijcas_sim.px4 import Px4Keeper, Px4Params, exact_time_backend_class, make_launch_tool  # noqa: E402
from ijcas_sim.sensors import GatedBarometer, GatedMagnetometer, IjcasGnss, IjcasImu  # noqa: E402
from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402
from ijcas_sim.vehicle import ForcedMultirotor  # noqa: E402

PHYS_DT = 1.0 / 200.0
pg = PegasusInterface()
pg._world_settings = {"physics_dt": PHYS_DT, "stage_units_in_meters": 1.0, "rendering_dt": 0.05}
pg._world = World(**pg._world_settings)
world = pg.world
pg.load_environment(SIMULATION_ENVIRONMENTS[args.env])
while is_stage_loading():
    app.update()

square = [[0, 0, 3], [15, 0, 3], [15, 15, 4], [0, 15, 3]]
traj = LoopTrajectory(square, speed=3.0, hover_s=8.0, ramp_s=4.0)
PHYS_STEPS_PER_RENDER = 10

px4_dir = os.environ["PX4_DIR"]
np.random.seed(7)                       # Pegasus baro/mag noise uses the global RNG
rng = np.random.default_rng(7)
faults = FaultSchedule()
holder = {}
clock = lambda: holder["vehicle"].state_time  # noqa: E731
imu = IjcasImu(200.0, rng, faults, clock)
gnss = IjcasGnss(5.0, rng, faults, clock)
backend = exact_time_backend_class()(PX4MavlinkBackendConfig({
    "vehicle_id": 0, "px4_autolaunch": True, "px4_dir": px4_dir,
    "px4_vehicle_model": "gazebo-classic_iris", "update_rate": 200.0}), imu)
params = Px4Params()
backend.px4_tool = make_launch_tool(px4_dir, params)

cfg = MultirotorConfig()
cfg.sensors = [GatedBarometer(50.0, faults, clock), imu, GatedMagnetometer(50.0, faults, clock),
               gnss]
cfg.backends = [backend]
cfg.graphical_sensors = []
start = traj.sample(0.0)
vehicle = ForcedMultirotor("/World/quadrotor", ROBOTS["Iris"], 0, list(start.position),
                           list(start.quat_xyzw), config=cfg, trajectory=traj)
holder["vehicle"] = vehicle
n_bodies = vehicle.physics_disabled

world.reset()
backend.px4_tool.launch_px4()
keeper = Px4Keeper(instance=0, hover_thrust=0.6, start_after_s=3.0)
omni.timeline.get_timeline_interface().play()

gt = []
steps = int(args.seconds / 0.05)
w0 = time.time()
try:
    for k in range(steps):
        world.step(render=True)  # one render = 10 physics steps (50 ms)
        t = vehicle.sim_time
        if True:
            keeper.tick(t)
            st = vehicle.state
            gt.append([t, *map(float, st.position), *map(float, st.attitude),
                       *map(float, st.linear_velocity), backend._current_utime * 1e-6])
    wall = time.time() - w0
    param_check = {name: keeper.read_param(name) for name in
                   ("COM_RC_IN_MODE", "EKF2_EV_CTRL", "COM_DISARM_LAND", "UXRCE_DDS_DOM_ID", "UXRCE_DDS_SYNCT",
                    "SDLOG_MODE", "EKF2_GPS_CTRL", "EKF2_HGT_REF", "EKF2_NOAID_TOUT")}
    np.savetxt(out / "gt_enu.txt", np.asarray(gt),
               header="t px py pz qx qy qz qw vx vy vz px4_time_s (ENU world, body FLU)")
    summary = {"sim_s": round(vehicle.sim_time, 3), "wall_s": round(wall, 1),
               "rtf": round(vehicle.sim_time / wall, 3), "rigid_bodies_gravity_off": n_bodies,
               "vehicle_physics_disabled": vehicle.physics_disabled, "keeper": keeper.summary(),
               "params_readback": param_check, "ros_domain": os.environ.get("ROS_DOMAIN_ID"),
               "imu_samples": imu.count, "gnss_samples": gnss.count}
    (out / "isaac_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "px4_mavlink.json").write_text(json.dumps(keeper.records))
    print("PROBE " + json.dumps({k: summary[k] for k in ("sim_s", "rtf")}),
          flush=True)
finally:
    omni.timeline.get_timeline_interface().stop()
    backend.stop()
    if backend.px4_tool is not None:
        backend.px4_tool.kill_px4()
    app.close()
