"""G1 diagnostic: is the IMU consistent with the forced motion?

Logs every physics step: the reference pose, Pegasus' state (position, attitude, velocity, body
rates) and the Pegasus IMU output (FRD).  Offline (tools/g1_check_imu.py) integrates the gyro and
compares the accelerometer with the reference specific force.

    isaac/run_isaac.sh isaac/probe_imu.py --seconds 30 --out results/g1/imu_probe
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=30.0)
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
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.sensors import IMU  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402

from ijcas_sim.trajectory import LoopTrajectory  # noqa: E402
from ijcas_sim.vehicle import ForcedMultirotor  # noqa: E402

PHYS_DT = 1.0 / 200.0
pg = PegasusInterface()
pg._world_settings = {"physics_dt": PHYS_DT, "stage_units_in_meters": 1.0, "rendering_dt": 0.05}
pg._world = World(**pg._world_settings)
world = pg.world
pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
while is_stage_loading():
    app.update()
traj = LoopTrajectory([[0, 0, 3], [15, 0, 3], [15, 15, 4], [0, 15, 3]], speed=3.0, hover_s=3.0,
                      ramp_s=4.0)
imu = IMU({"update_rate": 200.0})
cfg = MultirotorConfig()
cfg.sensors = [imu]
cfg.backends = []
cfg.graphical_sensors = []
s0 = traj.sample(0.0)
vehicle = ForcedMultirotor("/World/quadrotor", ROBOTS["Iris"], 0, list(s0.position),
                           list(s0.quat_xyzw), config=cfg, trajectory=traj)

rows = []


def log(dt):
    st = vehicle.state
    d = imu.state
    rows.append([vehicle.sim_time, *st.position, *st.attitude, *st.linear_velocity,
                 *st.angular_velocity, *d["angular_velocity"], *d["linear_acceleration"]])


world.add_physics_callback("ijcas_log", log)
world.reset()
omni.timeline.get_timeline_interface().play()
for _ in range(int(args.seconds / 0.05)):
    world.step(render=True)
np.savetxt(out / "imu_log.txt", np.asarray(rows),
           header="t p(3) q_xyzw(4) v_world(3) w_body_flu(3) imu_gyro_frd(3) imu_acc_frd(3)")
(out / "traj.json").write_text(json.dumps({"hover_s": 3.0, "ramp_s": 4.0, "speed": 3.0}))
print("PROBE rows", len(rows), flush=True)
omni.timeline.get_timeline_interface().stop()
app.close()
