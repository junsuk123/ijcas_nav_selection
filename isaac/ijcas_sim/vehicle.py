"""Pegasus multirotor that *is* the reference trajectory (Stage A, open loop).

PhysX is switched off for the vehicle (rigid bodies, articulation and collisions disabled).  Every
physics step ``update_state`` fills Pegasus' ``State`` from the analytic reference at the current
simulation time (position, attitude, world velocity, body rates, acceleration) and writes the same
pose to the vehicle's USD transform, so

* Pegasus sensors (IMU, GNSS, baro, mag) -> PX4 and our ROS publishers see exact kinematics,
* the cameras and the LiDAR, which are children of the body prim, render from the same pose,
* PX4's actuator outputs are ignored (no rotor forces exist).

Why not drive the PhysX body with teleports: Pegasus reads the attitude from USD, which PhysX only
writes back on render steps; with 200 Hz physics and 20 Hz rendering 91 % of the IMU samples were
computed with a stale attitude and the body-rate read-back was wrong (results/g1/imu_probe).
Isaac-only module.
"""
from __future__ import annotations

import numpy as np
from pegasus.simulator.logic.vehicles.multirotor import Multirotor
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from .trajectory import LoopTrajectory


def disable_vehicle_physics(stage, root_path: str) -> dict:
    counts = {"rigid_bodies": 0, "colliders": 0, "articulations": 0, "joints": 0}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            counts["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            counts["colliders"] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateArticulationEnabledAttr(False)
            counts["articulations"] += 1
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            counts["joints"] += 1
    return counts


class ForcedMultirotor(Multirotor):
    """Multirotor whose state is the reference trajectory; sensors and backends unchanged."""

    def __init__(self, *args, trajectory: LoopTrajectory, **kwargs):
        self.trajectory = trajectory
        self.sim_time = 0.0
        self.state_time = 0.0   # simulation time of the current Pegasus state (sensor stamp)
        self.prof_s = 0.0
        self._step = 0
        self._xform_op = None
        super().__init__(*args, **kwargs)
        self.physics_disabled = disable_vehicle_physics(self._world.stage, self._stage_prefix)
        root = self._world.stage.GetPrimAtPath(self._stage_prefix)
        xf = UsdGeom.Xformable(root)
        xf.ClearXformOpOrder()
        self._translate = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "ijcas")
        self._orient = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "ijcas")
        self._place(0.0)

    # Pegasus registers the vehicle as an articulation robot in the World scene; with PhysX off
    # there is no articulation to bind, so the scene's (re)initialisation hooks are no-ops.
    def initialize(self, physics_sim_view=None):
        pass

    def post_reset(self):
        pass

    def _place(self, t: float):
        s = self.trajectory.sample(t)
        x, y, z, w = (float(v) for v in s.quat_xyzw)
        self._translate.Set(Gf.Vec3d(*map(float, s.position)))
        self._orient.Set(Gf.Quatd(w, x, y, z))
        return s

    def update_state(self, dt: float):
        import time as _t
        _p = _t.perf_counter()
        self._update_state(dt)
        self.prof_s += _t.perf_counter() - _p

    def _row(self, dt: float):
        """Reference at the current step from a precomputed table (refilled every 2000 steps)."""
        tab = getattr(self, "_table", None)
        k = self._step - getattr(self, "_table_k0", 0)
        if tab is None or k >= len(tab["t"]):
            self._table_k0 = self._step
            self._table = tab = self.trajectory.table(self.sim_time, 2000, dt)
            k = 0
        return {name: tab[name][k] for name in tab}

    def _update_state(self, dt: float):
        self.state_time = self.sim_time
        r = self._row(dt)
        x, y, z, w = (float(v) for v in r["quat_xyzw"])
        self._translate.Set(Gf.Vec3d(*map(float, r["position"])))
        self._orient.Set(Gf.Quatd(w, x, y, z))
        st = self._state
        st.position = np.asarray(r["position"], float)
        st.attitude = np.asarray(r["quat_xyzw"], float)
        st.linear_velocity = np.asarray(r["velocity"], float)
        st.linear_body_velocity = Rotation.from_quat(st.attitude).inv().apply(st.linear_velocity)
        st.angular_velocity = np.asarray(r["body_rates"], float)
        st.linear_acceleration = np.asarray(r["acceleration"], float)

    def update(self, dt: float):
        self.sim_time += dt
        self._step += 1
        for backend in self._backends:
            backend.update(dt)
