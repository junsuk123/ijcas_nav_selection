"""PX4 SITL as an *estimator* inside the IJCAS rig.

Stage A (open loop) flies a forced trajectory: the rigid body follows the reference and PX4's
actuator outputs are not applied.  PX4 still has to believe it is flying, otherwise its land
detector reports ``landed``/``at_rest`` and EKF2 switches to on-ground behaviour (fake position,
zero-velocity fusion) that a real flight would not see.  ``Px4Keeper`` therefore arms PX4 and holds
it in OFFBOARD attitude mode at hover thrust over MAVLink.  Attitude offboard needs no position
estimate, so GNSS faults do not push PX4 into a position failsafe.

``Px4Params`` writes the parameters into a copy of the v1.14 posix ``rcS`` right where the stock
script sources ``px4-rc.params`` (before any module starts), so they hold from boot.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Parameters that keep PX4 an honest, independent GNSS/INS candidate in this rig.
#   * no RC / no GCS / no battery failsafes (SITL artefacts, not navigation behaviour)
#   * external vision, optical flow and range aiding OFF: the PX4 candidate uses only its own
#     IMU + GNSS + baro + mag (Notion 02: "외부 VIO/LIO 입력은 끄고 GNSS/INS 후보의 독립성을 유지")
#   * no auto-disarm: the forced trajectory must not end because a heuristic disarmed PX4
DEFAULT_PARAMS: dict[str, float | int] = {
    "COM_RC_IN_MODE": 4,        # stick input disabled
    "NAV_RCL_ACT": 0,
    "NAV_DLL_ACT": 0,
    "COM_DL_LOSS_T": 60,
    "COM_LOW_BAT_ACT": 0,
    "SIM_BAT_MIN_PCT": 80,
    "COM_DISARM_PRFLT": -1,
    "COM_DISARM_LAND": -1,
    "COM_OF_LOSS_T": 5,
    "COM_OBL_RC_ACT": 5,        # offboard loss -> hold (we never stop streaming)
    "EKF2_EV_CTRL": 0,
    "EKF2_OF_CTRL": 0,
    "EKF2_RNG_CTRL": 0,
    "SDLOG_MODE": -1,           # no ulog by default (disk); runs that need replay enable it
}
# Note: PX4 v1.14's uXRCE-DDS client always adds its timesync offset (re-estimated every second
# against the agent's wall clock) to /fmu/out timestamps, and v1.14 has no UXRCE_DDS_SYNCT to turn
# it off.  In lockstep at RTF != 1 those timestamps jump; Stage A therefore records PX4's estimate
# over MAVLink (time_boot_ms = lockstep = simulation time), see Px4Keeper.records.


@dataclass
class Px4Params:
    values: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))

    def rc_lines(self) -> str:
        out = []
        for name, value in sorted(self.values.items()):
            if not re.fullmatch(r"[A-Z0-9_]+", name):
                raise ValueError(f"bad PX4 parameter name {name!r}")
            out.append(f"param set {name} {value}")
        return "\n".join(out) + "\n"

    def write_rc(self, px4_dir: str, dest_dir: str) -> str:
        """A copy of the stock posix rcS with our ``param set`` lines spliced in."""
        stock = Path(px4_dir, "ROMFS/px4fmu_common/init.d-posix/rcS").read_text()
        anchor = ". px4-rc.params"
        if stock.count(anchor) != 1:
            raise RuntimeError("PX4 rcS layout changed: px4-rc.params anchor not found once")
        patched = stock.replace(anchor, anchor + "\n# --- IJCAS parameters ---\n"
                                + self.rc_lines() + "# --- end IJCAS parameters ---")
        path = Path(dest_dir, "rcS_ijcas")
        path.write_text(patched)
        return str(path)


def make_launch_tool(px4_dir: str, params: Px4Params, vehicle_id: int = 0,
                     model: str = "gazebo-classic_iris"):
    """Pegasus PX4LaunchTool that boots our rcS (imported lazily: Isaac-only module)."""
    from pegasus.simulator.logic.backends.tools.px4_launch_tool import PX4LaunchTool

    class IjcasPX4LaunchTool(PX4LaunchTool):
        def __init__(self):
            super().__init__(px4_dir, vehicle_id, model)
            self.rc_script = params.write_rc(px4_dir, self.root_fs.name)

    return IjcasPX4LaunchTool()


class Px4Keeper:
    """Arms PX4 and keeps it in OFFBOARD attitude mode at hover thrust (outputs not applied).

    Talks MAVLink on PX4's onboard port (UDP 14540 + instance).  ``tick(sim_time)`` is called
    from the simulation loop; it streams SET_ATTITUDE_TARGET at ``rate_hz`` and retries
    arming/mode switching until both are confirmed by HEARTBEAT.
    """

    PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

    def __init__(self, instance: int = 0, rate_hz: float = 20.0, hover_thrust: float = 0.6,
                 start_after_s: float = 2.0):
        from pymavlink import mavutil
        self._mavutil = mavutil
        self.conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{14540 + instance}",
                                               source_system=245, source_component=191)
        self.rate_hz = rate_hz
        self.hover_thrust = hover_thrust
        self.start_after_s = start_after_s
        self._last_send = -1e9
        self._last_cmd = -1e9
        self.armed = False
        self.offboard = False
        self.target_system = 1
        self.events: list[tuple[float, str]] = []
        self.statustexts: list[tuple[float, str]] = []
        # EKF2 output in PX4 lockstep time (time_boot_ms / time_usec), see record_streams()
        self.records: dict[str, list] = {"lpos": [], "att": [], "est": [], "ext": [], "gps": [],
                                         "himu": [], "origin": [], "odo": []}
        self._streams_requested = False

    def _poll(self, now: float) -> None:
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT" and msg.get_srcComponent() == 1:
                self.target_system = msg.get_srcSystem()
                armed = bool(msg.base_mode & self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                main_mode = (msg.custom_mode >> 16) & 0xFF
                offboard = main_mode == self.PX4_CUSTOM_MAIN_MODE_OFFBOARD
                if armed != self.armed or offboard != self.offboard:
                    self.events.append((now, f"armed={armed} offboard={offboard}"))
                self.armed, self.offboard = armed, offboard
            elif kind == "STATUSTEXT":
                self.statustexts.append((now, str(msg.text)))
            elif kind == "LOCAL_POSITION_NED":
                self.records["lpos"].append([msg.time_boot_ms * 1e-3, msg.x, msg.y, msg.z,
                                             msg.vx, msg.vy, msg.vz, now])
            elif kind == "ATTITUDE_QUATERNION":
                self.records["att"].append([msg.time_boot_ms * 1e-3, msg.q1, msg.q2, msg.q3,
                                            msg.q4, msg.rollspeed, msg.pitchspeed, msg.yawspeed,
                                            now])
            elif kind == "ESTIMATOR_STATUS":
                self.records["est"].append([msg.time_usec * 1e-6, int(msg.flags), msg.vel_ratio,
                                            msg.pos_horiz_ratio, msg.pos_vert_ratio, msg.mag_ratio,
                                            msg.hagl_ratio, msg.tas_ratio, msg.pos_horiz_accuracy,
                                            msg.pos_vert_accuracy, now])
            elif kind == "EXTENDED_SYS_STATE":
                self.records["ext"].append([now, int(msg.landed_state)])
            elif kind == "GPS_GLOBAL_ORIGIN":
                self.records["origin"].append([msg.latitude * 1e-7, msg.longitude * 1e-7,
                                               msg.altitude * 1e-3, now])
            elif kind == "HIGHRES_IMU":
                self.records["himu"].append([msg.time_usec * 1e-6, msg.xacc, msg.yacc, msg.zacc,
                                             msg.xgyro, msg.ygyro, msg.zgyro, now])
            elif kind == "ODOMETRY":
                # EKF2's own variances (vehicle_odometry -> MAVLink ODOMETRY, PX4 v1.14 streams/ODOMETRY.hpp):
                # pose_covariance / velocity_covariance are 6x6 upper-right triangles; PX4 fills the diagonal
                # (position x,y,z; orientation roll,pitch,yaw; velocity x,y,z) in frame_id / child_frame_id.
                pc, vc = msg.pose_covariance, msg.velocity_covariance
                self.records["odo"].append([msg.time_usec * 1e-6, int(msg.frame_id), int(msg.child_frame_id),
                                            pc[0], pc[6], pc[11], pc[15], pc[18], pc[20],
                                            vc[0], vc[6], vc[11], int(msg.reset_counter), int(msg.quality),
                                            now])
            elif kind == "GPS_RAW_INT":
                self.records["gps"].append([msg.time_usec * 1e-6, int(msg.fix_type), msg.lat,
                                            msg.lon, msg.alt, msg.eph, msg.satellites_visible, now])

    def request_streams(self, rates_hz: dict | None = None) -> None:
        """Ask PX4 for its estimator outputs on this link (MAV_CMD_SET_MESSAGE_INTERVAL)."""
        mav = self._mavutil.mavlink
        rates_hz = rates_hz or {mav.MAVLINK_MSG_ID_LOCAL_POSITION_NED: 50,
                                mav.MAVLINK_MSG_ID_ATTITUDE_QUATERNION: 50,
                                mav.MAVLINK_MSG_ID_ESTIMATOR_STATUS: 10,
                                mav.MAVLINK_MSG_ID_EXTENDED_SYS_STATE: 5,
                                mav.MAVLINK_MSG_ID_GPS_RAW_INT: 10,
                                mav.MAVLINK_MSG_ID_HIGHRES_IMU: 50,
                                mav.MAVLINK_MSG_ID_ODOMETRY: 50}
        for msg_id, hz in rates_hz.items():
            self.conn.mav.command_long_send(self.target_system, 1,
                                            mav.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id,
                                            int(1e6 / hz), 0, 0, 0, 0, 0)
        # the local-frame origin (for exact global evaluation of PX4-frame outputs)
        self.conn.mav.command_long_send(self.target_system, 1, mav.MAV_CMD_SET_MESSAGE_INTERVAL,
                                        0, mav.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, int(1e6), 0, 0,
                                        0, 0, 0)
        self._streams_requested = True

    def tick(self, now: float) -> None:
        self._poll(now)
        if not self._streams_requested and now >= 1.0:
            self.request_streams()
        if now < self.start_after_s:
            return
        if now - self._last_send >= 1.0 / self.rate_hz:
            self._last_send = now
            # level attitude, body rates ignored, hover thrust (type_mask: ignore rates)
            self.conn.mav.set_attitude_target_send(
                int(now * 1000) & 0xFFFFFFFF, self.target_system, 1, 0b00000111,
                [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, self.hover_thrust)
        if (not self.offboard or not self.armed) and now - self._last_cmd >= 1.0:
            self._last_cmd = now
            mav = self._mavutil.mavlink
            if not self.offboard:
                self.conn.mav.command_long_send(
                    self.target_system, 1, mav.MAV_CMD_DO_SET_MODE, 0,
                    mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, self.PX4_CUSTOM_MAIN_MODE_OFFBOARD,
                    0, 0, 0, 0, 0)
            if not self.armed:
                self.conn.mav.command_long_send(
                    self.target_system, 1, mav.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                    1, 0, 0, 0, 0, 0, 0)

    def read_param(self, name: str, timeout_s: float = 3.0):
        """Parameter value; INT32 parameters arrive bit-cast into the float field."""
        import struct
        self.conn.mav.param_request_read_send(self.target_system, 1, name.encode(), -1)
        end = time.time() + timeout_s
        while time.time() < end:
            msg = self.conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
            if msg is not None and msg.param_id.strip("\x00") == name:
                if msg.param_type == self._mavutil.mavlink.MAV_PARAM_TYPE_INT32:
                    return struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
                return msg.param_value
        return None

    def summary(self) -> dict:
        return {"armed": self.armed, "offboard": self.offboard,
                "events": self.events[-20:], "statustexts": self.statustexts[-30:]}


def exact_time_backend_class():
    """Pegasus PX4MavlinkBackend whose lockstep clock is the IMU sample time (Isaac-only).

    Stock Pegasus advances PX4 time by ``int(dt * 1e6)`` per call (4999 us for a float32 5 ms step)
    and only when a new IMU sample exists, and labels sample k with time k+1.  Here HIL_SENSOR /
    HIL_GPS carry the simulation time at which the sample was generated, so PX4 time == sim time.
    """
    import time as _time

    from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend

    class ExactTimePX4MavlinkBackend(PX4MavlinkBackend):
        def __init__(self, config, imu):
            super().__init__(config)
            self._ijcas_imu = imu

        def update(self, dt):
            if not self._received_first_hearbeat:
                self.wait_for_first_hearbeat()
                return
            if self._sensor_data.received_first_imu and not self._sensor_data.new_imu_data:
                return
            self.poll_mavlink_messages()
            if (_time.time() - self._last_heartbeat_sent_time) > 1.0:
                self.send_heartbeat()
                self._last_heartbeat_sent_time = _time.time()
            last = self._ijcas_imu.last
            if last is not None:
                self._current_utime = max(self._current_utime + 1,
                                          int(round(last["t"] * 1e6)) + 1)
            else:
                self._current_utime += int(round(dt * 1e6))
            self.send_sensor_msgs(self._current_utime)
            self.send_gps_msgs(self._current_utime)

    return ExactTimePX4MavlinkBackend
