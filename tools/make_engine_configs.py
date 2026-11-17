"""Generate OpenVINS and FAST-LIO2 configurations from a run's *measured* calibration (G1/G2).

    python3 tools/make_engine_configs.py $IJCAS_DATA/g1/warehouse_1/manifest.json

Writes config/engines/openvins/{estimator_config,kalibr_imu_chain,kalibr_imucam_chain}.yaml and
config/engines/fast_lio/ijcas.yaml.  The estimators themselves are untouched; only their inputs
(topics, calibration, noise, frames) are set.  Nothing from EuRoC/Ouster examples is reused as
calibration.  FAST-LIO2 (hku-mars/FAST_LIO, ROS2 branch, commit cc3c9e8) is built with the IJCAS
adapter patch config/engines/fast_lio/fast_lio_ijcas.patch: input parsing of the rig's cloud
(lidar_type 5: x y z intensity t[s]), reliable cloud QoS, and the filter velocity in the published
twist (world frame).  GLIM (config/engines/glim, function ``glim``) was the LIO engine until
2026-11-16 and is kept only to reproduce the archived runs.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OV_SRC = Path.home() / "ijcas_ws/src/open_vins/config/euroc_mav"
GLIM_SRC = Path.home() / "ijcas_ws/third_party/glim_root/opt/ros/humble/share/glim/config"

# IMU noise of ijcas_sim.sensors.ImuNoise (continuous-time densities), inflated x2 for the filters
IMU = {"gyro_noise_density": 1.8665e-4, "gyro_random_walk": 3.8785e-5,
       "accel_noise_density": 1.86e-3, "accel_random_walk": 6.0e-3}


def _mat(rows):
    return "[" + ", ".join("[" + ", ".join(f"{v:.9f}" for v in r) + "]" for r in rows) + "]"


def openvins(cal: dict, out: Path, rate_hz: float = 200.0):
    out.mkdir(parents=True, exist_ok=True)
    imu = (f"%YAML:1.0\n\nimu0:\n  T_i_b:\n    - [1.0, 0.0, 0.0, 0.0]\n    - [0.0, 1.0, 0.0, 0.0]\n"
           f"    - [0.0, 0.0, 1.0, 0.0]\n    - [0.0, 0.0, 0.0, 1.0]\n"
           f"  accelerometer_noise_density: {2 * IMU['accel_noise_density']:.6e}\n"
           f"  accelerometer_random_walk: {2 * IMU['accel_random_walk']:.6e}\n"
           f"  gyroscope_noise_density: {2 * IMU['gyro_noise_density']:.6e}\n"
           f"  gyroscope_random_walk: {2 * IMU['gyro_random_walk']:.6e}\n"
           f"  rostopic: /ijcas/imu\n  time_offset: 0.0\n"
           f"  update_rate: {rate_hz:.1f}\n"
           # IMU intrinsics block required by OpenVINS v2.7 (ideal IMU: identity, no g-sensitivity)
           '  model: "kalibr"\n'
           "  Tw:\n    - [ 1.0, 0.0, 0.0 ]\n    - [ 0.0, 1.0, 0.0 ]\n    - [ 0.0, 0.0, 1.0 ]\n"
           "  R_IMUtoGYRO:\n    - [ 1.0, 0.0, 0.0 ]\n    - [ 0.0, 1.0, 0.0 ]\n    - [ 0.0, 0.0, 1.0 ]\n"
           "  Ta:\n    - [ 1.0, 0.0, 0.0 ]\n    - [ 0.0, 1.0, 0.0 ]\n    - [ 0.0, 0.0, 1.0 ]\n"
           "  R_IMUtoACC:\n    - [ 1.0, 0.0, 0.0 ]\n    - [ 0.0, 1.0, 0.0 ]\n    - [ 0.0, 0.0, 1.0 ]\n"
           "  Tg:\n    - [ 0.0, 0.0, 0.0 ]\n    - [ 0.0, 0.0, 0.0 ]\n    - [ 0.0, 0.0, 0.0 ]\n")
    (out / "kalibr_imu_chain.yaml").write_text(imu)
    lines = ["%YAML:1.0", ""]
    for name in ("cam0", "cam1"):
        c = cal["cameras"][name]
        R = np.asarray(c["R_body_optical"])
        p = np.asarray(c["p_body_optical"])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p           # camera -> IMU (p_C in I, R_CtoI): OpenVINS T_imu_cam
        K = np.asarray(c["K"])
        lines += [f"{name}:", "  T_imu_cam: #rotation from camera to IMU, position of camera in IMU"]
        lines += [f"    - [{', '.join(f'{v:.9f}' for v in row)}]" for row in T]
        lines += ["  cam_overlaps: [" + ("1" if name == "cam0" else "0") + "]",
                  "  camera_model: pinhole",
                  "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]",
                  "  distortion_model: radtan",
                  f"  intrinsics: [{K[0, 0]:.6f}, {K[1, 1]:.6f}, {K[0, 2]:.6f}, {K[1, 2]:.6f}]",
                  f"  resolution: [{c['width']}, {c['height']}]",
                  f"  rostopic: /{name}/image_raw", ""]
    (out / "kalibr_imucam_chain.yaml").write_text("\n".join(lines))
    est = (OV_SRC / "estimator_config.yaml").read_text()
    repl = {"use_stereo:": "use_stereo: true", "max_cameras:": "max_cameras: 2",
            "calib_cam_extrinsics:": "calib_cam_extrinsics: false",
            "calib_cam_intrinsics:": "calib_cam_intrinsics: false",
            "calib_cam_timeoffset:": "calib_cam_timeoffset: false",
            "gravity_mag:": "gravity_mag: 9.80665",
            "verbosity:": 'verbosity: "INFO"',
            "record_timing_information:": "record_timing_information: false",
            "save_total_state:": "save_total_state: false",
            "use_mask:": "use_mask: false",
            # G2 initialisation choice (fixed before any test run): the rig takes off from hover
            # with a smooth ramp, so the static initializer never sees its jerk; use the official
            # dynamic initializer, and a lower jerk threshold for the static one.
            "init_dyn_use:": "init_dyn_use: true",
            "init_imu_thresh:": "init_imu_thresh: 0.5"}
    out_lines = []
    for line in est.splitlines():
        key = line.split("#")[0].strip().split(" ")[0] if line.strip() else ""
        out_lines.append(repl.get(key, line))
    (out / "estimator_config.yaml").write_text("\n".join(out_lines) + "\n")


def glim(cal: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in GLIM_SRC.glob("*.json"):
        shutil.copy(f, out / f.name)

    def edit(name, fn):
        path = out / name
        txt = path.read_text()
        path.write_text(fn(txt))

    edit("config.json", lambda t: t.replace('"config_odometry_gpu.json"', '"config_odometry_cpu.json"')
         .replace('"config_sub_mapping_gpu.json"', '"config_sub_mapping_passthrough.json"')
         .replace('"config_global_mapping_gpu.json"', '"config_global_mapping_pose_graph.json"'))
    rbl = np.asarray(cal["lidar"]["R_body_lidar"])
    pbl = np.asarray(cal["lidar"]["p_body_lidar"])
    # T_lidar_imu maps IMU-frame points into the LiDAR frame: p_l = R_bl^T (p_i - p_bl)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(rbl.T)
    t = -(rbl.T @ pbl)
    tli = [*map(float, t), *map(float, r.as_quat())]

    def sensors(txt):
        start = txt.index('"T_lidar_imu": [')
        end = txt.index("]", start) + 1
        return txt[:start] + '"T_lidar_imu": ' + json.dumps([round(v, 6) for v in tli]) + txt[end:]
    edit("config_sensors.json", sensors)

    def ros(txt):
        txt = txt.replace('"enable_local_mapping": true', '"enable_local_mapping": false')
        txt = txt.replace('"enable_global_mapping": true', '"enable_global_mapping": false')
        txt = txt.replace('"acc_scale": 0.0', '"acc_scale": 1.0')
        txt = txt.replace('"imu_frame_id": ""', '"imu_frame_id": "imu_link"')
        txt = txt.replace('"lidar_frame_id": ""', '"lidar_frame_id": "lidar"')
        txt = txt.replace('"odom_frame_id": "odom"', '"odom_frame_id": "glim_odom"')
        txt = txt.replace('"map_frame_id": "map"', '"map_frame_id": "glim_map"')
        txt = txt.replace('"publish_imu2lidar": true', '"publish_imu2lidar": false')
        txt = txt.replace('"libmemory_monitor.so",\n', '').replace('"libstandard_viewer.so",\n', '')
        txt = txt.replace('"/os_cloud_node/imu"', '"/ijcas/imu"')
        txt = txt.replace('"/os_cloud_node/points"', '"/ijcas/lidar/points"')
        # reliable QoS: 640 kB clouds over best-effort UDP lose fragments -> whole scans dropped
        txt = txt.replace('"imu_qos": {\n      "profile": "sensor_data",',
                          '"imu_qos": {\n      "profile": "default",')
        txt = txt.replace('"points_qos": {\n      "profile": "sensor_data"',
                          '"points_qos": {\n      "profile": "default"')
        return txt
    edit("config_ros.json", ros)
    edit("config_odometry_cpu.json", lambda t: t.replace('"num_threads": 2', '"num_threads": 4'))
    edit("config_preprocess.json", lambda t: t.replace('"num_threads": 2', '"num_threads": 4'))


def fast_lio(cal: dict, out: Path):
    """FAST-LIO2 parameters.  extrinsic_R/T = LiDAR pose in the IMU (= body FLU) frame:
    p_imu = R p_lidar + T, i.e. exactly the rig's R_body_lidar / p_body_lidar.  Process noise and
    map resolution stay at the upstream defaults (velodyne.yaml); outputs other than odometry off."""
    out.mkdir(parents=True, exist_ok=True)
    rbl = np.asarray(cal["lidar"]["R_body_lidar"], float)
    pbl = np.asarray(cal["lidar"]["p_body_lidar"], float)
    fmt = lambda v: "[" + ", ".join(f"{x:.9f}" for x in v) + "]"  # noqa: E731
    txt = f"""# Generated by tools/make_engine_configs.py from {cal.get('lidar', {}).get('config', '?')}
/**:
  ros__parameters:
    feature_extract_enable: false
    point_filter_num: 3
    max_iteration: 3
    filter_size_surf: 0.5
    filter_size_map: 0.5
    cube_side_length: 1000.0
    runtime_pos_log_enable: false
    map_file_path: ""
    common:
      lid_topic: "/ijcas/lidar/points"
      imu_topic: "/ijcas/imu"
      time_sync_en: false
      time_offset_lidar_to_imu: 0.0
    preprocess:
      lidar_type: 5            # IJCAS adapter: x y z intensity t (s from scan start)
      scan_line: 32
      scan_rate: 10
      timestamp_unit: 0        # seconds
      blind: 0.5
    mapping:
      acc_cov: 0.1
      gyr_cov: 0.1
      b_acc_cov: 0.0001
      b_gyr_cov: 0.0001
      fov_degree: 360.0
      det_range: 100.0
      extrinsic_est_en: false
      extrinsic_T: {fmt(pbl)}
      extrinsic_R: {fmt(rbl.reshape(-1))}
    publish:
      path_en: false
      effect_map_en: false
      map_en: false
      scan_publish_en: false
      dense_publish_en: false
      scan_bodyframe_pub_en: false
    pcd_save:
      pcd_save_en: false
      interval: -1
"""
    (out / "ijcas.yaml").write_text(txt)


def main(manifest: str):
    cal = json.loads(Path(manifest).read_text())["calibration"]
    openvins(cal, ROOT / "config/engines/openvins")
    fast_lio(cal, ROOT / "config/engines/fast_lio")
    (ROOT / "config/engines/calibration_source.json").write_text(json.dumps(
        {"manifest": manifest, "calibration": cal}, indent=2))
    print("wrote config/engines/openvins and config/engines/fast_lio")


if __name__ == "__main__":
    main(sys.argv[1])
