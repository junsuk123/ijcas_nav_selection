#!/usr/bin/env bash
# G0: OpenVINS official example without external data — the bundled simulator
# (ov_msckf run_simulation, config/rpng_sim, trajectory ov_data/sim/tum_corridor1_512_16_okvis.txt)
# followed by the official evaluator (ov_eval error_simulation).  EuRoC could not be fetched
# from this network (ETH Research Collection 429, OpenVINS Drive folder 404); see docs/G0_KO.md.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +u; source "$here/../env/ijcas_env.sh"; set -u
out=$IJCAS_ROOT/results/g0/openvins_sim
rm -rf "$out"; mkdir -p "$out"
cp -r "$IJCAS_WS/src/open_vins/config/rpng_sim" "$out/config"
cfg=$out/config/estimator_config.yaml
sed -i \
  -e "s|^save_total_state:.*|save_total_state: true|" \
  -e "s|^filepath_est:.*|filepath_est: \"$out/ov_estimate.txt\"|" \
  -e "s|^filepath_std:.*|filepath_std: \"$out/ov_estimate_std.txt\"|" \
  -e "s|^filepath_gt:.*|filepath_gt: \"$out/ov_groundtruth.txt\"|" \
  -e "s|^sim_traj_path:.*|sim_traj_path: \"$IJCAS_WS/src/open_vins/ov_data/sim/tum_corridor1_512_16_okvis.txt\"|" \
  -e "s|^verbosity:.*|verbosity: \"INFO\"|" \
  "$cfg"
start=$(date +%s)
# ROS 2: the visualizer reads save_total_state / filepath_* as node parameters, not from the YAML.
# The binary segfaults while tearing down its ROS 2 publishers after the run (exit 139); the
# estimate files are complete by then, so that exit code is tolerated and recorded.
set +e
ros2 run ov_msckf run_simulation "$cfg" --ros-args -p save_total_state:=true \
  -p filepath_est:="$out/ov_estimate.txt" -p filepath_std:="$out/ov_estimate_std.txt" \
  -p filepath_gt:="$out/ov_groundtruth.txt" > "$out/run_simulation.log" 2>&1
code=$?
set -e
echo "wall_s $(( $(date +%s) - start )) exit $code" > "$out/wall.txt"
grep -v "rcl publisher handle\|Failed to delete datawriter" "$out/run_simulation.log" \
  | grep -E "error to gt|nees|dist =" | tail -3 > "$out/final_metrics.txt" || true
MPLBACKEND=Agg ros2 run ov_eval error_simulation "$out/ov_estimate.txt" "$out/ov_estimate_std.txt" \
  "$out/ov_groundtruth.txt" > "$out/error_simulation.log" 2>&1 || true
tail -40 "$out/error_simulation.log"
