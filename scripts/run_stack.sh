#!/usr/bin/env bash
# One IJCAS run: XRCE agent + OpenVINS + FAST-LIO2 + context monitor + recorder + Isaac rig (Stage A).
#   scripts/run_stack.sh <scenario.yaml> <scene> <seed> <out_dir> [duration_s]
# Every process runs in its own process group (setsid) and the whole group is stopped on exit:
# `ros2 run` wrappers left their engine children alive between runs (duplicate publishers), so
# the engines are executed directly.  The engines' actual subscriptions are written to
# <out>/subscriptions.txt (ground-truth leak check: nothing under /ijcas/gt/).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +eu; source "$here/../env/ijcas_env.sh"; set -eu
scenario=$1; scene=$2; seed=$3; out=$4; dur=${5:-}
OV_BIN=$IJCAS_WS/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf
LIO_BIN=$IJCAS_WS/install/fast_lio/lib/fast_lio/fastlio_mapping
# patterns anchored at the start of a process command line (a shell whose command text merely
# mentions these paths must not match)
for pat in "^$OV_BIN" "^$LIO_BIN" "^$ISAACSIM_PATH/kit/python/bin/python3 $IJCAS_ROOT/isaac/ijcas_world.py" \
           "^python3 $IJCAS_ROOT/ros2/"; do
  if pgrep -f "$pat" >/dev/null; then echo "refusing: '$pat' is still running" >&2; exit 1; fi
done
if pgrep -x px4 >/dev/null || pgrep -x MicroXRCEAgent >/dev/null; then
  echo "refusing: PX4 SITL or an XRCE agent is already running" >&2; exit 1
fi
mkdir -p "$out"
groups=()
start() { local log=$1; shift; setsid "$@" > "$log" 2>&1 < /dev/null & groups+=($!); }
stop_all() {
  for g in "${groups[@]:-}"; do kill -INT -- "-$g" 2>/dev/null || true; done
  sleep 3
  for g in "${groups[@]:-}"; do kill -KILL -- "-$g" 2>/dev/null || true; done
}
trap stop_all EXIT
start "$out/agent.log" "$XRCE_AGENT" udp4 -p 8888
start "$out/openvins.log" "$OV_BIN" --ros-args -r __ns:=/ov_msckf \
  -p config_path:="$IJCAS_ROOT/config/engines/openvins/estimator_config.yaml" -p use_sim_time:=true
start "$out/fastlio.log" "$LIO_BIN" --ros-args --params-file "$IJCAS_ROOT/config/engines/fast_lio/ijcas.yaml" \
  -r /Odometry:=/fast_lio/odom -p use_sim_time:=true
if [[ -f "$IJCAS_ROOT/ros2/context_monitor.py" ]]; then
  start "$out/context_monitor.log" python3 "$IJCAS_ROOT/ros2/context_monitor.py"
fi
setsid python3 "$IJCAS_ROOT/ros2/recorder.py" --out "$out/record.npz" --seconds 3600 \
  > "$out/recorder.log" 2>&1 < /dev/null & rec=$!
sleep 3
args=(--scenario "$scenario" --scene "$scene" --seed "$seed" --out "$out")
[[ -n "$dur" ]] && args+=(--duration "$dur")
( sleep 90; { ros2 node info /ov_msckf/run_subscribe_msckf 2>/dev/null; ros2 node info /laser_mapping 2>/dev/null
              ros2 node info /ijcas_context_monitor 2>/dev/null; ros2 node list 2>/dev/null; } \
              > "$out/subscriptions.txt" ) &
"$IJCAS_ROOT/isaac/run_isaac.sh" "$IJCAS_ROOT/isaac/ijcas_world.py" "${args[@]}" \
  > "$out/isaac.log" 2>&1 || echo "isaac exit $?" >> "$out/isaac.log"
sleep 3   # let the engines drain their queues
kill -INT -- "-$rec" 2>/dev/null || true
for _ in $(seq 1 60); do kill -0 "$rec" 2>/dev/null || break; sleep 1; done
grep -E "IJCAS_RUN" "$out/isaac.log" | cut -c1-200 || true
tail -1 "$out/recorder.log"
