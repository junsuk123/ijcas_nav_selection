#!/usr/bin/env bash
# G0/G1: PX4 SITL v1.14.3 + uXRCE-DDS agent v2.4.2 + px4_msgs(release/1.14) on ROS domain 42,
# with the forced-trajectory Isaac probe.  Writes results/g0/px4_probe/{isaac_summary,dds}.json.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +eu; source "$here/../env/ijcas_env.sh"; set -eu
seconds=${1:-90}
out=$IJCAS_ROOT/results/g0/px4_probe
mkdir -p "$out"
if pgrep -x px4 >/dev/null; then echo "a PX4 SITL is already running; stop it first" >&2; exit 1; fi
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
"$XRCE_AGENT" udp4 -p 8888 > "$out/agent.log" 2>&1 & pids+=($!)
python3 "$IJCAS_ROOT/tools/px4_dds_listen.py" --seconds $((seconds + 150)) \
  --out "$out/dds.json" > "$out/listen.log" 2>&1 & listener=$!
"$IJCAS_ROOT/isaac/run_isaac.sh" "$IJCAS_ROOT/isaac/probe_px4.py" --seconds "$seconds" \
  --out "$out" > "$out/isaac.log" 2>&1 || echo "isaac exit $?"
kill -INT "$listener" 2>/dev/null || true
wait "$listener" 2>/dev/null || true
grep -E "PROBE|Traceback|Error" "$out/isaac.log" | head -20 || true
