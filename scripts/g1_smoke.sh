#!/usr/bin/env bash
# G1: one nominal run of the rig in a scene with the topic capture beside it.
#   scripts/g1_smoke.sh warehouse 1
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +eu; source "$here/../env/ijcas_env.sh"; set -eu
scene=${1:-warehouse}; seed=${2:-1}; dur=${3:-45}
out=$IJCAS_DATA/g1/${scene}_${seed}
mkdir -p "$out"
if pgrep -x px4 >/dev/null; then echo "a PX4 SITL is already running" >&2; exit 1; fi
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
"$XRCE_AGENT" udp4 -p 8888 > "$out/agent.log" 2>&1 & pids+=($!)
python3 "$IJCAS_ROOT/tools/g1_capture.py" --seconds $((dur * 3 + 120)) --out "$out/capture.npz" \
  > "$out/capture.log" 2>&1 & cap=$!
"$IJCAS_ROOT/isaac/run_isaac.sh" "$IJCAS_ROOT/isaac/ijcas_world.py" \
  --scenario "$IJCAS_ROOT/config/scenarios/smoke.yaml" --scene "$scene" --seed "$seed" \
  --duration "$dur" --out "$out" > "$out/isaac.log" 2>&1 || echo "isaac exit $?"
kill -INT "$cap" 2>/dev/null || true; wait "$cap" 2>/dev/null || true
grep -E "IJCAS_RUN|Traceback|Error:" "$out/isaac.log" | head -5 || true
cat "$out/capture.log"
