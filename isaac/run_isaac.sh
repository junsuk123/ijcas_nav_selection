#!/usr/bin/env bash
# Run an IJCAS Isaac Sim script with Isaac's own Python and its bundled ROS 2 (Humble) bridge.
#   isaac/run_isaac.sh isaac/probe_px4.py --seconds 60
# System ROS (Python 3.10) must not leak into Isaac's Python 3.11.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/.." && pwd)
isaacsim_path=${ISAACSIM_PATH:-$HOME/isaacsim/_build/linux-x86_64/release}
unset ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH \
      PYTHONPATH CYCLONEDDS_URI
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$isaacsim_path/exts/isaacsim.ros2.bridge/humble/lib"
export PYTHONPATH="$root/isaac:$root"
export PX4_DIR=${PX4_DIR:-$HOME/ijcas_ws/third_party/PX4-Autopilot}
script=$1; shift
exec "$isaacsim_path/python.sh" "$script" "$@"
