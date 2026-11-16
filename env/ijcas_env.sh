# Source me:  source env/ijcas_env.sh
# One environment for every IJCAS process (ROS 2 nodes, engines, tools).  Isaac Sim runs with its
# own Python (isaac/run_isaac.sh) but uses the same ROS domain and RMW.
IJCAS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export IJCAS_ROOT
export IJCAS_WS=${IJCAS_WS:-$HOME/ijcas_ws}
export IJCAS_DATA=${IJCAS_DATA:-$HOME/ijcas_data}
export ISAACSIM_PATH=${ISAACSIM_PATH:-$HOME/isaacsim/_build/linux-x86_64/release}
export PX4_DIR=${PX4_DIR:-$IJCAS_WS/third_party/PX4-Autopilot}
export XRCE_AGENT=${XRCE_AGENT:-$IJCAS_WS/third_party/xrce_install/bin/MicroXRCEAgent}
GLIM_ROOT=$IJCAS_WS/third_party/glim_root
export GLIM_ROOT

# ROS 2 Humble, Fast DDS (the PX4 XRCE agent and Isaac's bridge both use it), private domain
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
[[ -f $IJCAS_WS/install/setup.bash ]] && source "$IJCAS_WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
unset CYCLONEDDS_URI

# GLIM 1.2.2 (official PPA debs extracted without root, see docs/G0_KO.md)
export AMENT_PREFIX_PATH=$GLIM_ROOT/opt/ros/humble${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}
export LD_LIBRARY_PATH=$GLIM_ROOT/opt/ros/humble/lib:$GLIM_ROOT/usr/local/lib:$GLIM_ROOT/usr/lib/x86_64-linux-gnu:$IJCAS_WS/third_party/xrce_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$IJCAS_ROOT${PYTHONPATH:+:$PYTHONPATH}

# A terminal inside the VS Code snap exports snap GTK paths that crash rviz2; restore host GUI env.
if [[ -n "${SNAP:-}" ]]; then
  unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR \
        GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LOCPATH
  [[ -n "${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-}" ]] && export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
  [[ -n "${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-}" ]] && export XDG_CONFIG_DIRS="$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG"
fi
true  # always succeed when sourced under set -e
