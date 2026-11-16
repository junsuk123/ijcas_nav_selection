"""G0: record the exact software/hardware combination -> results/g0/env_manifest.json"""
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("IJCAS_WS", Path.home() / "ijcas_ws"))
ISAAC = Path(os.environ.get("ISAACSIM_PATH", Path.home() / "isaacsim/_build/linux-x86_64/release"))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def git(path):
    p = Path(path).resolve()
    return {"path": str(p), "commit": sh(f"git -C '{p}' rev-parse HEAD"),
            "describe": sh(f"git -C '{p}' describe --tags --always"),
            "remote": sh(f"git -C '{p}' remote get-url origin")}


m = {
    "os": sh("lsb_release -ds"), "kernel": platform.release(), "cpu": sh("lscpu | grep 'Model name'"),
    "cores": os.cpu_count(), "ram_gb": sh("free -g | awk 'NR==2{print $2}'"),
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "cuda": sh("nvcc --version | tail -1"),
    "ros": {"distro": os.environ.get("ROS_DISTRO"), "rmw": os.environ.get("RMW_IMPLEMENTATION"),
            "domain": os.environ.get("ROS_DOMAIN_ID")},
    "isaac_sim": {"version_file": (ISAAC / "VERSION").read_text().strip() if (ISAAC / "VERSION").exists() else None,
                  **git(Path.home() / "isaacsim")},
    "pegasus": git(WS / "third_party/PegasusSimulator"),
    "px4": git(WS / "third_party/PX4-Autopilot"),
    "openvins": git(WS / "src/open_vins"),
    "px4_msgs": git(WS / "src/px4_msgs"),
    "xrce_agent": git((WS / "third_party/xrce_install").resolve().parent / "Micro-XRCE-DDS-Agent"),
    "glim_debs_sha256": (WS / "third_party/glim_debs/SHA256SUMS").read_text().splitlines()
    if (WS / "third_party/glim_debs/SHA256SUMS").exists() else None,
    "python_system": platform.python_version(),
    "isaac_python_packages": sh(f"'{ISAAC}/python.sh' -m pip list 2>/dev/null | grep -iE "
                                "'^(numpy|scipy|opencv-python-headless|pymavlink|pegasus-simulator) '").splitlines(),
}
out = ROOT / "results/g0/env_manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(m, indent=2))
print(json.dumps({k: (v if not isinstance(v, dict) else v.get("describe", v)) for k, v in m.items()
                  if k not in ("glim_debs_sha256", "isaac_python_packages")}, indent=1))
