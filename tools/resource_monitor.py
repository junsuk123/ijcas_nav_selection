"""G7: per-component CPU / memory and GPU utilisation during a run (1 Hz).

    python3 tools/resource_monitor.py --out $RUN/resources.json --seconds 600
Components are matched on the executable / script name of each process.
"""
import argparse
import json
import subprocess
import time

import psutil

COMPONENTS = {"isaac_sim": "kit/python/bin/python3", "px4": "bin/px4", "openvins": "run_subscribe_msckf",
              "glim": "glim_rosnode", "xrce_agent": "MicroXRCEAgent",
              "context_monitor": "ros2/context_monitor.py", "selector": "ros2/navigation_selector.py",
              "recorder": "ros2/recorder.py"}


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5).stdout.strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:  # noqa: BLE001
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    procs = {}
    samples = []
    end = time.time() + a.seconds
    try:
        while time.time() < end:
            for p in psutil.process_iter(["pid", "cmdline"]):
                cmd = " ".join(p.info["cmdline"] or [])
                for name, key in COMPONENTS.items():
                    if key in cmd and p.pid not in procs:
                        procs[p.pid] = (name, p)
                        p.cpu_percent(None)
            row = {"t": time.time(), "cpu_total": psutil.cpu_percent(None)}
            for pid, (name, p) in list(procs.items()):
                try:
                    row[name] = row.get(name, 0.0) + p.cpu_percent(None)
                    row[f"{name}_rss_mb"] = row.get(f"{name}_rss_mb", 0.0) + p.memory_info().rss / 2 ** 20
                except psutil.Error:
                    procs.pop(pid, None)
            row["gpu_util"], row["gpu_mem_mb"] = gpu()
            samples.append(row)
            time.sleep(1.0)
    except BaseException:
        pass
    json.dump(samples, open(a.out, "w"))


if __name__ == "__main__":
    main()
