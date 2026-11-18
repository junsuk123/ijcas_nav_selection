function cfg = ijcas_config()
%IJCAS_CONFIG Paths and settings of the IJCAS fusion experiment (MATLAB R2025a).
%   cfg = ijcas_config() returns a struct used by every ijcas_*.m step.
%   Environment variables (optional): IJCAS_DATA, IJCAS_WS, ISAACSIM_PATH, IJCAS_PYTHON.
%   Simulation, estimators and learning run as native processes (bash, ROS 2, Python); MATLAB
%   orchestrates them through ijcas_sys and reads their JSON/MAT outputs.

here = fileparts(mfilename("fullpath"));
cfg.root = string(fileparts(here));
cfg.matlab_release_required = "2025a";
cfg.data = envOr("IJCAS_DATA", fullfile(getenv("HOME"), "ijcas_data"));
cfg.ws = envOr("IJCAS_WS", fullfile(getenv("HOME"), "ijcas_ws"));
cfg.isaac = envOr("ISAACSIM_PATH", fullfile(getenv("HOME"), "isaacsim", "_build", "linux-x86_64", "release"));
cfg.python = envOr("IJCAS_PYTHON", "/usr/bin/python3");
cfg.ros_setup = "/opt/ros/humble/setup.bash";
cfg.px4_bin = fullfile(cfg.ws, "third_party", "PX4-Autopilot", "build", "px4_sitl_default", "bin", "px4");
cfg.openvins_bin = fullfile(cfg.ws, "install", "ov_msckf", "lib", "ov_msckf", "run_subscribe_msckf");
cfg.fastlio_bin = fullfile(cfg.ws, "install", "fast_lio", "lib", "fast_lio", "fastlio_mapping");
cfg.set = "fusion_dev";                 % run set (tools/run_batch.py --set)
cfg.variants = ["rgat", "gru"];          % learned fusion variants to train / evaluate
cfg.seeds = [0 1 2];
cfg.results = fullfile(cfg.root, "results", "fusion");
cfg.figures = fullfile(cfg.results, "figures");
cfg.logs = fullfile(cfg.results, "logs");
cfg.experiment_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
end

function v = envOr(name, default)
v = string(getenv(name));
if strlength(v) == 0
    v = string(default);
end
end
