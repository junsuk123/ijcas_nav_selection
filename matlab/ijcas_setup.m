function report = ijcas_setup(cfg)
%IJCAS_SETUP Check the environment and record the experiment configuration (MATLAB R2025a).
%   report = ijcas_setup(cfg) checks the MATLAB release, Python packages, Isaac Sim, PX4, ROS 2,
%   OpenVINS, FAST-LIO2 and the fusion checkpoints, records git commit, seeds and the experiment id,
%   and writes <results>/setup_<id>.json.  Failed checks are reported, not hidden.
arguments
    cfg struct = ijcas_config()
end
r = struct();
r.experiment_id = cfg.experiment_id;
r.matlab_version = string(version);
r.matlab_release = string(version("-release"));
r.matlab_release_ok = ~isMATLABReleaseOlderThan("R" + cfg.matlab_release_required);
v = ver;
names = string({v.Name});
r.toolboxes = struct("ros", any(names == "ROS Toolbox"), "navigation", any(names == "Navigation Toolbox"), ...
    "sensor_fusion", any(names == "Sensor Fusion and Tracking Toolbox"));

[s, o] = system(cfg.python + " -c ""import torch, rdflib, scipy, numpy; print(torch.__version__, torch.cuda.is_available())""");
r.python = struct("ok", s == 0, "detail", strtrim(string(o)));
r.isaac_sim = isfolder(cfg.isaac);
r.px4 = isfile(cfg.px4_bin);
r.openvins = isfile(cfg.openvins_bin);
r.fastlio = isfile(cfg.fastlio_bin);
[s, o] = system("bash -lc 'source " + cfg.ros_setup + " && printenv ROS_DISTRO'");
r.ros2 = struct("ok", s == 0, "distro", strtrim(string(o)));
[s, o] = system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader");
r.gpu = strtrim(string(o));
[~, o] = system("git -C '" + cfg.root + "' rev-parse --short HEAD");
r.git_commit = strtrim(string(o));
[~, o] = system("git -C '" + cfg.root + "' rev-parse --abbrev-ref HEAD");
r.git_branch = strtrim(string(o));
[~, o] = system("git -C '" + cfg.root + "' status --porcelain | wc -l");
r.git_uncommitted_files = str2double(strtrim(o));
ck = dir(fullfile(cfg.results, cfg.set, "runs", "*", "seed*", "model.pt"));
r.fusion_checkpoints = numel(ck);
r.set = cfg.set;
r.seeds = cfg.seeds;
r.variants = cfg.variants;
r.data = cfg.data;

checks = ["matlab_release_ok", "isaac_sim", "px4", "openvins", "fastlio"];
r.failed = strings(0);
for c = checks
    if ~r.(c), r.failed(end+1) = c; end %#ok<AGROW>
end
if ~r.python.ok, r.failed(end+1) = "python"; end
if ~r.ros2.ok, r.failed(end+1) = "ros2"; end
if ~isfolder(cfg.results), mkdir(cfg.results); end
f = fullfile(cfg.results, "setup_" + cfg.experiment_id + ".json");
fid = fopen(f, "w");
fprintf(fid, "%s", jsonencode(r, PrettyPrint=true));
fclose(fid);
fprintf("[ijcas] MATLAB %s (required R%s): %s\n", r.matlab_release, cfg.matlab_release_required, string(r.matlab_release_ok));
if isempty(r.failed)
    fprintf("[ijcas] setup ok -> %s\n", f);
else
    warning("ijcas:setup", "failed checks: %s (see %s)", strjoin(r.failed, ", "), f);
end
report = r;
end
