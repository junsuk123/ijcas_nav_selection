function T = ijcas_collect(cfg)
%IJCAS_COLLECT Integrity of the recorded runs of cfg.set (MATLAB R2025a).
%   T = ijcas_collect(cfg) returns one row per run: DONE marker, manifest error, duration, output
%   counts of PX4 EKF2 / OpenVINS / FAST-LIO2 and the share of outputs that report a positive definite
%   covariance per block (position, velocity, attitude).  Data come from tools/fusion_inspect.py.
arguments
    cfg struct
end
if ~isfolder(cfg.results), mkdir(cfg.results); end
f = fullfile(cfg.results, "inspect_" + cfg.set + ".json");
[st, ~] = ijcas_sys(cfg, cfg.python + " tools/fusion_inspect.py --set " + cfg.set + " --out '" + f + "'", Name="inspect");
if st ~= 0, error("ijcas:collect", "inspection failed"); end
J = jsondecode(fileread(f));
R = J.runs;
if ~iscell(R), R = num2cell(R); end
n = numel(R);
run = strings(n, 1); done = false(n, 1); ok = false(n, 1); sim_s = nan(n, 1);
est = ["px4", "openvins", "fastlio"];
cov = nan(n, 9); outs = zeros(n, 3);
for i = 1:n
    r = R{i};
    run(i) = r.run;
    done(i) = r.done;
    ok(i) = isfield(r, "error") && isempty(r.error) && ~isfield(r, "inspect_error");
    if isfield(r, "sim_s") && ~isempty(r.sim_s), sim_s(i) = r.sim_s; end
    for k = 1:3
        if isfield(r, est(k))
            e = r.(est(k));
            outs(i, k) = e.outputs;
            if isfield(e, "pos_reported")
                cov(i, (k-1)*3 + (1:3)) = [e.pos_reported, e.vel_reported, e.att_reported];
            end
        end
    end
end
T = table(run, done, ok, sim_s, outs(:,1), outs(:,2), outs(:,3), ...
    cov(:,1), cov(:,2), cov(:,3), cov(:,4), cov(:,5), cov(:,6), cov(:,7), cov(:,8), cov(:,9), ...
    VariableNames=["run", "done", "ok", "sim_s", "px4_out", "vio_out", "lio_out", ...
    "px4_pos", "px4_vel", "px4_att", "vio_pos", "vio_vel", "vio_att", "lio_pos", "lio_vel", "lio_att"]);
fprintf("[ijcas] %s: %d runs, %d done, %d without manifest error, %d with all 9 covariance blocks reported >= 99%%\n", ...
    cfg.set, n, sum(done), sum(ok), sum(all(cov >= 0.99, 2)));
end
