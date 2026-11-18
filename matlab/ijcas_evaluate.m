function [S, T] = ijcas_evaluate(cfg, opts)
%IJCAS_EVALUATE Evaluate learned fusion models and comparison methods (MATLAB R2025a).
%   [S, T] = ijcas_evaluate(cfg, Split="val") runs tools/fusion_run.py eval on the split and returns
%   the decoded summary S and a table T (one row per method; learned variants averaged over seeds).
%   Accuracy and covariance consistency are separate columns; nothing here uses the test split unless
%   Split="test" is passed explicitly after the models are frozen.
arguments
    cfg struct
    opts.Split (1,1) string {mustBeMember(opts.Split, ["val", "test"])} = "val"
    opts.Run (1,1) logical = true
    opts.Device (1,1) string = "cuda"
end
if opts.Run
    st = ijcas_sys(cfg, cfg.python + " tools/fusion_run.py eval --set " + cfg.set + " --split " + opts.Split + ...
        " --device " + opts.Device, Name="eval_" + cfg.set + "_" + opts.Split);
    if st ~= 0, error("ijcas:evaluate", "evaluation failed"); end
end
S = jsondecode(fileread(fullfile(cfg.results, cfg.set, opts.Split + "_summary.json")));
keys = ["pos_rmse_m", "vel_rmse_mps", "att_rmse_deg", "pos_p95_m", "rte10_m", "availability", ...
    "nees_pos_mean", "nees_pos_over95", "enc_mean"];
names = strings(0); vals = zeros(0, numel(keys)); contrib = zeros(0, 3);
B = fieldnames(S.baselines);
for i = 1:numel(B)
    P = S.baselines.(B{i}).pooled;
    names(end+1) = string(B{i}); %#ok<AGROW>
    vals(end+1, :) = pick(P, keys); %#ok<AGROW>
    contrib(end+1, :) = vec3(P, "contrib_mean"); %#ok<AGROW>
end
if isfield(S, "learned") && ~isempty(S.learned)
    L = fieldnames(S.learned);
    for i = 1:numel(L)
        seeds = S.learned.(L{i});
        if ~iscell(seeds), seeds = num2cell(seeds); end
        v = zeros(numel(seeds), numel(keys)); c = zeros(numel(seeds), 3);
        for s = 1:numel(seeds)
            v(s, :) = pick(seeds{s}.pooled, keys);
            c(s, :) = vec3(seeds{s}.pooled, "contrib_mean");
        end
        names(end+1) = string(L{i}); %#ok<AGROW>
        vals(end+1, :) = mean(v, 1, "omitnan"); %#ok<AGROW>
        contrib(end+1, :) = mean(c, 1, "omitnan"); %#ok<AGROW>
    end
end
T = array2table(vals, VariableNames=keys, RowNames=names);
T.contrib_gnss_ins = contrib(:, 1);
T.contrib_vio = contrib(:, 2);
T.contrib_lio = contrib(:, 3);
save(fullfile(cfg.results, cfg.set, opts.Split + "_summary.mat"), "T", "S");
end

function v = pick(P, keys)
v = nan(1, numel(keys));
for k = 1:numel(keys)
    if isfield(P, keys(k)) && ~isempty(P.(keys(k))), v(k) = P.(keys(k)); end
end
end

function c = vec3(P, key)
c = nan(1, 3);
if isfield(P, key) && numel(P.(key)) == 3, c = reshape(P.(key), 1, 3); end
end
