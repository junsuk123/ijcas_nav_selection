function files = ijcas_plot(cfg, opts)
%IJCAS_PLOT Paper / presentation figures of the fusion study (MATLAB R2025a, English labels).
%   files = ijcas_plot(cfg, Run="S10_003", Variant="rgat", Seed=0, Split="val")
%   Fig 1 architecture, Fig 2 trajectories, Fig 3 errors with fault intervals, Fig 4 fusion weights,
%   contributions and covariance inflation, Fig 5 ontology-constrained graph, Fig 6 method comparison.
%   Per-run data come from tools/fusion_run.py export (MAT v5), the graph from tools/fusion_run.py graph,
%   the comparison from <results>/<set>/<Split>_summary.json.  Figures are written as PNG (300 dpi).
arguments
    cfg struct
    opts.Run (1,1) string = ""
    opts.Variant (1,1) string = "rgat"
    opts.Seed (1,1) double = 0
    opts.Split (1,1) string = "val"
    opts.Device (1,1) string = "cpu"
    opts.Mat (1,1) string = ""            % use an existing export instead of calling Python
end
if ~isfolder(cfg.figures), mkdir(cfg.figures); end
C = struct("gi", [42 120 214]/255, "vio", [235 104 52]/255, "lio", [27 175 122]/255, ...
    "fused", [237 161 0]/255, "gt", [0.1 0.1 0.1], "fault", [0.85 0.85 0.85]);
FS = 14;
files = strings(0);
files(end+1) = fig1(cfg, C, FS);

matf = opts.Mat;
if strlength(matf) == 0 && strlength(opts.Run) > 0
    matf = fullfile(cfg.results, cfg.set, "export_" + opts.Run + "_" + opts.Variant + ".mat");
    ijcas_sys(cfg, cfg.python + " tools/fusion_run.py export --set " + cfg.set + " --run " + opts.Run + ...
        " --variant " + opts.Variant + " --seed " + opts.Seed + " --device " + opts.Device + " --out '" + matf + "'", ...
        Name="export");
end
if strlength(matf) > 0 && isfile(matf)
    D = load(matf);
    files(end+1) = fig2(cfg, D, C, FS);
    files(end+1) = fig3(cfg, D, C, FS);
    files(end+1) = fig4(cfg, D, C, FS);
end
gj = fullfile(cfg.results, "fusion_graph.json");
ijcas_sys(cfg, cfg.python + " tools/fusion_run.py graph --out '" + gj + "'", Name="graph");
files(end+1) = fig5(cfg, jsondecode(fileread(gj)), C, FS);
sf = fullfile(cfg.results, cfg.set, opts.Split + "_summary.json");
if isfile(sf)
    files(end+1) = fig6(cfg, sf, C, FS, opts.Split);
end
fprintf("[ijcas] figures:\n%s\n", strjoin("  " + files, newline));
end

% ------------------------------------------------------------------------------------------------
function f = save_fig(cfg, h, name)
f = fullfile(cfg.figures, name + ".png");
exportgraphics(h, f, Resolution=300);
close(h);
end

function box(ax, x, y, w, h, txt, col, FS)
patch(ax, [x x+w x+w x], [y y y+h y+h], col, FaceAlpha=0.15, EdgeColor=col, LineWidth=1.8);
text(ax, x + w/2, y + h/2, txt, HorizontalAlignment="center", VerticalAlignment="middle", FontSize=FS-3, ...
    Interpreter="none", Color=[0.1 0.1 0.1]);
end

function arrow(ax, p, q, style)
if nargin < 4, style = "-"; end
plot(ax, [p(1) q(1)], [p(2) q(2)], style, Color=[0.3 0.3 0.3], LineWidth=1.4);
d = q - p; d = d / norm(d); n = [-d(2) d(1)];
tip = [q; q - 0.012*d + 0.007*n; q - 0.012*d - 0.007*n];
fill(ax, tip(:,1), tip(:,2), [0.3 0.3 0.3], EdgeColor="none");
end

function f = fig1(cfg, C, FS)
h = figure(Visible="off", Position=[50 50 1800 800]);
ax = axes(h, Position=[0 0 1 1]); hold(ax, "on"); axis(ax, [0 1 0 1]); axis(ax, "off");
box(ax, 0.01, 0.36, 0.14, 0.28, sprintf("Isaac Sim sensors\nIMU 200 Hz\nGNSS 5 Hz\nstereo 20 Hz\nLiDAR 10 Hz"), [0.5 0.5 0.5], FS);
box(ax, 0.19, 0.70, 0.16, 0.14, sprintf("GNSS/INS (PX4 EKF2)\nstate + covariance"), C.gi, FS);
box(ax, 0.19, 0.43, 0.16, 0.14, sprintf("VIO (OpenVINS)\nstate + covariance"), C.vio, FS);
box(ax, 0.19, 0.16, 0.16, 0.14, sprintf("LIO (FAST-LIO2)\nstate + covariance"), C.lio, FS);
box(ax, 0.39, 0.39, 0.15, 0.22, sprintf("Causal alignment\nand time sync\n(common frame)"), [0.4 0.4 0.4], FS);
box(ax, 0.39, 0.70, 0.15, 0.22, sprintf("Ontology:\ndependencies,\nshared IMU,\nreported covariances"), [0.45 0.3 0.6], FS);
box(ax, 0.58, 0.46, 0.17, 0.30, sprintf("Temporal GRU + R-GAT\n(ontology-constrained\ngraph)\nweights w_k\ninflation gamma_k >= 1"), [0.45 0.3 0.6], FS);
box(ax, 0.58, 0.10, 0.17, 0.24, sprintf("Covariance\nIntersection\n(per block,\non SO(3))"), [0.3 0.3 0.3], FS);
box(ax, 0.79, 0.28, 0.20, 0.26, sprintf("Fused navigation\nsolution: position,\nvelocity, attitude,\ncovariance, contributions"), C.fused, FS);
box(ax, 0.79, 0.70, 0.20, 0.18, sprintf("Training only:\nGT-based\nfused-state loss"), [0.6 0.1 0.1], FS);
arrow(ax, [0.15 0.50], [0.19 0.77]); arrow(ax, [0.15 0.50], [0.19 0.50]); arrow(ax, [0.15 0.50], [0.19 0.23]);
for y = [0.77 0.50 0.23], arrow(ax, [0.35 y], [0.39 0.50]); end
arrow(ax, [0.54 0.52], [0.58 0.58]); arrow(ax, [0.54 0.81], [0.58 0.70]);
arrow(ax, [0.54 0.45], [0.58 0.24]); arrow(ax, [0.665 0.46], [0.665 0.34]);
arrow(ax, [0.75 0.22], [0.79 0.36]);
arrow(ax, [0.89 0.70], [0.75 0.66], "--");
f = save_fig(cfg, h, "fig1_architecture");
end

function f = fig2(cfg, D, C, FS)
h = figure(Visible="off", Position=[50 50 1000 850]);
ax = axes(h); hold(ax, "on");
plot(ax, D.gt_pos(:,1), D.gt_pos(:,2), "--", Color=C.gt, LineWidth=2, DisplayName="Ground truth");
names = ["GNSS/INS (PX4 EKF2)", "VIO (OpenVINS)", "LIO (FAST-LIO2)"]; cols = {C.gi, C.vio, C.lio};
for k = 1:3
    p = squeeze(D.pos3(:, k, :)); p(D.valid3(:, k) == 0, :) = NaN;
    plot(ax, p(:,1), p(:,2), Color=cols{k}, LineWidth=1.2, DisplayName=names(k));
end
plot(ax, D.fused_pos(:,1), D.fused_pos(:,2), Color=C.fused, LineWidth=2.5, DisplayName="Proposed fused solution");
axis(ax, "equal"); grid(ax, "on"); xlabel(ax, "East [m]"); ylabel(ax, "North [m]");
title(ax, "Navigation trajectories - " + string(D.run) + " (" + string(D.scenario) + ")", FontWeight="normal");
legend(ax, Location="best"); ax.FontSize = FS;
f = save_fig(cfg, h, "fig2_trajectories_" + string(D.run));
end

function shade(ax, D, C, mode)
if nargin < 4, mode = "patch"; end
if mode == "lines"                      % over opaque area plots: boundaries only
    for i = 1:size(D.faults, 1)
        xline(ax, D.faults(i, :), "--", Color=[0.15 0.15 0.15], LineWidth=1.6, HandleVisibility="off");
    end
    return
end
yl = ylim(ax);
for i = 1:size(D.faults, 1)
    patch(ax, [D.faults(i,1) D.faults(i,2) D.faults(i,2) D.faults(i,1)], [yl(1) yl(1) yl(2) yl(2)], C.fault, ...
        EdgeColor="none", FaceAlpha=0.6, HandleVisibility="off");
end
ylim(ax, yl);
end

function f = fig3(cfg, D, C, FS)
h = figure(Visible="off", Position=[50 50 1400 600]);
ax = axes(h); hold(ax, "on");
cols = {C.gi, C.vio, C.lio}; names = ["GNSS/INS", "VIO", "LIO"];
for k = 1:3
    e = D.err3(:, k); e(D.valid3(:, k) == 0) = NaN;
    plot(ax, D.t, e, Color=cols{k}, LineWidth=1.1, DisplayName=names(k));
end
plot(ax, D.t, D.err_fused, Color=C.fused, LineWidth=2.4, DisplayName="Fused");
yline(ax, 2.0, ":", "alarm 2 m", Color=[0.4 0.4 0.4], LineWidth=1.2, HandleVisibility="off", FontSize=FS-2);
set(ax, YScale="log"); ylim(ax, [0.05 max(50, max(D.err_fused, [], "omitnan"))]);
shade(ax, D, C);
uistack(findobj(ax, Type="line"), "top");
xlabel(ax, "Time [s]"); ylabel(ax, "Position error [m]");
title(ax, "Position error (grey: injected fault interval, evaluation information only)", FontWeight="normal");
legend(ax, Location="northwest"); grid(ax, "on"); ax.FontSize = FS;
f = save_fig(cfg, h, "fig3_errors_" + string(D.run));
end

function f = fig4(cfg, D, C, FS)
h = figure(Visible="off", Position=[50 50 1400 950]);
tl = tiledlayout(h, 3, 1, TileSpacing="compact");
cols = [C.gi; C.vio; C.lio]; names = ["GNSS/INS", "VIO", "LIO"];
ax1 = nexttile(tl); a = area(ax1, D.t, D.weights, LineStyle="none"); for k = 1:3, a(k).FaceColor = cols(k,:); a(k).DisplayName = names(k); end
ylabel(ax1, "w_k"); ylim(ax1, [0 1]); shade(ax1, D, C, "lines"); legend(ax1, Location="eastoutside");
title(ax1, "R-GAT information weights w_k (input of the CI; dashed: injected fault interval)", FontWeight="normal");
ax2 = nexttile(tl); a = area(ax2, D.t, D.contrib, LineStyle="none"); for k = 1:3, a(k).FaceColor = cols(k,:); a(k).DisplayName = names(k); end
ylabel(ax2, "c_k"); ylim(ax2, [0 1]); shade(ax2, D, C, "lines"); legend(ax2, Location="eastoutside");
title(ax2, "Effective contribution to the fused position: tr(P_f w_k P_k^{-1})/3", FontWeight="normal");
ax3 = nexttile(tl); hold(ax3, "on");
for k = 1:3, plot(ax3, D.t, squeeze(D.log10_gamma(:, k, 1)), Color=cols(k,:), LineWidth=1.4, DisplayName=names(k)); end
ylabel(ax3, "log_{10}\gamma_k"); xlabel(ax3, "Time [s]"); shade(ax3, D, C); legend(ax3, Location="eastoutside");
title(ax3, "Learned covariance inflation (\gamma \geq 1) of the reported position covariance", FontWeight="normal");
for ax = [ax1 ax2 ax3], ax.FontSize = FS; grid(ax, "on"); end
f = save_fig(cfg, h, "fig4_weights_" + string(D.run));
end

function f = fig5(cfg, G, C, FS)
h = figure(Visible="off", Position=[50 50 1300 900]);
ax = axes(h);
E = G.edges;
if ~iscell(E), E = num2cell(E); end
src = strings(numel(E), 1); dst = src; prot = false(numel(E), 1);
for i = 1:numel(E)
    src(i) = E{i}.src; dst(i) = E{i}.dst; prot(i) = E{i}.protected;
end
keep = src ~= dst;
g = digraph(src(keep), dst(keep));
p = plot(ax, g, Layout="layered", Sources=["E_gnss" "E_vis" "E_geo" "E_mot"], Sinks="F_nav", ...
    NodeFontSize=FS-2, ArrowSize=10, LineWidth=0.6, EdgeColor=[0.8 0.8 0.8], MarkerSize=9);
if isprop(p, "Interpreter"), p.Interpreter = "none"; else, p.NodeLabel = strrep(p.NodeLabel, "_", "\_"); end
ks = src(keep); kd = dst(keep); pk = prot(keep);
for i = find(pk)'
    highlight(p, ks(i), kd(i), EdgeColor=[0.45 0.3 0.6], LineWidth=2.4);
end
kinds = string(G.kinds); nodes = string(G.nodes);
kc = containers.Map(["env", "sensor", "nav", "fusion"], {[0.5 0.5 0.5], [0.3 0.3 0.3], C.gi, C.fused});
for i = 1:numel(nodes)
    highlight(p, nodes(i), NodeColor=kc(kinds(i)));
end
title(ax, sprintf("Ontology-constrained fusion graph: %d protected (purple), %d gated edges", sum(pk), sum(~pk)), ...
    FontWeight="normal");
subtitle(ax, "attention = message aggregation; fusion weights w_k = separate head output", FontSize=FS-2);
ax.FontSize = FS; axis(ax, "off");
f = save_fig(cfg, h, "fig5_graph");
end

function f = fig6(cfg, sf, C, FS, split)
S = jsondecode(fileread(sf));
B = fieldnames(S.baselines);
names = strings(0); v = zeros(0, 4);
keys = ["pos_rmse_m", "availability", "nees_pos_mean", "enc_mean"];
for i = 1:numel(B)
    names(end+1) = string(B{i}); v(end+1, :) = pick(S.baselines.(B{i}).pooled, keys); %#ok<AGROW>
end
if isfield(S, "learned") && ~isempty(S.learned)
    L = fieldnames(S.learned);
    for i = 1:numel(L)
        seeds = S.learned.(L{i}); if ~iscell(seeds), seeds = num2cell(seeds); end
        m = zeros(numel(seeds), 4);
        for s = 1:numel(seeds), m(s, :) = pick(seeds{s}.pooled, keys); end
        names(end+1) = string(L{i}); v(end+1, :) = mean(m, 1, "omitnan"); %#ok<AGROW>
    end
end
h = figure(Visible="off", Position=[50 50 1600 900]);
tl = tiledlayout(h, 2, 2, TileSpacing="compact");
ttl = ["Position RMSE [m]", "Availability (error <= 2 m over evaluation time)", ...
    "Mean position NEES (consistent ~ 3, log scale)", "Effective number of contributors"];
for j = 1:4
    ax = nexttile(tl);
    barh(ax, categorical(names, names), v(:, j), FaceColor=[0.45 0.45 0.45]);
    ax.TickLabelInterpreter = "none";
    if j == 3
        set(ax, XScale="log"); xlim(ax, [0.1, max(10, max(v(:, 3), [], "omitnan") * 2)]);
        xline(ax, 3, "--", "3", Color=C.fused, LineWidth=1.5);
    end
    if j == 4, xlim(ax, [0 3.2]); end
    title(ax, ttl(j), FontWeight="normal"); ax.FontSize = FS - 1; grid(ax, "on");
end
title(tl, "Fusion methods, " + split + " split (same states, eligibility, reported covariances)", FontSize=FS);
f = save_fig(cfg, h, "fig6_comparison_" + split);
end

function v = pick(P, keys)
v = nan(1, numel(keys));
for k = 1:numel(keys)
    if isfield(P, keys(k)) && ~isempty(P.(keys(k))), v(k) = P.(keys(k)); end
end
end
