function info = ijcas_run_scenario(cfg, opts)
%IJCAS_RUN_SCENARIO Run Isaac Sim + PX4 + OpenVINS + FAST-LIO2 scenarios (MATLAB R2025a).
%   info = ijcas_run_scenario(cfg, Mode="batch")   starts tools/run_batch.py --set cfg.set detached
%          (resumable; one run takes ~5-7 min) and returns the PID file / log.
%   info = ijcas_run_scenario(cfg, Mode="single", Scenario="S00", Seed=901, Scene="open_field",
%          Out="/path/run") runs one scenario synchronously (scripts/run_stack.sh).
%   info = ijcas_run_scenario(cfg, Mode="status") reports the batch progress and whether it runs.
%   info = ijcas_run_scenario(cfg, Mode="stop") stops the batch process recorded in the PID file.
arguments
    cfg struct
    opts.Mode (1,1) string {mustBeMember(opts.Mode, ["batch", "single", "status", "stop"])} = "status"
    opts.Scenario (1,1) string = "S00"
    opts.Seed (1,1) double = 901
    opts.Scene (1,1) string = ""
    opts.Out (1,1) string = ""
end
name = "batch_" + cfg.set;
pidf = fullfile(cfg.logs, name + ".pid");
info = struct("mode", opts.Mode);
switch opts.Mode
    case "batch"
        [st, ~] = ijcas_sys(cfg, cfg.python + " tools/run_batch.py --set " + cfg.set, Background=true, Name=name);
        info.started = st == 0;
        info.pid_file = pidf;
    case "single"
        gen = fullfile(cfg.root, "config", "generated", opts.Scenario + "_" + opts.Seed + "_matlab.yaml");
        py = cfg.python + " -c ""import sys,yaml; sys.path.insert(0,'.'); from ijcas.scenarios import resolve; " + ...
            "c=resolve('" + opts.Scenario + "'," + opts.Seed + "); open('" + gen + "','w').write(yaml.safe_dump(c, sort_keys=False)); print(c['scene'])""";
        [~, scene] = ijcas_sys(cfg, py, Name="resolve");
        scene = strtrim(string(scene));
        if strlength(opts.Scene) > 0, scene = opts.Scene; end
        out = opts.Out;
        if strlength(out) == 0
            out = fullfile(cfg.data, "runs", "matlab", opts.Scenario + "_" + opts.Seed);
        end
        [st, txt] = ijcas_sys(cfg, "bash scripts/run_stack.sh '" + gen + "' " + scene + " " + opts.Seed + " '" + out + "'", ...
            Name="single_" + opts.Scenario + "_" + opts.Seed);
        info.exit_code = st;
        info.out = out;
        info.recorded = isfile(fullfile(out, "record.npz"));
        info.last_line = strtrim(string(txt));
    case "status"
        logf = fullfile(cfg.data, "runs", "batch_" + cfg.set + ".log");
        info.done = 0;
        if isfile(logf)
            L = readlines(logf);
            L = L(strlength(L) > 0);
            info.done = sum(contains(L, "ok=True"));
            info.failed = sum(contains(L, "ok=False"));
            if ~isempty(L), info.last = L(end); end
        end
        info.running = false;
        if isfile(pidf)
            pid = strtrim(fileread(pidf));
            info.running = system("kill -0 " + pid + " 2>/dev/null") == 0;
            info.pid = pid;
        end
    case "stop"
        if isfile(pidf)
            pid = strtrim(fileread(pidf));
            info.stopped = system("kill -TERM " + pid) == 0;
        else
            info.stopped = false;
        end
end
end
