function out = ijcas_main(opts)
%IJCAS_MAIN Orchestrate the IJCAS navigation-solution fusion experiment (MATLAB R2025a).
%
%   out = ijcas_main()                                   % setup + run status + inspection + figures
%   out = ijcas_main(Steps=["setup" "build" "train" "evaluate" "plot"], Run="S10_003")
%
%   Steps (in this order when listed):
%     setup     environment and configuration record          (ijcas_setup)
%     run       start the resumable scenario batch of cfg.set  (ijcas_run_scenario, detached)
%     status    batch progress                                  (ijcas_run_scenario Mode="status")
%     inspect   integrity of the recorded runs                  (ijcas_collect)
%     build     fusion tensors                                  (ijcas_build_dataset)
%     train     GT-supervised fusion weights                    (ijcas_train)
%     evaluate  validation metrics, comparison methods          (ijcas_evaluate)
%     plot      figures 1-6                                     (ijcas_plot)
%   Isaac Sim, PX4, the estimators, ROS 2 and PyTorch run as native processes; MATLAB starts them,
%   records PIDs, exit codes and logs (<results>/logs) and reads their JSON / MAT outputs.
arguments
    opts.Steps (1,:) string = ["setup", "status", "inspect", "plot"]
    opts.Set (1,1) string = ""
    opts.Run (1,1) string = ""
    opts.Variant (1,1) string = "rgat"
    opts.Split (1,1) string = "val"
    opts.Device (1,1) string = "cuda"
    opts.Epochs (1,1) double = 0
end
cfg = ijcas_config();
if strlength(opts.Set) > 0, cfg.set = opts.Set; end
out = struct("cfg", cfg);
order = ["setup", "run", "status", "inspect", "build", "train", "evaluate", "plot"];
for s = order(ismember(order, opts.Steps))
    fprintf("\n[ijcas] ===== %s =====\n", s);
    switch s
        case "setup",    out.setup = ijcas_setup(cfg);
        case "run",      out.run = ijcas_run_scenario(cfg, Mode="batch");
        case "status",   out.status = ijcas_run_scenario(cfg, Mode="status"); disp(out.status);
        case "inspect",  out.inspect = ijcas_collect(cfg);
        case "build",    out.build = ijcas_build_dataset(cfg);
        case "train",    out.train = ijcas_train(cfg, Epochs=opts.Epochs, Device=opts.Device);
        case "evaluate", [out.summary, out.table] = ijcas_evaluate(cfg, Split=opts.Split, Device=opts.Device); disp(out.table);
        case "plot",     out.figures = ijcas_plot(cfg, Run=opts.Run, Variant=opts.Variant, Split=opts.Split);
    end
end
end
