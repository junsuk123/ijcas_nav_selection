function st = ijcas_train(cfg, opts)
%IJCAS_TRAIN Train the fusion-weight models with the GT-based fused-state loss (MATLAB R2025a).
%   st = ijcas_train(cfg) trains cfg.variants x cfg.seeds on the pre-registered training split
%   (tools/fusion_run.py train).  Background=true starts it detached and returns immediately.
arguments
    cfg struct
    opts.Background (1,1) logical = false
    opts.Epochs (1,1) double = 0          % 0 = the registered schedule
    opts.Device (1,1) string = "cuda"
end
cmd = cfg.python + " tools/fusion_run.py train --set " + cfg.set + " --variants " + strjoin(cfg.variants, ",") + ...
    " --seeds " + strjoin(string(cfg.seeds), ",") + " --device " + opts.Device;
if opts.Epochs > 0
    cmd = cmd + " --epochs " + opts.Epochs;
end
st = ijcas_sys(cfg, cmd, Background=opts.Background, Name="train_" + cfg.set);
end
