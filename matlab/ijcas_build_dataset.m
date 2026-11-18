function st = ijcas_build_dataset(cfg, opts)
%IJCAS_BUILD_DATASET Build the fusion tensors of every finished run of cfg.set (MATLAB R2025a).
%   Common frame alignment, causal time propagation, reported covariance blocks, online features and
%   separately stored ground-truth labels (ijcas/fusion/dataset.py via tools/fusion_build_dataset.py).
arguments
    cfg struct
    opts.Workers (1,1) double {mustBeInteger, mustBePositive} = 4
end
st = ijcas_sys(cfg, cfg.python + " tools/fusion_build_dataset.py --set " + cfg.set + " --workers " + opts.Workers, ...
    Name="build_" + cfg.set);
end
