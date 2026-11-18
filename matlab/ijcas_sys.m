function [status, out] = ijcas_sys(cfg, cmd, opts)
%IJCAS_SYS Run a shell command from the repository root, log it and return its exit status.
%   [status, out] = ijcas_sys(cfg, cmd) runs synchronously (bash -lc).
%   ijcas_sys(cfg, cmd, Background=true, Name="batch") starts it detached (setsid nohup) and
%   writes <logs>/<Name>.pid and <Name>.log; status is 0 when the process started.
%   ROS 2 is sourced when Ros=true.  MATLAB R2025a.
arguments
    cfg struct
    cmd (1,1) string
    opts.Background (1,1) logical = false
    opts.Name (1,1) string = "cmd"
    opts.Ros (1,1) logical = false
end
if ~isfolder(cfg.logs), mkdir(cfg.logs); end
pre = "cd '" + cfg.root + "' && export IJCAS_DATA='" + cfg.data + "'";
if opts.Ros
    pre = pre + " && source " + cfg.ros_setup;
end
logf = fullfile(cfg.logs, opts.Name + ".log");
if opts.Background
    pidf = fullfile(cfg.logs, opts.Name + ".pid");
    full = pre + " && setsid nohup " + cmd + " > '" + logf + "' 2>&1 < /dev/null & echo $! > '" + pidf + "'";
    [status, out] = system("bash -lc " + quote(full));
    fprintf("[ijcas] started %s (pid file %s)\n", opts.Name, pidf);
else
    fprintf("[ijcas] %s\n", cmd);
    [status, out] = system("bash -lc " + quote(pre + " && " + cmd + " 2>&1 | tee -a '" + logf + "'; exit ${PIPESTATUS[0]}"));
    if status ~= 0
        warning("ijcas:sys", "command failed (exit %d): %s", status, cmd);
    end
end
end

function q = quote(s)
q = "'" + strrep(s, "'", "'\''") + "'";
end
