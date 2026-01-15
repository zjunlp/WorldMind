function sysCall_beforeMainScript()
    -- Called before the main script is run.
    -- Can be used to step a simulation in a custom manner.
    local outData={doNotRunMainScript=false} -- when true, then the main script won't be executed
    return outData
end
