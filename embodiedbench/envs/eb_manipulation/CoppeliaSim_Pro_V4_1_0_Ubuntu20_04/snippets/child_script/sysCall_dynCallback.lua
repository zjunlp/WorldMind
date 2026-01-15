function sysCall_dynCallback(inData)
    -- This function gets called often, so it might slow down the simulation
    --     (this is called twice at each dynamic simulation step, by default 20x more often than a child script)
    -- We have:
    -- inData.passCnt : the current dynamics calculation pass. 1-10 by default. See next item for details.
    -- inData.totalPasses : the number of dynamics calculation passes for each "regular" simulation pass.
    --                      10 by default (i.e. 10*5ms=50ms which is the default simulation time step)
    -- inData.dynStepSize : the step size used for the dynamics calculations (by default 5ms)
    -- inData.afterStep : false when called before, and true after a dynamics step was computed.

    local txt=string.format(" the %ith dynamics calculation step (out of %i steps)",inData.passCnt,inData.totalPasses)
    if inData.afterStep then
        txt="After"..txt
    else
        txt="Before"..txt
    end
    print(txt)
end
