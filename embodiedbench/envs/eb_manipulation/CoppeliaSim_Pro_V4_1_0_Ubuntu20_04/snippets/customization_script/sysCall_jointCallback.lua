function sysCall_jointCallback(inData)
    -- This function gets called often, so it might slow down the simulation
    --     (this is called at each dynamic simulation step, by default 10x more often than a child script)
    -- We have:
    -- inData.first : whether this is the first call from the physics engine, since the joint
    --                was initialized (or re-initialized) in it.
    -- inData.revolute : whether the joint associated with this script is revolute or prismatic
    -- inData.cyclic : whether the joint associated with this script is cyclic or not
    -- inData.handle : the handle of the joint associated with this script
    -- inData.lowLimit : the lower limit of the joint associated with this script (if the joint is not cyclic)
    -- inData.highLimit : the higher limit of the joint associated with this script (if the joint is not cyclic)
    -- inData.passCnt : the current dynamics calculation pass. 1-10 by default. See next item for details.
    -- inData.totalPasses : the number of dynamics calculation passes for each "regular" simulation pass.
    --                      10 by default (i.e. 10*5ms=50ms which is the default simulation time step)
    -- inData.currentPos : the current position of the joint
    -- inData.targetPos : the desired position of the joint
    -- inData.errorValue : targetPos-currentPos (with revolute cyclic joints we take the shortest cyclic distance)
    -- inData.effort : the last force or torque that acted on this joint along/around its axis. With Bullet,
    --                 torques from joint limits are not taken into account
    -- inData.dynStepSize : the step size used for the dynamics calculations (by default 5ms)
    -- inData.targetVel : the joint target velocity (as set in the user interface)
    -- inData.maxForce : the joint maximum force/torque (as set in the user interface)
    -- inData.velUpperLimit : the joint velocity upper limit (as set in the user interface)
    --
    -- Make sure that the joint is dynamically enabled, is in force/torque mode, motor enabled and
    -- control loop enabled, otherwise this function won't be called

    if inData.first then
        PID_P=0.1
        PID_I=0
        PID_D=0
        pidCumulativeErrorForIntegralParam=0
    end
    
    -- The control happens here:
    -- 1. Proportional part:
    local ctrl=inData.errorValue*PID_P
    
    -- 2. Integral part:
    if PID_I~=0 then
        pidCumulativeErrorForIntegralParam=pidCumulativeErrorForIntegralParam+inData.errorValue*inData.dynStepSize
    else
        pidCumulativeErrorForIntegralParam=0
    end
    ctrl=ctrl+pidCumulativeErrorForIntegralParam*PID_I
    
    -- 3. Derivative part:
    if not inData.first then
        ctrl=ctrl+(inData.errorValue-pidLastErrorForDerivativeParam)*PID_D/inData.dynStepSize
    end
    pidLastErrorForDerivativeParam=inData.errorValue
    
    -- 4. Calculate the velocity needed to reach the position in one dynamic time step:
    local maxVelocity=ctrl/inData.dynStepSize -- max. velocity allowed.
    if (maxVelocity>inData.velUpperLimit) then
        maxVelocity=inData.velUpperLimit
    end
    if (maxVelocity<-inData.velUpperLimit) then
        maxVelocity=-inData.velUpperLimit
    end
    local forceOrTorqueToApply=inData.maxForce -- the maximum force/torque that the joint will be able to exert

    -- 5. Following data must be returned to CoppeliaSim:
    firstPass=false
    local outData={}
    outData.velocity=maxVelocity
    outData.force=forceOrTorqueToApply
    return outData
end
