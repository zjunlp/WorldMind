function sysCall_contactCallback(inData)
    -- Will objects with inData.handle1 and inData.handle2 respond to dynamic collision?
    local retData={}
    retData.ignoreContact=false -- handle contact here
    retData.collisionResponse=true -- shapes will collide

    if inData.engine==sim.physics_bullet then
        retData.bullet={}
        retData.bullet.friction=0
        retData.bullet.restitution=0
    end

    if inData.engine==sim.physics_ode then
        retData.ode={}
        retData.ode.maxContacts=16
        retData.ode.mu=0
        retData.ode.mu2=0
        retData.ode.bounce=0
        retData.ode.bounceVel=0
        retData.ode.softCfm=0
        retData.ode.softErp=0
        retData.ode.motion1=0
        retData.ode.motion2=0
        retData.ode.motionN=0
        retData.ode.slip1=0
        retData.ode.slip2=0
        retData.ode.fDir1={0,0,0}
        local mode=1 -- bit-coded. See below
        -- 1=dContactMu2
        -- 2=dContactFDir1
        -- 4=dContactBounce
        -- 8=dContactSoftERP
        -- 16=dContactSoftCFM
        -- 32=dContactMotion1
        -- 64=dContactMotion2
        -- 128=dContactSlip1
        -- 256=dContactSlip2
        -- 512=dContactApprox1_1
        -- 1024=dContactApprox1_2
        -- 2048=dContactApprox1
        retData.ode.contactMode=mode
    end

    if inData.engine==sim.physics_vortex then
    end

    if inData.engine==sim.physics_newton then
        retData.newton={}
        retData.newton.staticFriction=0
        retData.newton.kineticFriction=0
        retData.newton.restitution=0
    end

    return(retData)
end
