function sysCall_init()
    sim.handleSimulationStart()
    sim.openModule(sim.handle_all)
    sim.handleGraph(sim.handle_all_except_explicit,0)
end

function sysCall_actuation()
    -- Update plugin simulation time:
    if simBWF.isSystemOnline() then
        local res, data=simBWF.query('get_online_time',{})
        local time
        if res=='ok' then
            time=data.time
        else
            time=sim.getSimulationTime()
        end
        sim.setFloatSignal('__brOnlineTime__',time)
    else
        local data={}
        data.time=sim.getSimulationTime()+sim.getSimulationTimeStep()
        simBWF.query('time_step',data)
    end
    
    sim.resumeThreads(sim.scriptthreadresume_default)
    sim.resumeThreads(sim.scriptthreadresume_actuation_first)
    sim.launchThreadedChildScripts()
    sim.handleChildScripts(sim.childscriptcall_actuation)
    sim.resumeThreads(sim.scriptthreadresume_actuation_last)
    sim.handleCustomizationScripts(sim.customizationscriptcall_simulationactuation)
    sim.handleModule(sim.handle_all,false)
    sim.handleMechanism(sim.handle_all_except_explicit)
    sim.handleIkGroup(sim.handle_all_except_explicit)
    sim.handleDynamics(sim.getSimulationTimeStep())
    sim.handleMill(sim.handle_all_except_explicit)

    -- Handle ragnar pose (incl. grasping, etc.) correctly:
    sim.handleChildScripts(sim.syscb_customcallback1)
end    

function sysCall_sensing()
    sim.handleSensingStart()
    sim.handleCollision(sim.handle_all_except_explicit)
    sim.handleDistance(sim.handle_all_except_explicit)
    sim.handleProximitySensor(sim.handle_all_except_explicit)
    sim.handleVisionSensor(sim.handle_all_except_explicit)
    sim.resumeThreads(sim.scriptthreadresume_sensing_first)
    sim.handleChildScripts(sim.childscriptcall_sensing)
    sim.resumeThreads(sim.scriptthreadresume_sensing_last)
    sim.handleCustomizationScripts(sim.customizationscriptcall_simulationsensing)
    sim.handleModule(sim.handle_all,true)
    sim.resumeThreads(sim.scriptthreadresume_allnotyetresumed)
    sim.handleGraph(sim.handle_all_except_explicit,sim.getSimulationTime()+sim.getSimulationTimeStep())
end

function sysCall_cleanup()
    sim.resetMilling(sim.handle_all)
    sim.resetMill(sim.handle_all_except_explicit)
    sim.resetCollision(sim.handle_all_except_explicit)
    sim.resetDistance(sim.handle_all_except_explicit)
    sim.resetProximitySensor(sim.handle_all_except_explicit)
    sim.resetVisionSensor(sim.handle_all_except_explicit)
    sim.closeModule(sim.handle_all)
end

function sysCall_suspend()
    sim.handleChildScripts(sim.syscb_suspend)
    sim.handleCustomizationScripts(sim.syscb_suspend)
end

function sysCall_resume()
    sim.handleChildScripts(sim.syscb_resume)
    sim.handleCustomizationScripts(sim.syscb_resume)
end

