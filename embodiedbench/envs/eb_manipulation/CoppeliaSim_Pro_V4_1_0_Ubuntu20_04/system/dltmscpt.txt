-- This is the main script. The main script is not supposed to be modified,
-- unless there is a very good reason to do it.
-- Without main script, there is no simulation.
-- A main script marked as "default" (this is the default case) will use the
-- content of following file: system/dltmscpt.txt. This allows your old simulation
-- scenes to be automatically also using newer features, without explicitely coding
-- them. If you modify the main script, it will be marked as "customized", and you
-- won't benefit of that automatic forward compatibility mechanism. 

function sysCall_init()
    sim.handleSimulationStart()
    sim.openModule(sim.handle_all)
    sim.handleGraph(sim.handle_all_except_explicit,0)
end

function sysCall_actuation()
    sim.resumeThreads(sim.scriptthreadresume_default)
    sim.resumeThreads(sim.scriptthreadresume_actuation_first)
    sim.launchThreadedChildScripts()
    sim.handleChildScripts(sim.syscb_actuation)
    sim.resumeThreads(sim.scriptthreadresume_actuation_last)
    sim.handleCustomizationScripts(sim.syscb_actuation)
    sim.handleAddOnScripts(sim.syscb_actuation)
    sim.handleSandboxScript(sim.syscb_actuation)
    sim.handleModule(sim.handle_all,false)
    sim.handleIkGroup(sim.handle_all_except_explicit)
    sim.handleDynamics(sim.getSimulationTimeStep())
end

function sysCall_sensing()
    -- put your sensing code here
    sim.handleSensingStart()
    sim.handleCollision(sim.handle_all_except_explicit)
    sim.handleDistance(sim.handle_all_except_explicit)
    sim.handleProximitySensor(sim.handle_all_except_explicit)
    sim.handleVisionSensor(sim.handle_all_except_explicit)
    sim.resumeThreads(sim.scriptthreadresume_sensing_first)
    sim.handleChildScripts(sim.syscb_sensing)
    sim.resumeThreads(sim.scriptthreadresume_sensing_last)
    sim.handleCustomizationScripts(sim.syscb_sensing)
    sim.handleAddOnScripts(sim.syscb_sensing)
    sim.handleSandboxScript(sim.syscb_sensing)
    sim.handleModule(sim.handle_all,true)
    sim.resumeThreads(sim.scriptthreadresume_allnotyetresumed)
    sim.handleGraph(sim.handle_all_except_explicit,sim.getSimulationTime()+sim.getSimulationTimeStep())
end

function sysCall_cleanup()
    sim.resetCollision(sim.handle_all_except_explicit)
    sim.resetDistance(sim.handle_all_except_explicit)
    sim.resetProximitySensor(sim.handle_all_except_explicit)
    sim.resetVisionSensor(sim.handle_all_except_explicit)
    sim.closeModule(sim.handle_all)
end

function sysCall_suspend()
    sim.handleChildScripts(sim.syscb_suspend)
    sim.handleCustomizationScripts(sim.syscb_suspend)
    sim.handleAddOnScripts(sim.syscb_suspend)
    sim.handleSandboxScript(sim.syscb_suspend)
end

function sysCall_suspended()
    sim.handleChildScripts(sim.syscb_suspended)
    sim.handleCustomizationScripts(sim.syscb_suspended)
    sim.handleAddOnScripts(sim.syscb_suspended)
    sim.handleSandboxScript(sim.syscb_suspended)
end

function sysCall_resume()
    sim.handleChildScripts(sim.syscb_resume)
    sim.handleCustomizationScripts(sim.syscb_resume)
    sim.handleAddOnScripts(sim.syscb_resume)
    sim.handleSandboxScript(sim.syscb_resume)
end

-- You can define additional system calls here:
--[[
function sysCall_beforeCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        sim.addLog(sim.verbosity_scriptinfos,"Object with handle "..key.." will be copied")
    end
end

function sysCall_afterCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        sim.addLog(sim.verbosity_scriptinfos,"Object with handle "..key.." was copied")
    end
end

function sysCall_beforeDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        sim.addLog(sim.verbosity_scriptinfos,"Object with handle "..key.." will be deleted")
    end
    -- inData.allObjects indicates if all objects in the scene will be deleted
end

function sysCall_afterDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        sim.addLog(sim.verbosity_scriptinfos,"Object with handle "..key.." was deleted")
    end
    -- inData.allObjects indicates if all objects in the scene were deleted
end
--]]
