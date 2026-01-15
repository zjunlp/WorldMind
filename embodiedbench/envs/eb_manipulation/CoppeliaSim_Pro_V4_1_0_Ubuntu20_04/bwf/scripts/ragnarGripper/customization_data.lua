function model.removeFromPluginRepresentation()
    -- Destroy/remove the plugin's counterpart to the simulation model
    
    local data={}
    data.id=model.handle
    simBWF.query('object_delete',data)
    model._previousPackedPluginData=nil
end

function model.updatePluginRepresentation()
    -- Create or update the plugin's counterpart to the simulation model
    
    local c=model.readInfo()
    local data={}
    data.id=model.handle
    data.name=simBWF.getObjectAltName(model.handle)
    data.type=c.subtype
    data.stacking=c.stacking
    data.speed=c.pickAndPlaceInfo.speed
    data.accel=c.pickAndPlaceInfo.accel
    data.dwellTime=c.pickAndPlaceInfo.dwellTime
    data.approachHeight=c.pickAndPlaceInfo.approachHeight
    data.useAbsoluteApproachHeight = c.pickAndPlaceInfo.useAbsoluteApproachHeight
    data.departHeight=c.pickAndPlaceInfo.departHeight
    data.offset=c.pickAndPlaceInfo.offset
    data.rounding=c.pickAndPlaceInfo.rounding
    data.nullingAccuracy=c.pickAndPlaceInfo.nullingAccuracy
    --data.freeModeTiming=c.pickAndPlaceInfo.freeModeTiming
    --data.actionModeTiming=c.pickAndPlaceInfo.actionModeTiming
    data.pickActions={}
    for i=1,#c.pickAndPlaceInfo.pickActions,1 do
        local v=c.pickAndPlaceInfo.pickActions[i]
        data.pickActions[i]={cmd=c.pickAndPlaceInfo.actionTemplates[v.name].cmd,dt=v.dt}
    end
    data.multiPickActions={}
    for i=1,#c.pickAndPlaceInfo.multiPickActions,1 do
        local v=c.pickAndPlaceInfo.multiPickActions[i]
        data.multiPickActions[i]={cmd=c.pickAndPlaceInfo.actionTemplates[v.name].cmd,dt=v.dt}
    end
    data.placeActions={}
    for i=1,#c.pickAndPlaceInfo.placeActions,1 do
        local v=c.pickAndPlaceInfo.placeActions[i]
        data.placeActions[i]={cmd=c.pickAndPlaceInfo.actionTemplates[v.name].cmd,dt=v.dt}
    end
    data.relativeToBelt=c.pickAndPlaceInfo.relativeToBelt
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('gripper_update',data)
    end
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    -- Make sure stored jobs are consistent with current scene:


    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.createNewJob()
    -- Create new job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.deleteJob()
    -- Delete current job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.renameJob()
    -- Rename job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.switchJob()
    -- Switch job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end
