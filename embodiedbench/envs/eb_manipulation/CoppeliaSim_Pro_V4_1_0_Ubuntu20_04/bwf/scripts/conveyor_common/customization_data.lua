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
    data.pos=sim.getObjectPosition(model.handle,-1)
    data.quat=sim.getObjectQuaternion(model.handle,-1)
    data.size={c.length,c.width,c.height}
    data.robotId,data.channel=model.getConnectedRobotAndChannel() -- Change to this does not trigger call to updatePluginRepresentation
    data.type=c['subtype']
    data.maxTrackingDistance=c['length']*0.001
    data.calibration=c['calibration']*0.001
    --[[
    data.deviceId=''
    if c.deviceId~=simBWF.NONE_TEXT then
        data.deviceId=c.deviceId
    end
    --]]
    data.startCmd=c.startCmd
    data.stopCmd=c.stopCmd
    data.triggerDistance=c.triggerDistance
    data.outputboxId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX)
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('conveyor_update',data)
    end
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    simBWF.handleJobConsistency_generic(removeJobsExceptCurrent)
end

function model.createNewJob()
    -- Job was created by the system. Reflect changes in this model:
    simBWF.createNewJob_generic()
end

function model.deleteJob()
    -- Job was deleted by the system. Reflect changes in this model:
    simBWF.deleteJob_generic()
end

function model.renameJob()
    -- Job was renamed by the system. Reflect changes in this model:
    simBWF.renameJob_generic()
end

function model.switchJob()
    -- Job was switched by the system. Reflect changes in this model:
    simBWF.switchJob_generic()
end
