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
    data.conveyorId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
    data.inputObjectId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    data.palletId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
    data.calibrationBallDistance=c.calibrationBallDistance
    --[[
    data.deviceId=''
    if c.deviceId~=simBWF.NONE_TEXT then
        data.deviceId=c.deviceId
    end
    --]]
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        data.detectionOffset=c.detectionOffset[2]
    else
        data.detectionOffset=c.detectionOffset[1]
    end
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('ragnarSensor_update',data)
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
