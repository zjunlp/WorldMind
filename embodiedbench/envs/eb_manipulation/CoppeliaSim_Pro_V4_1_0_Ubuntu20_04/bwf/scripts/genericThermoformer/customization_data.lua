function model.removeFromPluginRepresentation()
    -- Destroy/remove the plugin's counterpart to the simulation model
    
    -- Following for the conveyor part in the thermoformer:
    local data={}
    data.id=model.handle
    simBWF.query('object_delete',data)
    model._previousPackedPluginData=nil
    
    -- Following for the trigger part in the thermoformer:
    local data={}
    data.id=model.handles.trigger
    simBWF.query('object_delete',data)
    model._previousPackedPluginData_trigger=nil
end

function model.updatePluginRepresentation()
    -- Create or update the plugin's counterpart to the simulation model
    
    -- Following for the conveyor part in the thermoformer:
    local c=model.readInfo()
    local data={}
    data.id=model.handle
    data.name=simBWF.getObjectAltName(model.handle)
    data.pos=sim.getObjectPosition(model.handle,-1)
    data.quat=sim.getObjectQuaternion(model.handle,-1)
    data.size=model.getModelSize()
    data.robotId,data.channel=model.getConnectedRobotAndChannel() -- Change to this does not trigger call to updatePluginRepresentation
    data.type=c['subtype']
    data.maxTrackingDistance=data.size[1]*0.001
    data.calibration=c['calibration']*0.001
    data.startCmd=c.startCmd
    data.stopCmd=c.stopCmd
    data.triggerDistance=c.triggerDistance
    data.outputboxId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX)
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('conveyor_update',data)
    end
    
    -- Following for the trigger part in the thermoformer:
    local c=model.readInfo()
    local data={}
    data.id=model.handles.trigger
    data.name=simBWF.getObjectAltName(model.handles.trigger)
    data.pos=sim.getObjectPosition(model.handles.trigger,-1)
    data.quat=sim.getObjectQuaternion(model.handles.trigger,-1)
    data.conveyorId=model.handle -- simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
    data.inputObjectId=-1 -- simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    data.palletId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
    data.calibrationBallDistance=0 -- c.calibrationBallDistance
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        data.detectionOffset=c.detectionOffset[2]
    else
        data.detectionOffset=c.detectionOffset[1]
    end
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData_trigger then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData_trigger=packedData
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
