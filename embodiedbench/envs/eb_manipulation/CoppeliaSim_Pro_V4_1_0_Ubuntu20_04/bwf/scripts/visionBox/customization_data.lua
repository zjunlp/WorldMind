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
    data.version=1
    data.name=simBWF.getObjectAltName(model.handle)
    data.pos=sim.getObjectPosition(model.handle,-1)
    data.quat=sim.getObjectQuaternion(model.handle,-1)
    data.visionServerName=c.visionBoxServerName
    if sim.getBoolParameter(sim.boolparam_online_mode)then
        data.visionJson=c.visionBoxJsonOnline
    else
        data.visionJson=c.visionBoxJsonOffline
    end
    data.cameras={}
    for i=1,C.CAMERACNT,1 do
        data.cameras[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CAMERA1+i-1)
    end
    data.visionWindows={}
    for i=1,C.VISIONWINDOWCNT,1 do
        data.visionWindows[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.VISIONWINDOW1+i-1)
    end
    data.locationFrames={}
    for i=1,C.LOCATIONFRAMECNT,1 do
        data.locationFrames[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.LOCATIONFRAME1+i-1)
    end

    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('visionBox_update',data)
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
