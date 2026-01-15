function model.removeFromPluginRepresentation_template(partHandle)
    local data={}
    data.id=partHandle
    simBWF.query('object_delete',data)
end

function model.updatePluginRepresentation_template(partHandle)
    local data={}
    data.id=partHandle
    data.displayName=simBWF.getObjectAltName(partHandle)
    local geomData=sim.readCustomDataBlock(partHandle,simBWF.modelTags.GEOMETRY_PART)
    if geomData then
        geomData=sim.unpackTable(geomData)
        data.vertices=geomData.vertices
        data.indices=geomData.indices
        data.normals=geomData.normals
    end
    simBWF.query('part_model_update',data)
    
    
    local pData=simBWF.readPartInfo(partHandle)
    local data={}
    data.id=partHandle
    data.name=simBWF.getObjectAltName(partHandle)
    data.overrideGripperSettings=pData.robotInfo.overrideGripperSettings
    data.speed=pData.robotInfo.speed
    data.accel=pData.robotInfo.accel
    data.dwellTime=pData.robotInfo.dwellTime
    data.approachHeight=pData.robotInfo.approachHeight
    data.useAbsoluteApproachHeight = pData.robotInfo.useAbsoluteApproachHeight
    data.departHeight=pData.robotInfo.departHeight
    data.offset=pData.robotInfo.offset
    data.rounding=pData.robotInfo.rounding
    data.nullingAccuracy=pData.robotInfo.nullingAccuracy
    --data.freeModeTiming=pData.robotInfo.freeModeTiming
    --data.actionModeTiming=pData.robotInfo.actionModeTiming
    data.pickActions={}
    for i=1,#pData.robotInfo.pickActions,1 do
        local v=pData.robotInfo.pickActions[i]
        data.pickActions[i]={cmd=pData.robotInfo.actionTemplates[v.name].cmd,dt=v.dt}
    end
    data.multiPickActions={}
    for i=1,#pData.robotInfo.multiPickActions,1 do
        local v=pData.robotInfo.multiPickActions[i]
        data.multiPickActions[i]={cmd=pData.robotInfo.actionTemplates[v.name].cmd,dt=v.dt}
    end    
    data.placeActions={}
    data.relativeToBelt=pData.robotInfo.relativeToBelt
    local dest={}
    for i=simBWF.PART_DESTINATIONFIRST_REF,simBWF.PART_DESTINATIONLAST_REF,1 do
        local h=simBWF.getReferencedObjectHandle(partHandle,i)
        if h>=0 then
            --print(h,sim.getObjectName(h+sim.handleflag_altname))
            dest[#dest+1]=h
        end
    end
    data.destinations=dest
    
    simBWF.query('part_settings',data)
end

function model.removeFromPluginRepresentation()
    local parts=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        model.removeFromPluginRepresentation_template(parts[i])
    end
end

function model.updatePluginRepresentation()
    local parts=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        model.updatePluginRepresentation_template(parts[i])
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
