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
    data.type=c['type']
    local dt=model.ext.getCalibrationDataForCurrentMode()
    data.realCalibration=dt.realCalibration
    data.ball1=dt.ball1
    data.ball2=dt.ball2
    data.ball3=dt.ball3
    data.sizes=c['sizes']
    data.offsets=c['offsets']
    data.stopLineDist=c['stopLinePos']
    data.startLineDist=c['startLinePos']
    data.upstreamMarginDist=-c['upstreamMarginPos']
    data.stopLine=sim.boolAnd32(c['bitCoded'],16)>0
    data.inputObjectId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    data.lineControlPartHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PARTTYPE)
    data.calibrationBallOffset=c['calibrationBallOffset']
    data.calibrationBallDistance=c['calibrationBallOffset'][1]
    --data.palletId=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('trackingWindow_update',data)
    end
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    simBWF.handleJobConsistency_generic(removeJobsExceptCurrent)
    model.setSizes() -- Do not forget to update the visual appearance of the tracking window, in case params have changed
end

function model.createNewJob()
    -- Job was created by the system. Reflect changes in this model:
    simBWF.createNewJob_generic()
end

function model.deleteJob()
    -- Job was deleted by the system. Reflect changes in this model:
    simBWF.deleteJob_generic()
    model.setSizes() -- Do not forget to update the visual appearance of the tracking window, in case params have changed
end

function model.renameJob()
    -- Job was renamed by the system. Reflect changes in this model:
    simBWF.renameJob_generic()
end

function model.switchJob()
    -- Job was switched by the system. Reflect changes in this model:
    simBWF.switchJob_generic()
    model.setSizes() -- Do not forget to update the visual appearance of the tracking window, in case params have changed
end
