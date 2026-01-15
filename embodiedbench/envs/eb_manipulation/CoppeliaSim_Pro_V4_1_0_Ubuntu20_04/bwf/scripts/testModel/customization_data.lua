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
    data.message=c.myStoredString
    data.connectedTestModelHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONNECTION)
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('testModel_update',data)
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


