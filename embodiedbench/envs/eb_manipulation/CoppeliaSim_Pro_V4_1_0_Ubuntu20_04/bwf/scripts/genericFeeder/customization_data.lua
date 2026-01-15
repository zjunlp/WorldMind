function model.removeFromPluginRepresentation()
    -- Destroy/remove the plugin's counterpart to the simulation model
    
end

function model.updatePluginRepresentation()
    -- Create or update the plugin's counterpart to the simulation model
    
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    simBWF.handleJobConsistency_generic(removeJobsExceptCurrent)
    model.setModelSize() -- to reflect any possible size change
end

function model.createNewJob()
    -- Job was created by the system. Reflect changes in this model:
    simBWF.createNewJob_generic()
end

function model.deleteJob()
    -- Job was deleted by the system. Reflect changes in this model:
    simBWF.deleteJob_generic()
    model.setModelSize() -- to reflect any possible size change
end

function model.renameJob()
    -- Job was renamed by the system. Reflect changes in this model:
    simBWF.renameJob_generic()
end

function model.switchJob()
    -- Job was switched by the system. Reflect changes in this model:
    simBWF.switchJob_generic()
    model.setModelSize() -- to reflect any possible size change
end
