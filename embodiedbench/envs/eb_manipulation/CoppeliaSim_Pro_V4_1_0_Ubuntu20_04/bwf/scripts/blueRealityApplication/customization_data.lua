-------------------------------------------------------
-- PLUGIN COMMUNICATION:
-------------------------------------------------------

function model.removeFromPluginRepresentation_brApp()
    -- Destroy/remove the plugin's counterpart to the simulation model
    
    local data={}
    data.id=model.handle
    simBWF.query('object_delete',data)
    model._previousPackedPluginData_brApp=nil
    model._previousPackedPluginData_gp=nil
end

function model.updatePluginRepresentation_brApp()
    -- Create or update the plugin's counterpart to the simulation model

    local c=model.readInfo()
    local data={}
    data.id=model.handle
    data.code=c.packMlCode
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData_brApp then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData_brApp=packedData
        simBWF.query('packml_update',data)
    end
end

function model.updatePluginRepresentation_generalProperties()
    -- Create or update the plugin's counterpart to the simulation model
    
    local c=model.readInfo()
    local data={}
    data.id=model.handle
    data.masterIp=c.masterIp
    data.sceneName=sim.getStringParameter(sim.stringparam_scene_name)
    data.jobName=model.currentJob
    data.appObjectName=sim.getObjectName(model.handle)
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData_gp then
        model._previousPackedPluginData_gp=packedData
        simBWF.query('generalProperties_update',data)
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
    local sel=sim.getObjectSelection()
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local cmd='createNewJob'
    model.relayJobCommand(cmd,'Creating a new Job...')
    sim.removeObjectFromSelection(sim.handle_all)
    sim.addObjectToSelection(sel)
    model.refreshAllDialogs()
end

function model.deleteJob()
    -- Delete current job menu bar cmd
    local sel=sim.getObjectSelection()
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local cmd='deleteJob'
    model.relayJobCommand(cmd,'Deleting current Job...')
    sim.removeObjectFromSelection(sim.handle_all)
    sim.addObjectToSelection(sel)
    model.refreshAllDialogs()
end

function model.renameJob()
    -- Rename job menu bar cmd
    local sel=sim.getObjectSelection()
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local cmd='renameJob'
    model.relayJobCommand(cmd,nil)
    sim.removeObjectFromSelection(sim.handle_all)
    sim.addObjectToSelection(sel)
    model.refreshAllDialogs()
end

function model.switchJob()
    -- Switch job menu bar cmd
    local sel=sim.getObjectSelection()
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local cmd='switchJob'
    model.relayJobCommand(cmd,'\nSwitching to another Job...\n')
    sim.removeObjectFromSelection(sim.handle_all)
    sim.addObjectToSelection(sel)
    model.refreshAllDialogs()
end

function model.relayJobCommand(cmd,txt,arg1)
    local dlg
    if txt then
        dlg=sim.displayDialog('Job',txt,sim.dlgstyle_message,false,'')
    end
    cmd='model.'..cmd
    local tags={simBWF.modelTags.INPUTBOX,
        simBWF.modelTags.OUTPUTBOX,
        simBWF.modelTags.IOHUB,
        simBWF.modelTags.VISIONBOX,
        simBWF.modelTags.TESTMODEL,
        simBWF.modelTags.LOCATIONFRAME,
        simBWF.modelTags.TRACKINGWINDOW,
        simBWF.modelTags.RAGNAR,
        simBWF.modelTags.CONVEYOR,
        simBWF.modelTags.PARTFEEDER,
        simBWF.modelTags.MULTIFEEDER,
        simBWF.modelTags.PARTSINK,
        simBWF.modelTags.VISIONWINDOW,
        simBWF.modelTags.RAGNARCAMERA,
        simBWF.modelTags.RAGNARSENSOR,
        simBWF.modelTags.RAGNARDETECTOR
    }
    for j=1,#tags,1 do
        local objs=sim.getObjectsWithTag(tags[j],true)
        for i=1,#objs,1 do
            simBWF.callCustomizationScriptFunction_noError(cmd,objs[i],arg1)
        end
    end
    if dlg then
        sim.endDialog(dlg)
    end
end