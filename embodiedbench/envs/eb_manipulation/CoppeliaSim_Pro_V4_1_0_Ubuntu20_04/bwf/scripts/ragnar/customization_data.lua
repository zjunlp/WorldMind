function model.removeFromPluginRepresentation()
    -- Destroy/remove the plugin's counterpart to the simulation model
    
    local data={}
    data.id=model.handle
    simBWF.query('object_delete',data)
    model._previousPackedPluginData=nil
    model._previousPackedPluginData_dynParams=nil
end

function model.updatePluginRepresentation()
    -- Create or update the plugin's counterpart to the simulation model
    
    local c=model.readInfo()
    local data={}
    data.id=model.handle
    data.name=simBWF.getObjectAltName(model.handle)
    data.pos=sim.getObjectPosition(model.handles.ragnarRef,-1)
    data.quat=sim.getObjectQuaternion(model.handles.ragnarRef,-1)
    data.alias=c.robotAlias
    data.primaryArmLength=c.primaryArmLengthInMM/1000
    data.secondaryArmLength=c.secondaryArmLengthInMM/1000
    data.platformId=model.platform
    data.pickTrackingWindows={}
    data.placeTrackingWindows={}
    data.pickFrames={}
    data.placeFrames={}
    data.robotAlias=c.robotAlias
    data.motorType=c.motorType
    for i=1,C.CIC,1 do
        data.pickTrackingWindows[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)
        data.placeTrackingWindows[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)
        data.pickFrames[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)
        data.placeFrames[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)
        -- trigger a window/location frame plugin update (the robot pose might have changed):
        if data.pickTrackingWindows[i]~=-1 then
            simBWF.callCustomizationScriptFunction('model.ext.associatedRobotChangedPose',data.pickTrackingWindows[i])
        end
        if data.placeTrackingWindows[i]~=-1 then
            simBWF.callCustomizationScriptFunction('model.ext.associatedRobotChangedPose',data.placeTrackingWindows[i])
        end
    end
    
--    data.conveyors={simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1),simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR2)}
    data.inputs={}
    data.outputs={}
    for i=1,8,1 do
        data.inputs[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)
        data.outputs[i]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)
    end

    data.wsBox=c.wsBox
    data.speed=c.maxVel
    data.accel=c.maxAccel
    data.waitingLocationAfterPick=c.waitLocAfterPickOrPlace[1]
    data.waitingLocationAfterPlace=c.waitLocAfterPickOrPlace[2]
    -- Following not really transmitted, but required to track pose change
    data.robotPose={sim.getObjectPosition(model.handle,-1),sim.getObjectOrientation(model.handle,-1)}
    data.pickWithoutTargetInSight=(sim.boolAnd32(c.jobBitCoded,2048)~=0)
    data.ignorePartDestinations=(sim.boolAnd32(c.jobBitCoded,4096)~=0)
    
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginData then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginData=packedData
        simBWF.query('ragnar_update',data)
    end
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    -- Make sure stored jobs are consistent with current scene:
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local gripper=model.getGripper()
    local data=model.readInfo()
    simBWF.handleJobConsistencyInObjectReferences(model.handle,model.objRefJobInfo,data.jobData)
    model.writeInfo(data)
    if (not simBWF.isJobDataConsistent(data.jobData)) or removeJobsExceptCurrent then
        -- Remove all stored models:
        local serializedJobModels=model.readJobModelInfo()
        -- Remove all jobs in this model:
        data.jobData.jobs={}
        -- Set-up job data to be identical for all jobs:
        local jobNames=simBWF.getAllJobNames()
        local objRefs=simBWF.readObjectReferencesForSpecificJob(model.handle,model.objRefJobInfo,1)
        sim.setReferencedHandles(model.handle,{})
        for i=1,#jobNames,1 do
            local newJob={jobIndex=i}
            data.jobData.jobs[jobNames[i]]=newJob
            newJob.platform={}
            newJob.gripper={}
            if gripper>=0 then
                newJob.gripper.id=sim.getObjectStringParameter(gripper,sim.objstringparam_unique_id)
                newJob.gripper.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',gripper))
                newJob.gripper.altName=sim.getObjectName(gripper+sim.handleflag_altname)
            end
            if model.platform>=0 then
                newJob.platform.id=sim.getObjectStringParameter(model.platform,sim.objstringparam_unique_id)
                newJob.platform.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',model.platform))
                newJob.platform.altName=sim.getObjectName(model.platform+sim.handleflag_altname)
            end
            model.copyModelParamsToJob(data,jobNames[i])
            simBWF.writeObjectReferencesForSpecificJob(model.handle,objRefs,i)
        end
        data.jobData.activeJobInModel=model.currentJob
        model.writeInfo(data)
        
        -- Just make sure we have also serialized the current platform and/or gripper models:
        model.updateJobDataFromCurrentSituation(model.currentJob)
    else
        -- Switch to the correct job if needed (could have been copied before switching job, but pasted after switching job)
        local jobNames=simBWF.getAllJobNames()
        if #jobNames>1 then
            if data.jobData.activeJobInModel~=model.currentJob then
                model.switchFromJobToCurrent(data.jobData.activeJobInModel)
            end
        else
            data.jobData.activeJobInModel=model.currentJob
            model.writeInfo(data)
        end
        
        -- Just in case, remove any non-referenced stored model:
        model.removeNonReferencedJobModels()
    end
    
    printJobDebugInfo()
end

function model.createNewJob()
    -- Job was created by the system. Reflect changes in this model:
    -- 1. Create a new job:
    local oldJobName=model.currentJob
    local newJobName=sim.getStringParameter(sim.stringparam_job)
    model.currentJob=newJobName
    model.updateJobDataFromCurrentSituation(oldJobName)
    local data=model.readInfo()
    local objRef_jobIndex=simBWF.createNewJobInObjectReferences(model.handle,model.objRefJobInfo,data.jobData)
    data.jobData.jobs[newJobName]={jobIndex=objRef_jobIndex,platform={},gripper={}}
    if model.gripper>=0 then
        local id=sim.getObjectStringParameter(model.gripper,sim.objstringparam_unique_id)
        data.jobData.jobs[newJobName].gripper.id=id
        data.jobData.jobs[newJobName].gripper.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',model.gripper))
        data.jobData.jobs[newJobName].gripper.altName=sim.getObjectName(model.gripper+sim.handleflag_altname)
    else
        data.jobData.jobs[newJobName].gripper={}
    end
    if model.platform>=0 then
        local id=sim.getObjectStringParameter(model.platform,sim.objstringparam_unique_id)
        data.jobData.jobs[newJobName].platform.id=id
        data.jobData.jobs[newJobName].platform.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',model.platform))
        data.jobData.jobs[newJobName].platform.altName=sim.getObjectName(model.platform+sim.handleflag_altname)
    else
        data.jobData.jobs[newJobName].platform={}
    end
    model.copyModelParamsToJob(data,newJobName)
    model.writeInfo(data)
    -- 2. Switch to it:
    model.switchFromJobToCurrent(oldJobName)
    printJobDebugInfo()
end

function model.deleteJob()
    -- Job was deleted by the system. Reflect changes in this model:
    -- 1. Switch to current job:
    local oldJobName=model.currentJob
    model.switchFromJobToCurrent(oldJobName)
    -- 2. Delete previous job:
    local data=model.readInfo()
    simBWF.deleteJobInObjectReferences(model.handle,model.objRefJobInfo,data.jobData,oldJobName)
    data.jobData.jobs[oldJobName]=nil
    model.writeInfo(data)
    model.removeNonReferencedJobModels()
    printJobDebugInfo()
end

function model.renameJob()
    -- Job was renamed by the system. Reflect changes in this model:
    local oldJobName=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local data=model.readInfo()
    data.jobData.jobs[model.currentJob]=data.jobData.jobs[oldJobName]
    data.jobData.jobs[oldJobName]=nil
    data.jobData.activeJobInModel=model.currentJob
    model.writeInfo(data)
    printJobDebugInfo()
end

function model.switchJob()
    -- Switch job menu bar cmd
    model.switchFromJobToCurrent(model.currentJob)
    printJobDebugInfo()
end
    
function model.removeNonReferencedJobModels()

    -- Create a map with all model ids of serialized models:
    local toRemove={}
    local serializedJobModels=model.readJobModelInfo()
    for id,val in pairs(serializedJobModels) do
        toRemove[id]=true
    end

    -- Go through all jobs, remove the referenced model ids from the map above:
    local data=model.readInfo()
    for job,val in pairs(data.jobData.jobs) do
        if val.gripper.id then
            toRemove[val.gripper.id]=nil -- do not remove that one
        end
        if val.platform.id then
            toRemove[val.platform.id]=nil -- do not remove that one
        end
    end
    
    -- Remove the serialized platform and grippers that are not referenced anymore:
    for id,val in pairs(toRemove) do
        serializedJobModels[id]=nil
    end
    model.writeJobModelInfo(serializedJobModels)
end

function model.switchFromJobToCurrent(oldJobName)
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    
    -- Serialize the platform and gripper if present, update the job data for the old job:
    model.updateJobDataFromCurrentSituation(oldJobName)
    
    -- Remove the platform and gripper model of the old job (the serialized version remains!):
    if model.platform>=0 then
        model.removeAndDeletePlatform()
    end
    
    -- prepare the new job's models (with correct job data):
    local data=model.readInfo()
    
    local oldJ=data.jobData.jobs[oldJobName]
    local currentJ=data.jobData.jobs[model.currentJob]
    -- oldJ.jobIndex should always be 1: we always switch from current to another
    simBWF.swapJobsInObjectReferences(model.handle,model.objRefJobInfo,currentJ.jobIndex)
    model.copyModelParamsToJob(data,oldJobName)
    model.copyJobToModelParams(data,model.currentJob)
    local tmp=oldJ.jobIndex
    oldJ.jobIndex=currentJ.jobIndex
    currentJ.jobIndex=tmp
    
    local serializedJobModels=model.readJobModelInfo()
    local platformId=data.jobData.jobs[model.currentJob].platform.id
    if platformId then
        local newPlatform=sim.loadModel(serializedJobModels[platformId])
        simBWF.callCustomizationScriptFunction('model.ext.replaceInfo',newPlatform,sim.unpackTable(data.jobData.jobs[model.currentJob].platform.info))
        sim.setObjectName(newPlatform+sim.handleflag_altname+sim.handleflag_silenterror,data.jobData.jobs[model.currentJob].platform.altName) -- setting name can fail (normal)
        local gripperId=data.jobData.jobs[model.currentJob].gripper.id
        if gripperId then
            local newGripper=sim.loadModel(serializedJobModels[gripperId])
            simBWF.callCustomizationScriptFunction('model.ext.attachGripper',newPlatform,newGripper)
            simBWF.callCustomizationScriptFunction('model.ext.replaceInfo',newGripper,sim.unpackTable(data.jobData.jobs[model.currentJob].gripper.info))
            sim.setObjectName(newGripper+sim.handleflag_altname+sim.handleflag_silenterror,data.jobData.jobs[model.currentJob].gripper.altName) -- setting name can fail (normal)
        end
        model.attachPlatformToEmptySpot(newPlatform)
        sim.removeObjectFromSelection(sim.handle_all) -- When loaded, a model is automatically selected. Make sure this doesn't happen
    end
    data.jobData.activeJobInModel=model.currentJob
    model.writeInfo(data)
    model.hideHousing(sim.boolAnd32(data.jobBitCoded,8)~=0)
    local s=0
    if data.frameType~=C.FRAMETYPELIST[1] then
        if sim.boolAnd32(data.jobBitCoded,16)~=0 then
            s=2
        else
            s=1
        end
    end
    model.setFrameState(s)
end

function model.updateJobDataFromCurrentSituation(jobName)
    
    -- Serialize the platform and gripper if present:
    local serializedJobModels=model.readJobModelInfo()
    if model.gripper>=0 then
        local id=sim.getObjectStringParameter(model.gripper,sim.objstringparam_unique_id)
        serializedJobModels[id]=sim.saveModel(model.gripper)
    end
    if model.platform>=0 then
        local attachPt
        if model.gripper>=0 then 
            -- detach the gripper temporarily (otherwise it gets serialized together with the platform)
            attachPt=sim.getObjectParent(model.gripper)
            sim.setObjectParent(model.gripper,-1,true)
        end
        local id=sim.getObjectStringParameter(model.platform,sim.objstringparam_unique_id)
        serializedJobModels[id]=sim.saveModel(model.platform)
        if model.gripper>=0 then
            -- reattach the gripper
            sim.setObjectParent(model.gripper,attachPt,true)
        end
    end
    model.writeJobModelInfo(serializedJobModels)
    -- Update the job data:
    local data=model.readInfo()
    if model.gripper>=0 then
        local id=sim.getObjectStringParameter(model.gripper,sim.objstringparam_unique_id)
        data.jobData.jobs[jobName].gripper.id=id
        data.jobData.jobs[jobName].gripper.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',model.gripper))
        data.jobData.jobs[jobName].gripper.altName=sim.getObjectName(model.gripper+sim.handleflag_altname)
    else
        data.jobData.jobs[jobName].gripper={}
    end
    if model.platform>=0 then
        local id=sim.getObjectStringParameter(model.platform,sim.objstringparam_unique_id)
        data.jobData.jobs[jobName].platform.id=id
        data.jobData.jobs[jobName].platform.info=sim.packTable(simBWF.callCustomizationScriptFunction('model.ext.readInfo',model.platform))
        data.jobData.jobs[jobName].platform.altName=sim.getObjectName(model.platform+sim.handleflag_altname)
    else
        data.jobData.jobs[jobName].platform={}
    end
    model.writeInfo(data)
    -- Remove serialized models that are not used anymore:
    model.removeNonReferencedJobModels()

    printJobDebugInfo()
end

function printJobDebugInfo()
    --[[
    print("-----------------")
    local serializedJobModels=model.readJobModelInfo()
    for id,val in pairs(serializedJobModels) do
        print("serialized model: "..id)
    end
    local data=model.readInfo()
    for job,val in pairs(data.jobData.jobs) do
        print("Job: "..job.." :")
        if val.platform.id then
            print("    Platform: "..val.platform.id)
        else
            print("    Platform: none")
        end
        if val.gripper.id then
            print("    Gripper: "..val.gripper.id)
        else
            print("    Gripper: none")
        end
    end
    print("Last stored job: "..data.jobData.activeJobInModel)
    print("Current job: "..model.currentJob)
    print("-----------------")
    --]]
end
