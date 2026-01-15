function model.setShapeSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function model.getAvailableSensors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.BINARYSENSOR)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableMasterConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
            if data then
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.getAvailableOutputboxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    local isSelectedBoxAvailable = false
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.OUTPUTBOX)
            if data then
                local connectionHandle,p=simBWF.getInputOutputBoxConnectedItem(l[i])
                if connectionHandle>=0 then
                    retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
                    if l[i] == simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX) then
                        isSelectedBoxAvailable = true
                    end
                end
            end
        end
    end
    if not isSelectedBoxAvailable then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX,-1)
    end
    return retL
end

function model.getConnectedRobotAndChannel()
    local allRobots=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    for i=1,#allRobots,1 do
        local convs=simBWF.callCustomizationScriptFunction("model.ext.getConnectedConveyors",allRobots[i])
        for j=1,#convs,1 do
            if convs[j]==model.handle then
                return allRobots[i],j
            end
        end
    end
    local allIOHubs=sim.getObjectsWithTag(simBWF.modelTags.IOHUB,true)
    for i=1,#allIOHubs,1 do
        local convs=simBWF.callCustomizationScriptFunction("model.ext.getConnectedConveyors",allIOHubs[i])
        for j=1,#convs,1 do
            if convs[j]==model.handle then
                return allIOHubs[i],j
            end
        end
    end
    return -1,-1
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()

    local info=model.readInfo()
    -- Following for backward compatibility, around middle of 2017:
    if info['stopTrigger'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['stopTrigger']))
        info['stopTrigger']=nil
    end
    if info['startTrigger'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['startTrigger']))
        info['startTrigger']=nil
    end
    if info['masterConveyor'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['masterConveyor']))
        info['masterConveyor']=nil
    end
    ----------------------------------------
    model.writeInfo(info)

    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.dlg.showOrHideDlgIfNeeded()
    model.ext.outputPluginRuntimeMessages()
end

function sysCall_suspended()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_afterSimulation()
    model.dlg.updateEnabledDisabledItems()
    local conf=model.readInfo()
    conf['encoderDistance']=0
    conf['stopRequests']={}
    model.writeInfo(conf)
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
