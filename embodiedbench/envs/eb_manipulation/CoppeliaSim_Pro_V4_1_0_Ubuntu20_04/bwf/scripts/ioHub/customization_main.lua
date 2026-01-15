function model.getAvailableConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableInputBoxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.INPUTBOX)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableOutputBoxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.OUTPUTBOX)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getModelInputOutputConnectionIndex(modelHandle,input)
    -- returns the connection index (1-6) if yes, otherwise -1:
    if modelHandle~=-1 then
        for i=1,8,1 do
            if input then
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)==modelHandle then
                    return i
                end
            else
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)==modelHandle then
                    return i
                end
            end
        end
    end
    return -1
end

function model.disconnectInputOrOutputBoxConnection(modelHandle,input)
    local refreshDlg=false
    if modelHandle~=-1 then
        for i=1,8,1 do
            if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)==modelHandle then
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1,-1)
                refreshDlg=true
                break
            end
            if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)==modelHandle then
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1,-1)
                refreshDlg=true
                break
            end
        end
    end
    if refreshDlg then
        model.dlg.refresh()
    end
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end


function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_beforeSimulation()
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,1)>0 then
        sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,0)
    end
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    model.dlg.removeDlg()
end

--[[
function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.ext.outputPluginRuntimeMessages()
end
--]]

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,1)
    model.dlg.showOrHideDlgIfNeeded()
    model.dlg.updateEnabledDisabledItems()
    model.updatePluginRepresentation()
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
