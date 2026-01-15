function model.isPartDetected(partHandle)
    local shapesToTest={}
    if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_model)>0 then
        -- We have a single shape which is not a model. Is the shape detectable?
        if sim.boolAnd32(sim.getObjectSpecialProperty(partHandle),sim.objectspecialproperty_detectable_all)>0 then
            shapesToTest[1]=partHandle -- yes, it is detectable
        end
    else
        -- We have a model. Does the model have the detectable flags overridden?
        if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_detectable)==0 then
            -- No, now take all model shapes that are detectable:
            local t=sim.getObjectsInTree(partHandle,sim.object_shape_type,0)
            for i=1,#t,1 do
                if sim.boolAnd32(sim.getObjectSpecialProperty(t[i]),sim.objectspecialproperty_detectable_all)>0 then
                    shapesToTest[#shapesToTest+1]=t[i]
                end
            end
        end
    end
    for i=1,#shapesToTest,1 do
        if sim.checkProximitySensor(model.handles.sensor,shapesToTest[i])>0 then
            return true
        end
    end
    return false
end

function model.getGripperTypeString(data)
    return (data[1]..data[2]..data[3]..'.'..data[4]..data[5]..data[6]..'.'..data[7]..data[8]..data[9])
end

function model.updateAppearance()
    -- We probably have replaced the data stored in this model. Now we need to adjust the model to replace those changes:
    local data=model.readInfo()
    data.subtype=model.getGripperTypeString(data.gripperType)
    model.writeInfo(data)
    if data.gripperType[9]==0 then
        sim.setObjectInt32Parameter(model.handles.nails,sim.objintparam_visibility_layer,0)
    else
        sim.setObjectInt32Parameter(model.handles.nails,sim.objintparam_visibility_layer,1)
    end
    model.dlg.refresh()
end
    
function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
    model.previousParent=sim.getObjectParent(model.handle)
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    local p=sim.getObjectParent(model.handle)
    if p==-1 and p~=model.previousParent then
        -- The gripper was detached and is parentless. Move it a little bit to show that it is not attached anymore:
        local pos=sim.getObjectPosition(model.handle,-1)
        sim.setObjectPosition(model.handle,-1,{pos[1]+0.15,pos[2]+0.15,pos[3]-0.1})
    end
    model.previousParent=p
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
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