model.ext={}

function model.ext.isPickWithoutTargetOverridden()
    local c=model.readInfo()
    return sim.boolAnd32(c['bitCoded'],2048)>0
end

function model.ext.getItemData_pricing()
    local c=model.readInfo()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='ragnar'
    obj.ragnarType='default'
    obj.brVersion=1
    obj.motors=C.MOTORTYPES[c.motorType].pricingText
    obj.exterior=C.EXTERIORTYPES[c.exteriorType].pricingText
    obj.frame=C.FRAMETYPES[c.frameType].pricingText
    obj.primary_arms=c.primaryArmLengthInMM
    obj.secondary_arms=c.secondaryArmLengthInMM
    local dep={}
    for i=1,C.CIC,1 do
        local ids={}
        ids[1]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)
        ids[2]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)
        ids[3]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)
        ids[4]=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)
        for j=1,4,1 do
            if ids[j]>=0 then
                dep[#dep+1]=simBWF.getObjectAltName(ids[j])
            end
        end
    end
    if model.platform>=0 then
        obj.gripper_platform=simBWF.callCustomizationScriptFunction('model.ext.getItemData_pricing',model.platform)
--        dep[#dep+1]=simBWF.getObjectAltName(platform)
        local grip=model.getGripper()
        if grip>=0 then
            obj.gripper=simBWF.callCustomizationScriptFunction('model.ext.getItemData_pricing',grip)
        end
    end
    if #dep>0 then
        obj.dependencies=dep
    end
    obj.software_configuration={}
    obj.software_configuration.ragnar_joint1_base_x=sim.getObjectPosition(model.handles.motorJoints[1],model.handles.ragnarRef)[1]
    obj.software_configuration.ragnar_joint1_base_y=sim.getObjectPosition(model.handles.motorJoints[1],model.handles.ragnarRef)[2]
    obj.software_configuration.ragnar_joint2_base_x=sim.getObjectPosition(model.handles.motorJoints[2],model.handles.ragnarRef)[1]
    obj.software_configuration.ragnar_joint2_base_y=sim.getObjectPosition(model.handles.motorJoints[2],model.handles.ragnarRef)[2]
    obj.software_configuration.ragnar_joint3_base_x=sim.getObjectPosition(model.handles.motorJoints[3],model.handles.ragnarRef)[1]
    obj.software_configuration.ragnar_joint3_base_y=sim.getObjectPosition(model.handles.motorJoints[3],model.handles.ragnarRef)[2]
    obj.software_configuration.ragnar_joint4_base_x=sim.getObjectPosition(model.handles.motorJoints[4],model.handles.ragnarRef)[1]
    obj.software_configuration.ragnar_joint4_base_y=sim.getObjectPosition(model.handles.motorJoints[4],model.handles.ragnarRef)[2]
    obj.software_configuration.ragnar_joint1_base_tilt=model.handles.tiltAdjustmentAngles[1]
    obj.software_configuration.ragnar_joint2_base_tilt=model.handles.tiltAdjustmentAngles[2]
    obj.software_configuration.ragnar_joint3_base_tilt=model.handles.tiltAdjustmentAngles[3]
    obj.software_configuration.ragnar_joint4_base_tilt=model.handles.tiltAdjustmentAngles[4]
    obj.software_configuration.ragnar_joint1_base_pan=model.handles.panAdjustmentAngles[1]
    obj.software_configuration.ragnar_joint2_base_pan=model.handles.panAdjustmentAngles[2]
    obj.software_configuration.ragnar_joint3_base_pan=model.handles.panAdjustmentAngles[3]
    obj.software_configuration.ragnar_joint4_base_pan=model.handles.panAdjustmentAngles[4]
    obj.software_configuration.belt_encoder_E1_enable=false
    obj.software_configuration.belt_encoder_E2_enable=false
    obj.software_configuration.LATCH_E1_enabled=false
    obj.software_configuration.LATCH_E2_enabled=false
    local off=1
    local windowsTmp={}
    for i=1,C.CIC,1 do
        local tmp=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)
        if tmp>=0 then
            windowsTmp[off]=tmp
            off=off+1
        end
        local tmp=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)
        if tmp>=0 then
            windowsTmp[off]=tmp
            off=off+1
        end
    end
    if #windowsTmp>0 then
        obj.software_configuration.belt_encoder_E1_enable=true
        local s=simBWF.callCustomizationScriptFunction('model.ext.getAssociatedSensorDetectorOrVisionHandle',windowsTmp[1])
        obj.software_configuration.LATCH_E1_enabled=(sim.readCustomDataBlock(s,simBWF.modelTags.RAGNARSENSOR)~=nil)
    end
    if #windowsTmp>1 then
        obj.software_configuration.belt_encoder_E2_enable=true
        local s=simBWF.callCustomizationScriptFunction('model.ext.getAssociatedSensorDetectorOrVisionHandle',windowsTmp[2])
        obj.software_configuration.LATCH_E2_enabled=(sim.readCustomDataBlock(s,simBWF.modelTags.RAGNARSENSOR)~=nil)
    end
    return obj
end

function model.ext.checkIfRobotIsAssociatedWithGripperPlatform(id)
    if id>=0 then
        return id==model.platform
    end
    return false
end

function model.ext.checkIfRobotIsAssociatedWithLocationFrameOrTrackingWindow(id)
    for i=1,C.CIC,1 do
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)==id then
            return true
        end
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)==id then
            return true
        end
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)==id then
            return true
        end
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)==id then
            return true
        end
    end
    return false
end

function model.ext.getReferenceObject()
    return model.handles.ragnarRef
end

function model.ext.getConnectedConveyors()
    local retValue={simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1),simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR2)}
    return retValue
end

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    if model.platform<0 then
        msg="WARNING (set-up): Has no attached gripper platform"..nm
    else
        if model.getGripper()<0 then
            msg="WARNING (set-up): Gripper platform has no attached gripper"..nm
        else
            local pickPlace={false,false}
            for i=1,C.CIC,1 do
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)>=0 then
                    pickPlace[1]=true
                end
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)>=0 then
                    pickPlace[2]=true
                end
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)>=0 then
                    pickPlace[1]=true
                end
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)>=0 then
                    pickPlace[2]=true
                end
            end
            if not pickPlace[1] then
                msg="WARNING (set-up): Has no associated pick location frame or pick tracking window"..nm
            else
                if not pickPlace[2] then
                    msg="WARNING (set-up): Has no associated place location frame or place tracking window"..nm
                end
            end
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.ext.outputPluginSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    local data={}
    data.id=model.handle
    local result,msgs=simBWF.query('get_objectSetupMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs.messages[i]..nm
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.ext.outputPluginRuntimeMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    local data={}
    data.id=model.handle
    local result,msgs=simBWF.query('get_objectRuntimeMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs.messages[i]..nm
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.ext.getPlatform()
    return model.getPlatform()
end

function model.ext.getGripper()
    return model.getGripper()
end

function model.ext.isInputBoxConnection(modelHandle)
    return model.getModelInputOutputConnectionIndex(modelHandle,true)
end
    
function model.ext.isOutputBoxConnection(modelHandle)
    return model.getModelInputOutputConnectionIndex(modelHandle,false)
end

function model.ext.disconnectInputOrOutputBoxConnection(modelHandle)
    model.disconnectInputOrOutputBoxConnection(modelHandle)
end

function model.ext.refreshDlg()
    if model.dlg then
        model.dlg.refresh()
    end
end
---------------------------------------------------------------
-- SERIALIZATION (e.g. for replacement of old with new models):
---------------------------------------------------------------

function model.ext.getSerializationData()
    local data={}
    data.objectName=sim.getObjectName(model.handle)
    data.objectAltName=sim.getObjectName(model.handle+sim.handleflag_altname)
    data.matrix=sim.getObjectMatrix(model.handle,-1)
    local parentHandle=sim.getObjectParent(model.handle)
    if parentHandle>=0 then
        data.parentName=sim.getObjectName(parentHandle)
    end
    data.embeddedData=model.readInfo()
    
end

function model.ext.applySerializationData(data)
end
