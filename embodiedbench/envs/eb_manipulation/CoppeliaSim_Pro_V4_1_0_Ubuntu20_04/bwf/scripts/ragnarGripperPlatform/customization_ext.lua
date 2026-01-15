model.ext={}

function model.ext.getItemData_pricing()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='ragnarGripperPlatform'
    obj.platformType='default'
    obj.brVersion=1
    local depCnt=1
    local ob=sim.getObjectsInTree(model.handle)
    local dep={}
    for i=1,#ob,1 do
        local data=sim.readCustomDataBlock(ob[i],simBWF.modelTags.RAGNARGRIPPER)
        if data then
            dep[#dep+1]=simBWF.getObjectAltName(ob[i])
            break
        end
    end
    if #dep>0 then
        obj.dependencies=dep
    end
    return obj
end

function model.ext.checkIfPlatformIsAssociatedWithGripper(id)
    if id>=0 then
        return id==sim.getObjectChild(model.handles.gripperAttachmentPoint,0)
    end
    return false
end

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local robots=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    local present=false
    for i=1,#robots,1 do
        if simBWF.callCustomizationScriptFunction_noError('model.ext.checkIfRobotIsAssociatedWithGripperPlatform',robots[i],model.handle) then
            present=true
            break
        end
    end
    local msg=""
    if not present then
        msg="WARNING (set-up): Not attached to any robot"..nm
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

function model.ext.attachGripper(gripperId)
    sim.setObjectParent(gripperId+sim.handleflag_assembly,model.handles.gripperAttachmentPoint,false)
end

function model.ext.getGripper()
    return sim.getObjectChild(model.handles.gripperAttachmentPoint,0)
end

function model.ext.readInfo()
    return model.readInfo()
end

function model.ext.replaceInfo(data)
    model.writeInfo(data) -- We can simply replace the data structure (no need to adjust color/size and similar more troublesome properties)
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
