model.ext={}

function model.ext.getItemData_pricing()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='ragnarVision'
    obj.visionType='default'
    obj.brVersion=1
    local dep={}
    local id=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
    if id>=0 then
        dep[#dep+1]=id
    end
--[=[    
    local id=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CAMERA)
    if id>=0 then
        dep[#dep+1]=id
    end
--]=]    
    local id=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    if id>=0 then
        dep[#dep+1]=id
    end
    if #dep>0 then
        obj.dependencies=dep
    end
    return obj
end


function model.ext.getInputObjectHande()
    return simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
end

function model.ext.getAssociatedConveyorHandle()
    return simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
end

function model.ext.checkIfIfModelIsUsedAsCamera(id)
--[=[
    if id>=0 then
        return id==simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CAMERA)
    end
    --]=]
    return false
end

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    if false then -- simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CAMERA)<0 then
        msg="WARNING (set-up): Not associated with any RagnarCamera object"..nm
    else
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)<0 and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)<0 then
            msg="WARNING (set-up): Not associated with any conveyor belt, and has no input"..nm
        else
            local h=simBWF.getModelThatUsesThisModelAsInput(model.handle)
            if h==-1 then
                msg="WARNING (set-up): Not used as input (e.g. by a tracking window)"..nm
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

function model.ext.alignCalibrationBallsWithInputAndReturnRedBall()
    return model.alignCalibrationBallsWithInputAndReturnRedBall()
end

function model.ext.avoidCircularInput(inputItem)
    return model.avoidCircularInput(inputItem)
end

function model.ext.forbidInput(inputItem)
    return model.forbidInput(inputItem)
end

function model.ext.announcePalletWasRenamed()
    model.dlg.refresh()
end

function model.ext.announcePalletWasCreated()
    model.dlg.refresh()
end

function model.ext.announcePalletWasDestroyed()
    model.dlg.refresh()
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
