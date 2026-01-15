model.ext={}

function model.ext.getItemData_pricing()
    local c=model.readInfo()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='conveyor'
    obj.conveyorType='default'
    obj.brVersion=1
    obj.length=c.length*1000 -- in mm here
    obj.width=c.width*1000 -- in mm here
    return obj
end

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
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

function model.ext.refreshDlg()
    if model.dlg then
        model.dlg.refresh()
    end
end

-- Trigger (i.e. sensor) related:
function model.ext.getAssociatedConveyorHandle()
    return model.handle -- the thermoformer is a specialized conveyor and sensor at the same time
end

function model.ext.getInputObjectHande()
    return -1 -- a thermoformer does not have any input
end

function model.ext.alignCalibrationBallsWithInputAndReturnRedBall()
    return model.handles.trigger
end

function model.ext.avoidCircularInput(inputItem)
--    return model.avoidCircularInput(inputItem)
end

function model.ext.forbidInput(inputItem)
--    return model.forbidInput(inputItem)
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
