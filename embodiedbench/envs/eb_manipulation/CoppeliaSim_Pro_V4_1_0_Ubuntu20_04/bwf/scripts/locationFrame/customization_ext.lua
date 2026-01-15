model.ext={}

function model.ext.clearCalibration()
    local c=model.readInfo()
    c.calibration=nil
    c.calibrationMatrix=nil
    model.writeInfo(c)
    model.applyCalibrationColor()
    model.updatePluginRepresentation()
end

function model.ext.getItemData_pricing()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='locationFrame'
    obj.frameType='place'
    obj.brVersion=1
    if model.isPick then
        obj.frameType='pick'
    end
    return obj
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

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local robots=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    local present=false
    for i=1,#robots,1 do
        if simBWF.callCustomizationScriptFunction_noError('model.ext.checkIfRobotIsAssociatedWithLocationFrameOrTrackingWindow',robots[i],model.handle) then
            present=true
            break
        end
    end
    local msg=""
    if not present then
        msg="WARNING (set-up): Not referenced by any robot"..nm
    else
        if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)==-1 then
            msg="WARNING (set-up): Has no associated pallet"..nm
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

function model.ext.announceOnlineModeChanged(isNowOnline)
    model.updatePluginRepresentation()
end

function model.ext.getCalibrationDataForCurrentMode()
    local data={}
    local c=model.readInfo()
    local onlSw=sim.getBoolParameter(sim.boolparam_online_mode)
    if c.calibration and onlSw then
        data.realCalibration=true
        data.ball1=c.calibration[1]
        data.ball2=c.calibration[2]
        data.ball3=c.calibration[3]
        data.matrix=c.calibrationMatrix
    else
        data.realCalibration=false
        local rob=model.getAssociatedRobotHandle()
        if rob>=0 then
            local associatedRobotRef=simBWF.callCustomizationScriptFunction('model.ext.getReferenceObject',rob)
            data.ball1=sim.getObjectPosition(model.handles.calibrationBalls[1],associatedRobotRef)
            data.ball2=sim.getObjectPosition(model.handles.calibrationBalls[2],associatedRobotRef)
            data.ball3=sim.getObjectPosition(model.handles.calibrationBalls[3],associatedRobotRef)
            data.matrix=sim.getObjectMatrix(model.handles.calibrationBalls[1],associatedRobotRef)
        end
    end
    return data
end

function model.ext.setLocationFrameIntoOnlineCalibrationPose()
    model.locationFrameNormalM=sim.getObjectMatrix(model.handle,-1)
    model.applyCalibrationData(false)
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
