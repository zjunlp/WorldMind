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

function model.getAvailableInputs()
    local thisInfo=model.readInfo()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data1=sim.readCustomDataBlock(l[i],simBWF.modelTags.RAGNARDETECTOR)
            local data2=sim.readCustomDataBlock(l[i],simBWF.modelTags.VISIONWINDOW)
            local data3=sim.readCustomDataBlock(l[i],simBWF.modelTags.RAGNARSENSOR)
            local data4=sim.readCustomDataBlock(l[i],simBWF.modelTags.TRACKINGWINDOW)
            local data5=sim.readCustomDataBlock(l[i],simBWF.modelTags.THERMOFORMER) -- has internal trigger and pallet
            if data1 or data2 or data3 or data4 or data5 then
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.setObjectSize(h,x,y,z)
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

function model.getObjectSize(h)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    return {sx,sy,sz}
end

function model.setDetectorBoxSizeAndPos()
    local c=model.readInfo()
    local relGreenBallPos=sim.getObjectPosition(model.handles.calibrationBalls[2],model.handles.calibrationBalls[1])
    local relBlueBallPos=sim.getObjectPosition(model.handles.calibrationBalls[3],model.handles.calibrationBalls[1])
    local s={relGreenBallPos[1],math.abs(relBlueBallPos[2]),c.detectorHeight}
    
    local p={s[1]*0.5,relBlueBallPos[2]*0.5,s[3]*0.5+c.detectorHeightOffset}
    -- Do the change only if something will be different:
    local correctIt=false
    local ds=model.getObjectSize(model.handles.detectorBox)
    local pp=sim.getObjectPosition(model.handles.detectorBox,model.handles.calibrationBalls[1])
    for i=1,3,1 do
        if math.abs(ds[i]-s[i])>0.001 or math.abs(pp[i]-p[i])>0.001 then
            correctIt=true
            break
        end
    end
    local ss=model.getObjectSize(model.handles.detectorSensor)
    if math.abs(ss[3]-s[3])>0.001 then
        correctIt=true
    end
    if correctIt then
        model.setObjectSize(model.handles.detectorBox,s[1],s[2],s[3])
        model.setObjectSize(model.handles.detectorSensor,ss[1],ss[2],s[3])
        sim.setObjectPosition(model.handles.detectorBox,model.handles.calibrationBalls[1],p)
        sim.setObjectPosition(model.handles.detectorSensor,model.handles.calibrationBalls[1],{p[1],p[2],s[3]+c.detectorHeightOffset})
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
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    model.setGreenAndBlueCalibrationBallsInPlace()
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    simJustStarted=nil
    model.dlg.showOrHideDlgIfNeeded()
    model.ext.outputPluginRuntimeMessages()
end

function sysCall_suspended()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_afterSimulation()
    model.dlg.updateEnabledDisabledItems()
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,1)>0 then
        sim.setObjectInt32Parameter(model.handles.detectorBox,sim.objintparam_visibility_layer,1)
    end
end

function sysCall_beforeSimulation()
    simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,1)>0 then
        sim.setObjectInt32Parameter(model.handles.detectorBox,sim.objintparam_visibility_layer,0)
    end
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
