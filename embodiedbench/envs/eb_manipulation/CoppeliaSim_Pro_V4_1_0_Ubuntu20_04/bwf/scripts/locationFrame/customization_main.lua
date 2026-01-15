function model.showOrHideCalibrationBalls(show)
    if show then
        sim.setModelProperty(model.handles.calibrationBalls[1],0)
    else
        sim.setModelProperty(model.handles.calibrationBalls[1],sim.modelproperty_not_showasinsidemodel+sim.modelproperty_not_visible)
    end
end

function model.getAssociatedRobotHandle()
    local ragnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    for i=1,#ragnars,1 do
        if simBWF.callCustomizationScriptFunction_noError('model.ext.checkIfRobotIsAssociatedWithLocationFrameOrTrackingWindow',ragnars[i],model.handle) then
            return ragnars[i]
        end
    end
    return -1
end

function model.applyCalibrationData(attach)
    local retVal=false
    local associatedRobot=model.getAssociatedRobotHandle()
    local c=model.readInfo()
    local calData=c['calibration']
    if associatedRobot>=0 then
        local associatedRobotRef=simBWF.callCustomizationScriptFunction('model.ext.getReferenceObject',associatedRobot)
        if calData then
            -- Find the matrix:
            local m=simBWF.getMatrixFromCalibrationBallPositions(calData[1],calData[2],calData[3])
            -- Apply it to this model:
            sim.setObjectMatrix(model.handle,associatedRobotRef,m)
            -- Place the green and blue balls:
            sim.invertMatrix(m)
            sim.setObjectPosition(model.handles.calibrationBalls[2],model.handle,sim.multiplyVector(m,calData[2]))
            sim.setObjectPosition(model.handles.calibrationBalls[3],model.handle,sim.multiplyVector(m,calData[3]))
            retVal=true
            if attach then
                sim.setObjectParent(model.handle,associatedRobotRef,true)
            end
        end
    else
        if calData then
            c['calibration']=nil
            c['calibrationMatrix']=nil
            model.writeInfo(c)
        end
    end
    if not attach then
        sim.setObjectParent(model.handle,-1,true)
    end
    return retVal
end

function model.applyCalibrationColor()
    local associatedRobot=model.getAssociatedRobotHandle()
    local c=model.readInfo()
    local calData=c['calibration']
    local col
    if model.isPick then
        col={0.5,1,0.5}
    else
        col={1,0.5,0.5}
    end
    if associatedRobot>=0 then
        if calData then
            col={1,1,0}
        end
    end
    local obj=sim.getObjectsInTree(model.handles.calibrationBalls[1],sim.object_shape_type,1)
    for i=1,#obj,1 do
        sim.setShapeColor(obj[i],'CALAUX',sim.colorcomponent_ambient_diffuse,col)
    end
end

function sysCall_init()
    model.codeVersion=1

    model.isPick=(model.readInfo()['type']==0)

    model.dlg.init()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
--    local c=model.readInfo()
--    simBWF.printJobData(model.handle,model.objRefJobInfo,c.jobData.jobs)
    
    model.dlg.showOrHideDlgIfNeeded()
    local hideBalls=false
    if sim.getSimulationState()~=sim.simulation_stopped then
        hideBalls=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],2)~=0)
    end
    model.setGreenAndBlueCalibrationBallsInPlace()
    model.showOrHideCalibrationBalls(not hideBalls)
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.ext.outputPluginRuntimeMessages()
end

function sysCall_afterSimulation()
    if model.locationFrameNormalM then
        model.applyCalibrationData(false)
        sim.setObjectMatrix(model.handle,-1,model.locationFrameNormalM)
        model.locationFrameNormalM=nil
    end
    sim.setObjectInt32Parameter(model.handles.frameShape,sim.objintparam_visibility_layer,1)
    model.dlg.showOrHideDlgIfNeeded()
    model.dlg.updateEnabledDisabledItems()
    model.updatePluginRepresentation()
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    model.dlg.removeDlg()
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
