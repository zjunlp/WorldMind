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

function model.getAvailableInputs()
    local thisInfo=model.readInfo()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data1=sim.readCustomDataBlock(l[i],simBWF.modelTags.TRACKINGWINDOW)
            local data2=sim.readCustomDataBlock(l[i],simBWF.modelTags.VISIONWINDOW)
            local data3=sim.readCustomDataBlock(l[i],simBWF.modelTags.RAGNARSENSOR)
            local data4=sim.readCustomDataBlock(l[i],simBWF.modelTags.RAGNARDETECTOR)
            local data5=sim.readCustomDataBlock(l[i],simBWF.modelTags.THERMOFORMER) -- has internal trigger and pallet
            if data1 or data2 or data3 or data4 or data5 then
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.getAvailableParts()
    local repo,partHolder=simBWF.getPartRepositoryHandles()
    if repo then
        local retVal={}
        local l=sim.getObjectsInTree(partHolder,sim.handle_all,1+2)
        for i=1,#l,1 do
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.PART)
            if data then
                data=sim.unpackTable(data)
                retVal[#retVal+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
        return retVal
    end
end

function model.setSizes()
    local c=model.readInfo()
    local stopLine=c['stopLinePos']
    local startLine=c['startLinePos']
    local ump=c['upstreamMarginPos']
    local w=c['sizes'][2]
    local h=c['sizes'][3]
    local offsets=c['offsets']
    local l={offsets[1]+c['sizes'][1],offsets[1]}
    local slEnabled=sim.boolAnd32(c['bitCoded'],16)>0
    if slEnabled then
        local r,lay=sim.getObjectInt32Parameter(model.handles.trackBox1,sim.objintparam_visibility_layer)
        sim.setObjectInt32Parameter(model.handles.stopLineBox,sim.objintparam_visibility_layer,lay)
        if model.handles.startLineBox>=0 then
            sim.setObjectInt32Parameter(model.handles.startLineBox,sim.objintparam_visibility_layer,lay) -- might not exist on old models
        end
    else
        sim.setObjectInt32Parameter(model.handles.stopLineBox,sim.objintparam_visibility_layer,0)
        if model.handles.startLineBox>=0 then
            sim.setObjectInt32Parameter(model.handles.startLineBox,sim.objintparam_visibility_layer,0) -- might not exist on old models
        end
    end
    if model.isPick then
        model.setObjectSize(model.handles.trackBox1,0.1,w,h)
        sim.setObjectPosition(model.handles.trackBox1,model.handle,{l[1]-0.1*0.5,offsets[2]+w*0.5,offsets[3]+h*0.5})

        model.setObjectSize(model.handles.trackBox2,0.1,w,h)
        sim.setObjectPosition(model.handles.trackBox2,model.handle,{l[2]+0.1*0.5,offsets[2]+w*0.5,offsets[3]+h*0.5})

        model.setObjectSize(model.handles.stopLineBox,w+0.005,0.005,h+0.005)
        if model.handles.startLineBox>=0 then
            model.setObjectSize(model.handles.startLineBox,w+0.005,0.005,h+0.005) -- might not exist on old models
        end
        model.setObjectSize(model.handles.upstreamMarginBox,w+0.005,0.005,h+0.005)
    else
        model.setObjectSize(model.handles.trackBox1,0.1-0.005,w-0.005,h-0.005) -- so that we can still see when two same sized pick and place windows overlap
        sim.setObjectPosition(model.handles.trackBox1,model.handle,{l[1]-0.1*0.5,offsets[2]+w*0.5,offsets[3]+h*0.5})

        model.setObjectSize(model.handles.trackBox2,0.1-0.005,w-0.005,h-0.005) -- so that we can still see when two same sized pick and place windows overlap
        sim.setObjectPosition(model.handles.trackBox2,model.handle,{l[2]+0.1*0.5,offsets[2]+w*0.5,offsets[3]+h*0.5})

        model.setObjectSize(model.handles.stopLineBox,w+0.005-0.005,0.005+0.003,h+0.005-0.005)
        if model.handles.startLineBox>=0 then
            model.setObjectSize(model.handles.startLineBox,w+0.005-0.005,0.005+0.003,h+0.005-0.005) -- might not exist on old models
        end
        model.setObjectSize(model.handles.upstreamMarginBox,w+0.005-0.005,0.005+0.003,h+0.005-0.005)
    end
    sim.setObjectPosition(model.handles.stopLineBox,model.handle,{l[2]+stopLine,offsets[2]+w*0.5,offsets[3]+h*0.5})
    if model.handles.startLineBox>=0 then
        sim.setObjectPosition(model.handles.startLineBox,model.handle,{l[2]+startLine,offsets[2]+w*0.5,offsets[3]+h*0.5}) -- might not exist on old models
    end
    sim.setObjectPosition(model.handles.upstreamMarginBox,model.handle,{l[2]-ump,offsets[2]+w*0.5,offsets[3]+h*0.5})
end

function model.showOrHideCalibrationBalls(show)
    if show~=nil then
        if show then
            sim.setModelProperty(model.handles.calibrationBalls[1],0)
        else
            sim.setModelProperty(model.handles.calibrationBalls[1],sim.modelproperty_not_showasinsidemodel+sim.modelproperty_not_visible)
        end
    end
end

function model.getAssociatedRobotHandle()
    local ragnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    for i=1,#ragnars,1 do
        if simBWF.callCustomizationScriptFunction('model.ext.checkIfRobotIsAssociatedWithLocationFrameOrTrackingWindow',ragnars[i],model.handle) then
            return ragnars[i]
        end
    end
    return -1
end

function model.applyCalibrationData()
    -- (we do not modify the pose of the red calibration ball!!)
    local associatedRobot=model.getAssociatedRobotHandle()
    local c=model.readInfo()
    local calData=c['calibration']
    if associatedRobot>=0 then
        local associatedRobotRef=simBWF.callCustomizationScriptFunction('model.ext.getReferenceObject',associatedRobot)
        if calData then
            -- now set the location frame green and blue balls in place:
            local mi=c['calibrationMatrix']
            sim.invertMatrix(mi)
            sim.setObjectPosition(model.handles.calibrationBalls[2],model.handle,sim.multiplyVector(mi,calData[2]))
            sim.setObjectPosition(model.handles.calibrationBalls[3],model.handle,sim.multiplyVector(mi,calData[3]))
        end
    else
        if calData then
            c['calibration']=nil
            c['calibrationMatrix']=nil
            model.writeInfo(c)
        end
    end
end

function model.applyCalibrationColor()
    -- (we do not modify the pose of the red calibration ball!!)
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
    model.dlg.showOrHideDlgIfNeeded()
    local c=model.readInfo()
    local hideBalls=false
    if sim.getSimulationState()~=sim.simulation_stopped then
        hideBalls=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],2)~=0)
    end
    model.showOrHideCalibrationBalls(not hideBalls)
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    model.setGreenAndBlueCalibrationBallsInPlace()
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if model.simJustStarted then
        if simBWF.isSystemOnline() then
            model.applyCalibrationData() -- can potentially change the position/orientation of the robot
        end
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
    sim.setObjectInt32Parameter(model.handles.trackBox1,sim.objintparam_visibility_layer,1)
    sim.setObjectInt32Parameter(model.handles.trackBox2,sim.objintparam_visibility_layer,1)
    sim.setObjectInt32Parameter(model.handles.refFrame,sim.objintparam_visibility_layer,1)
    sim.setObjectInt32Parameter(model.handles.upstreamMarginBox,sim.objintparam_visibility_layer,1)
    local c=model.readInfo()
    local showStartStopLine=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],16)~=0)
    if showStartStopLine then
        sim.setObjectInt32Parameter(model.handles.stopLineBox,sim.objintparam_visibility_layer,1)
        if model.handles.startLineBox>=0 then
            sim.setObjectInt32Parameter(model.handles.startLineBox,sim.objintparam_visibility_layer,1) -- might not exist on old models
        end
    end
    model.dlg.showOrHideDlgIfNeeded()
    model.dlg.updateEnabledDisabledItems()
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    local c=model.readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1)==0)
    if not show then
        sim.setObjectInt32Parameter(model.handles.trackBox1,sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.handles.trackBox2,sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.handles.refFrame,sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.handles.stopLineBox,sim.objintparam_visibility_layer,256)
        if model.handles.startLineBox>=0 then
            sim.setObjectInt32Parameter(model.handles.startLineBox,sim.objintparam_visibility_layer,256) -- might not exist on old models
        end
        sim.setObjectInt32Parameter(model.handles.upstreamMarginBox,sim.objintparam_visibility_layer,256)
    end
    local hideBalls=false
    hideBalls=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],2)~=0)
    model.showOrHideCalibrationBalls(not hideBalls)
--    model.alignCalibrationBallsWithInputAndReturnRedBall()
--    model.setGreenAndBlueCalibrationBallsInPlace()
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
