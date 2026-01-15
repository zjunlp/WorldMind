require 'utils'

function model.setColor(rgb)
    local objs=sim.getObjectsInTree(model.handles.boxes,sim.handle_all,1)
    for i=1,#objs,1 do
        sim.setShapeColor(objs[i],nil,sim.colorcomponent_ambient_diffuse,rgb)
    end
end

function model.updateConveyor()
end

function model.updateThermoformer()
    local data=model.readInfo()
    local stationS={data.thermo_rowColCnt[1]*data.thermo_rowColStep[1],data.thermo_rowColCnt[2]*data.thermo_rowColStep[2],data.thermo_extrusionSize[3]+data.thermo_wallThickness}
    local baseS=model.getModelSize()
    model.setShapeSize(model.handles.base,baseS[1],baseS[2],baseS[3])
    sim.setObjectPosition(model.handles.base,model.handle,{0,0,-baseS[3]*0.5})
    sim.setObjectPosition(model.handles.trigger,model.handle,{-baseS[1]/2,-baseS[2]/2,0})
    local objs=sim.getObjectsInTree(model.handles.otherStations,sim.handle_all,1)
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    local objs=sim.getObjectsInTree(model.handles.boxes,sim.handle_all,1)
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    local off=-(data.thermo_stationCnt-1)*0.5*(stationS[1]+data.thermo_stationSpacing)
    sim.setObjectPosition(model.handles.station,model.handle,{off,0,-stationS[3]*0.5})
    model.setShapeSize(model.handles.station,stationS[1]-0.002,stationS[2]-0.002,stationS[3]-0.002)
    for i=1,data.thermo_stationCnt-1,1 do
        off=off+stationS[1]+data.thermo_stationSpacing 
        local h=sim.copyPasteObjects({model.handles.station},0)[1]
        sim.setObjectParent(h,model.handles.otherStations,true)
        sim.setObjectPosition(h,model.handle,{off,0,-stationS[3]*0.5})
    end
    model.createOriginPallet()
    
    
--    local borderHeight=conf.thermo_borderHeight
    local bitCoded=data.thermo_bitCoded
--    local wt=conf.thermo_wallThickness


end

function model.createOriginPallet(currentDispl)
    if not currentDispl then
        currentDispl=0
    end
    local data=model.readInfo()
    local stationS={data.thermo_rowColCnt[1]*data.thermo_rowColStep[1],data.thermo_rowColCnt[2]*data.thermo_rowColStep[2],data.thermo_extrusionSize[3]+data.thermo_wallThickness}
    local off={0,0,0}
    off[1]=-(data.thermo_stationCnt-1)*0.5*(stationS[1]+data.thermo_stationSpacing)-(data.thermo_rowColCnt[1]-1)*0.5*data.thermo_rowColStep[1]
    off[2]=-(data.thermo_rowColCnt[2]-1)*0.5*data.thermo_rowColStep[2]
    local bs={data.thermo_extrusionSize[1]+data.thermo_wallThickness*2,data.thermo_extrusionSize[2]+data.thermo_wallThickness*2,data.thermo_extrusionSize[3]+data.thermo_wallThickness}
    for i=1,data.thermo_rowColCnt[1],1 do
        for j=1,data.thermo_rowColCnt[2],1 do
            local xy={off[1]+(i-1)*data.thermo_rowColStep[1],off[2]+(j-1)*data.thermo_rowColStep[2]}
            local h=createOpenBox(bs,data.thermo_wallThickness,data.thermo_wallThickness,1000,1,true,true,data.thermo_color)
            sim.setObjectProperty(h,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
            sim.setObjectParent(h,model.handles.boxes,true)
            sim.setObjectOrientation(h,sim.handle_parent,{0,0,0})
            sim.setObjectPosition(h,model.handle,{xy[1],xy[2],-bs[3]*0.5})
            local t={stationIndex=0,initX=xy[1]-currentDispl}
            sim.writeCustomDataBlock(h,'thermoformerOpenBox',sim.packTable(t))
        end
    end
end

----------------------------------------------------------------------------------------------------------------------

function model.getModelSize()
    local data=model.readInfo()
    local stationS={data.thermo_rowColCnt[1]*data.thermo_rowColStep[1],data.thermo_rowColCnt[2]*data.thermo_rowColStep[2],data.thermo_extrusionSize[3]+data.thermo_wallThickness}
    local s={1,1,1}
    s[1]=(stationS[1]+data.thermo_stationSpacing)*data.thermo_stationCnt-data.thermo_stationSpacing 
    s[2]=stationS[2]
    s[3]=stationS[3]
    return s
end

function model.setShapeSize(h,x,y,z)
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

function model.getAvailableSensors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.BINARYSENSOR)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableMasterConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
            if data then
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.getAvailableOutputboxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    local isSelectedBoxAvailable = false
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.OUTPUTBOX)
            if data then
                local connectionHandle,p=simBWF.getInputOutputBoxConnectedItem(l[i])
                if connectionHandle>=0 then
                    retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
                    if l[i] == simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX) then
                        isSelectedBoxAvailable = true
                    end
                end
            end
        end
    end
    if not isSelectedBoxAvailable then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX,-1)
    end
    return retL
end

function model.getConnectedRobotAndChannel()
    local allRobots=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    for i=1,#allRobots,1 do
        local convs=simBWF.callCustomizationScriptFunction("model.ext.getConnectedConveyors",allRobots[i])
        for j=1,#convs,1 do
            if convs[j]==model.handle then
                return allRobots[i],j
            end
        end
    end
    return -1,-1
end

function model.displayPallet()
    if model.drawingContainer_pallet then
        sim.addDrawingObjectItem(model.drawingContainer_pallet,nil)
    end
    local palletHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
    if palletHandle>=0 then
        if model.drawingContainer_pallet==nil then
            model.drawingContainer_pallet=sim.addDrawingObject(sim.drawing_spherepoints,0.03,0,-1,999,{1,0,1})
        end
        local data=simBWF.readPalletInfo(palletHandle)
        local m=sim.getObjectMatrix(model.handles.trigger,-1)
        local c=model.readInfo()
        if sim.getBoolParameter(sim.boolparam_online_mode) then
            m[4]=m[4]+c.detectionOffset[2][1]
            m[8]=m[8]+c.detectionOffset[2][2]
            m[12]=m[12]+c.detectionOffset[2][3]
        else
            m[4]=m[4]+c.detectionOffset[1][1]
            m[8]=m[8]+c.detectionOffset[1][2]
            m[12]=m[12]+c.detectionOffset[1][3]
        end
        for i=1,#data.palletItemList,1 do
            local pt={data.palletItemList[i].locationX,data.palletItemList[i].locationY,data.palletItemList[i].locationZ}
            pt=sim.multiplyVector(m,pt)
            sim.addDrawingObjectItem(model.drawingContainer_pallet,pt)
        end
    else
        if model.drawingContainer_pallet then
            sim.removeDrawingObject(model.drawingContainer_pallet)
            model.drawingContainer_pallet=nil
        end
    end
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()

    local info=model.readInfo()
    -- Following for backward compatibility, around middle of 2017:
    if info['stopTrigger'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['stopTrigger']))
        info['stopTrigger']=nil
    end
    if info['startTrigger'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['startTrigger']))
        info['startTrigger']=nil
    end
    if info['masterConveyor'] then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR,sim.getObjectHandle_noErrorNoSuffixAdjustment(info['masterConveyor']))
        info['masterConveyor']=nil
    end
    ----------------------------------------
    model.writeInfo(info)

    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
    model.displayPallet()
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
    local conf=model.readInfo()
    conf['encoderDistance']=0
    conf['stopRequests']={}
    model.writeInfo(conf)
    local objs=sim.getObjectsInTree(model.handles.boxes,sim.handle_all,1)
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    model.createOriginPallet()
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    if model.drawingContainer_pallet then
        sim.removeDrawingObject(model.drawingContainer_pallet)
    end
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
