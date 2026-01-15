function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function ext_getItemData_pricing()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.type='trackingWindow'
    obj.windowType='pickOrPlace'
    obj.brVersion=0

    local dep={}
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_INPUT_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    if #dep>0 then
        obj.dependencies=dep
    end
    return obj
end

function createPalletPointsIfNeeded()
    local data=readInfo()
    if #data['palletPoints']==0 then
        data['palletPoints']=simBWF.generatePalletPoints(data)
    end
    writeInfo(data)
end

function updatePalletPoints()
    local data=readInfo()
    if data['palletPattern']~=5 then -- 5 is custom/imported
        data['palletPoints']={} -- remove them
        writeInfo(data)
        createPalletPointsIfNeeded()
    end
end

function setObjectSize(h,x,y,z)
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

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='window'
    end
    if not info['width'] then
        info['width']=0.5
    end
    if not info['length'] then
        info['length']=0.5
    end
    if not info['transferStart'] then
        info['transferStart']=0.3
    end
    if not info['transferLength'] then
        info['transferLength']=0.2
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['stopLinePos'] then
        info['stopLinePos']=0.1
    end
    if not info['stopLineProcessingStage'] then
        info['stopLineProcessingStage']=1
    end
    if not info['bitCoded'] then
        info['bitCoded']=1 -- 1=hidden, 2=show console, 4=showPts, 8=pallet override, 16=stopLine enable
    end
    if not info['trackedItemsInWindow'] then
        info['trackedItemsInWindow']={}
    end
    if not info['trackedTargetsInWindow'] then
        info['trackedTargetsInWindow']={}
    end
    if not info['transferItems'] then
        info['transferItems']={}
    end
    if not info['itemsToRemoveFromTracking'] then
        info['itemsToRemoveFromTracking']={}
    end
    if not info['targetPositionsToMarkAsProcessed'] then
        info['targetPositionsToMarkAsProcessed']={}
    end
    if not info['palletPattern'] then
        info['palletPattern']=0 -- 0=none, 1=single, 2=circular, 3=line (rectangle), 4=honeycomb, 5=custom/imported
    end
    if not info['circularPatternData3'] then
        info['circularPatternData3']={{0,0,0},0.1,6,0,true,1,0.05} -- offset, radius, count,angleOffset, center, layers, layers step
    end
    if not info['customPatternData'] then
        info['customPatternData']=''
    end
    if not info['linePatternData'] then
        info['linePatternData']={{0,0,0},3,0.03,3,0.03,1,0.05} -- offset, rowCnt, rowStep, colCnt, colStep, layers, layers step
    end
    if not info['honeycombPatternData'] then
        info['honeycombPatternData']={{0,0,0},3,0.03,3,0.03,1,0.05,false} -- offset, rowCnt, rowStep, colCnt, colStep, layers, layers step, firstRowOdd
    end
    if not info['palletPoints'] then
        info['palletPoints']={}
    end
    if not info['associatedRobotTrackingCorrectionTime'] then
        info['associatedRobotTrackingCorrectionTime']=0
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    getDefaultInfoForNonExistingFields(data)
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW,'')
    end
end

function getAvailableConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableDetectionOrTrackingWindows()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model then
            local data1=sim.readCustomDataBlock(l[i],simBWF.modelTags.TRACKINGWINDOW)
            local data2=sim.readCustomDataBlock(l[i],'XYZ_DETECTIONWINDOW_INFO')
            if data1 or data2 then
                retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function getAvailableDecorationPatterns()
    local retL={'No decoration','Circular decoration pattern'}
    return retL
end

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local sl=c['stopLinePos']
    local ts=c['transferStart']
    local tl=c['transferLength']
    local h=c['height']
    local slEnabled=sim.boolAnd32(c['bitCoded'],16)>0
    if slEnabled then
        local r,lay=sim.getObjectInt32Parameter(trackBox,sim.objintparam_visibility_layer)
        sim.setObjectInt32Parameter(stopLineBox,sim.objintparam_visibility_layer,lay)
    else
        sim.setObjectInt32Parameter(stopLineBox,sim.objintparam_visibility_layer,0)
    end
    setObjectSize(trackBox,w,l,h)
    sim.setObjectPosition(trackBox,model,{0,0,h*0.5})
    setObjectSize(stopLineBox,w+0.005,0.005,h+0.0025)
    sim.setObjectPosition(stopLineBox,model,{0,sl,h*0.5+0.0025})
    setObjectSize(transferBox,w,tl,h)
    sim.setObjectPosition(transferBox,model,{0,ts+(l+tl)*0.5,h*0.5})
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local config=readInfo()
        local overridePallet=sim.boolAnd32(config['bitCoded'],8)~=0
        local stopLine=sim.boolAnd32(config['bitCoded'],16)~=0
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,20,enabled,true)
        simUI.setEnabled(ui,21,enabled,true)
        simUI.setEnabled(ui,22,enabled,true)
        simUI.setEnabled(ui,23,enabled,true)
        simUI.setEnabled(ui,24,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,4,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
        simUI.setEnabled(ui,19,enabled,true)
        simUI.setEnabled(ui,50,enabled,true)
        simUI.setEnabled(ui,51,enabled and stopLine,true)
        simUI.setEnabled(ui,52,enabled and stopLine,true)
        simUI.setEnabled(ui,18,enabled and overridePallet,true)

    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,22,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setCheckboxValue(ui,50,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)
        simUI.setEditValue(ui,51,simBWF.format("%.0f",config['stopLinePos']/0.001),true)
        simUI.setEditValue(ui,52,simBWF.format("%.0f",config['stopLineProcessingStage']),true)
        simUI.setEditValue(ui,24,simBWF.format("%.0f",config['transferStart']/0.001),true)
        simUI.setEditValue(ui,23,simBWF.format("%.0f",config['transferLength']/0.001),true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)
        simUI.setCheckboxValue(ui,19,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        updateEnabledDisabledItemsDlg()
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function setPalletDlgItemContent()
    if palletUi then
        local config=readInfo(h)
        local sel=simBWF.getSelectedEditWidget(palletUi)
        local pattern=config['palletPattern']
        simUI.setRadiobuttonValue(palletUi,101,simBWF.getRadiobuttonValFromBool(pattern==0),true)
        simUI.setRadiobuttonValue(palletUi,103,simBWF.getRadiobuttonValFromBool(pattern==2),true)
        simUI.setRadiobuttonValue(palletUi,104,simBWF.getRadiobuttonValFromBool(pattern==3),true)
        simUI.setRadiobuttonValue(palletUi,105,simBWF.getRadiobuttonValFromBool(pattern==4),true)
        simUI.setRadiobuttonValue(palletUi,106,simBWF.getRadiobuttonValFromBool(pattern==5),true)
        local circular=config['circularPatternData3']
        local off=circular[1]
        simUI.setEditValue(palletUi,3004,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,3000,simBWF.format("%.0f",circular[2]/0.001),true) -- radius
        simUI.setEditValue(palletUi,3001,simBWF.format("%.0f",circular[3]),true) -- count
        simUI.setEditValue(palletUi,3002,simBWF.format("%.0f",180*circular[4]/math.pi),true) -- angle off
        simUI.setCheckboxValue(palletUi,3003,simBWF.getCheckboxValFromBool(circular[5]),true) --center
        simUI.setEditValue(palletUi,3005,simBWF.format("%.0f",circular[6]),true) -- layers
        simUI.setEditValue(palletUi,3006,simBWF.format("%.0f",circular[7]/0.001),true) -- layer step

        local lin=config['linePatternData']
        off=lin[1]
        simUI.setEditValue(palletUi,4000,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,4001,simBWF.format("%.0f",lin[2]),true) -- rows
        simUI.setEditValue(palletUi,4002,simBWF.format("%.0f",lin[3]/0.001),true) -- row step
        simUI.setEditValue(palletUi,4003,simBWF.format("%.0f",lin[4]),true) -- cols
        simUI.setEditValue(palletUi,4004,simBWF.format("%.0f",lin[5]/0.001),true) -- col step
        simUI.setEditValue(palletUi,4005,simBWF.format("%.0f",lin[6]),true) -- layers
        simUI.setEditValue(palletUi,4006,simBWF.format("%.0f",lin[7]/0.001),true) -- layer step

        local honey=config['honeycombPatternData']
        off=honey[1]
        simUI.setEditValue(palletUi,5000,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,5001,simBWF.format("%.0f",honey[2]),true) -- rows
        simUI.setEditValue(palletUi,5002,simBWF.format("%.0f",honey[3]/0.001),true) -- row step
        simUI.setEditValue(palletUi,5003,simBWF.format("%.0f",honey[4]),true) -- cols
        simUI.setEditValue(palletUi,5004,simBWF.format("%.0f",honey[5]/0.001),true) -- col step
        simUI.setEditValue(palletUi,5005,simBWF.format("%.0f",honey[6]),true) -- layers
        simUI.setEditValue(palletUi,5006,simBWF.format("%.0f",honey[7]/0.001),true) -- layer step
        simUI.setCheckboxValue(palletUi,5007,simBWF.getCheckboxValFromBool(honey[8]),true) -- firstRowOdd

        simUI.setEnabled(palletUi,201,(pattern==0),true)
        simUI.setEnabled(palletUi,203,(pattern==2),true)
        simUI.setEnabled(palletUi,204,(pattern==3),true)
        simUI.setEnabled(palletUi,205,(pattern==4),true)
        simUI.setEnabled(palletUi,206,(pattern==5),true)
        simBWF.setSelectedEditWidget(palletUi,sel)
    end
end


function hidden_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function console_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function showPoints_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function palletOverride_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.1 then v=0.1 end
        if v>2 then v=2 end
        if v~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function lengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>1 then v=1 end
        if v~=c['length'] then
            local sl=c['stopLinePos']
            if sl<-v*0.45 then sl=-v*0.45 end
            if sl>v*0.45 then sl=v*0.45 end
            simBWF.markUndoPoint()
            c['length']=v
            c['stopLinePos']=sl
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function stopLine_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setSizes()
    setDlgItemContent()
end


function stopLineChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        local l=c['length']
        v=v*0.001
        if v<-l*0.45 then v=-l*0.45 end
        if v>l*0.45 then v=l*0.45 end
        if v~=c['stopLinePos'] then
            simBWF.markUndoPoint()
            c['stopLinePos']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function stopLineProcessingStageChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['stopLineProcessingStage'] then
            simBWF.markUndoPoint()
            c['stopLineProcessingStage']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function transferStartChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0 then v=0 end
        if v>1 then v=1 end
        if v~=c['transferStart'] then
            simBWF.markUndoPoint()
            c['transferStart']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function transferLengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>1 then v=1 end
        if v~=c['transferLength'] then
            simBWF.markUndoPoint()
            c['transferLength']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>0.5 then v=0.5 end
        if v~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function circularPattern_offsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['circularPatternData3'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end

function circularPattern_radiusChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.5 then v=0.5 end
        if v~=c['circularPatternData3'][2] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][2]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_angleOffsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<-359 then v=-359 end
        if v>359 then v=359 end
        v=v*math.pi/180
        if v~=c['circularPatternData3'][4] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][4]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_countChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>40 then v=40 end
        if v~=c['circularPatternData3'][3] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][3]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_layersChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['circularPatternData3'][6] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][6]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_layerStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['circularPatternData3'][7] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][7]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_centerChange_callback(ui,id,newVal)
    local c=readInfo()
    c['circularPatternData3'][5]=(newVal~=0)
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end


function linePattern_offsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['linePatternData'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end

function linePattern_rowsChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][2] then
            simBWF.markUndoPoint()
            c['linePatternData'][2]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_rowStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][3] then
            simBWF.markUndoPoint()
            c['linePatternData'][3]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_colsChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][4] then
            simBWF.markUndoPoint()
            c['linePatternData'][4]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_colStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][5] then
            simBWF.markUndoPoint()
            c['linePatternData'][5]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_layersChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][6] then
            simBWF.markUndoPoint()
            c['linePatternData'][6]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_layerStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][7] then
            simBWF.markUndoPoint()
            c['linePatternData'][7]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end




function honeyPattern_offsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['honeycombPatternData'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end

function honeyPattern_rowsChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][2] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][2]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_rowStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][3] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][3]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_colsChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][4] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][4]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_colStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][5] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][5]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_layersChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][6] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][6]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_layerStepChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][7] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][7]=v
            writeInfo(c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_rowIsOddChange_callback(ui,id,newVal)
    local c=readInfo()
    c['honeycombPatternData'][8]=(newVal~=0)
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end









function conveyorChange_callback(ui,id,newIndex)
    local newLoc=comboConveyor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF,newLoc)
    simBWF.markUndoPoint()
end

function inputChange_callback(ui,id,newIndex)
    local newLoc=comboInput[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_INPUT_REF,newLoc)
    simBWF.markUndoPoint()
end

function editPatternCode_callback(ui,id,newVal)
    local prop=readInfo()
    local s="600 100"
    local p="200 200"
    if distributionDlgSize then
        s=distributionDlgSize[1]..' '..distributionDlgSize[2]
    end
    if distributionDlgPos then
        p=distributionDlgPos[1]..' '..distributionDlgPos[2]
    end
    local xml = [[ <editor title="Pallet points" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            
    local initialText=simBWF.palletPointsToString(prop['palletPoints'])

    initialText=initialText.."\n\n--[[".."\n\nFormat as in following example:\n\n"..[[
{{pt1X,pt1Y,pt1Z},{pt1Alpha,pt1Beta,pt1Gamma},pt1Layer},
{{pt2X,pt2Y,pt2Z},{pt2Alpha,pt2Beta,pt2Gamma},pt2Layer}]].."\n\n--]]"

    local modifiedText
    while true do
        modifiedText,distributionDlgSize,distributionDlgPos=sim.openTextEditor(initialText,xml)
        local newPalletPoints=simBWF.stringToPalletPoints(modifiedText)
        if newPalletPoints then
            if not simBWF.arePalletPointsSame_posOrientAndLayer(newPalletPoints,prop['palletPoints']) then
                prop['palletPoints']=newPalletPoints
                simBWF.markUndoPoint()
                writeInfo(prop)
            end
            break
        else
            if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_yesno,'Input Error',"The input is not formated correctly. Do you wish to discard the changes?") then
                break
            end
            initialText=modifiedText
        end
    end
end

function importPallet_callback(ui,id,newVal)
    local file=sim.fileDialog(sim.filedlg_type_load,'Loading pallet items','','','pallet items','txt')
    if file then
        local newPalletPoints=simBWF.readPalletFromFile(file)
        if newPalletPoints then
            local prop=readInfo()
            prop['palletPoints']=newPalletPoints
            simBWF.markUndoPoint()
            writeInfo(prop)
        else
            sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'File Read Error',"The specified file could not be read.")
        end
    end
end

function patternTypeClick_callback(ui,id)
    local c=readInfo()
    local changed=(c['palletPattern']~=id-101)
    c['palletPattern']=id-101
--    if c['palletPattern']==5 and changed then
--        c['palletPoints']={} -- clear the pallet points when we select 'imported'
--    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setPalletDlgItemContent()
end

function palletCreation_callback(ui,id,newVal)
    createPalletDlg()
end

function onPalletCloseClicked()
    if palletUi then
        local x,y=simUI.getPosition(palletUi)
        previousPalletDlgPos={x,y}
        simUI.destroy(palletUi)
        palletUi=nil
        updatePalletPoints()
    end
end

function createPalletDlg()
    if not palletUi then
        local xml =[[
    <tabs id="77">
            <tab title="None">
            <radiobutton text="Do not create a pallet" on-click="patternTypeClick_callback" id="101" />
            <group layout="form" flat="true" id="201">
            </group>
            <label text="" style="* {margin-left: 380px;}"/>
            </tab>

            <tab title="Circular type">
            <radiobutton text="Create a pallet with items arranged in a circular pattern" on-click="patternTypeClick_callback" id="103" />
            <group layout="form" flat="true"  id="203">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="circularPattern_offsetChange_callback" id="3004"/>

                <label text="Items on circumference"/>
                <edit on-editing-finished="circularPattern_countChange_callback" id="3001"/>

                <label text="Angle offset (deg)"/>
                <edit on-editing-finished="circularPattern_angleOffsetChange_callback" id="3002"/>

                <label text="Radius (mm)"/>
                <edit on-editing-finished="circularPattern_radiusChange_callback" id="3000"/>

                <label text="Center in use"/>
                <checkbox text="" on-change="circularPattern_centerChange_callback" id="3003" />

                <label text="Layers"/>
                <edit on-editing-finished="circularPattern_layersChange_callback" id="3005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="circularPattern_layerStepChange_callback" id="3006"/>
            </group>
            </tab>

            <tab title="Line type">
            <radiobutton text="Create a pallet with items arranged in a rectangular pattern" on-click="patternTypeClick_callback" id="104" />
            <group layout="form" flat="true"  id="204">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="linePattern_offsetChange_callback" id="4000"/>

                <label text="Rows"/>
                <edit on-editing-finished="linePattern_rowsChange_callback" id="4001"/>

                <label text="Row step (mm)"/>
                <edit on-editing-finished="linePattern_rowStepChange_callback" id="4002"/>

                <label text="Columns"/>
                <edit on-editing-finished="linePattern_colsChange_callback" id="4003"/>

                <label text="Columns step (mm)"/>
                <edit on-editing-finished="linePattern_colStepChange_callback" id="4004"/>

                <label text="Layers"/>
                <edit on-editing-finished="linePattern_layersChange_callback" id="4005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="linePattern_layerStepChange_callback" id="4006"/>
            </group>
            </tab>

            <tab title="Honeycomb type">
            <radiobutton text="Create a pallet with items arranged in a honeycomb pattern" on-click="patternTypeClick_callback" id="105" />
            <group layout="form" flat="true"  id="205">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="honeyPattern_offsetChange_callback" id="5000"/>

                <label text="Rows (longest)"/>
                <edit on-editing-finished="honeyPattern_rowsChange_callback" id="5001"/>

                <label text="Row step (mm)"/>
                <edit on-editing-finished="honeyPattern_rowStepChange_callback" id="5002"/>

                <label text="Columns"/>
                <edit on-editing-finished="honeyPattern_colsChange_callback" id="5003"/>

                <label text="Columns step (mm)"/>
                <edit on-editing-finished="honeyPattern_colStepChange_callback" id="5004"/>

                <label text="Layers"/>
                <edit on-editing-finished="honeyPattern_layersChange_callback" id="5005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="honeyPattern_layerStepChange_callback" id="5006"/>

                <label text="1st row is odd"/>
                <checkbox text="" on-change="honeyPattern_rowIsOddChange_callback" id="5007" />
            </group>
            </tab>

            <tab title="Custom/imported">
            <radiobutton text="Create a pallet with items arranged in a customized pattern" on-click="patternTypeClick_callback" id="106" />
            <group layout="vbox" flat="true"  id="206">
                <button text="Edit pallet items"  on-click="editPatternCode_callback"  id="6000"/>
                <button text="Import pallet items"  on-click="importPallet_callback"  id="6001"/>
                <label text="" style="* {margin-left: 380px;}"/>
            </group>
            </tab>

            </tabs>
        ]]
        palletUi=simBWF.createCustomUi(xml,"Pallet Creation",'center',true,'onPalletCloseClicked',true--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])


        setPalletDlgItemContent()
        local c=readInfo()
        local pattern=c['palletPattern']
        local pat={}
        pat[0]=0
        pat[2]=1
        pat[3]=2
        pat[4]=3
        pat[5]=4
        simUI.setCurrentTab(palletUi,77,pat[pattern],true)
    end
end


function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="20"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="21"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="22"/>

                <checkbox text="Stop line (mm)" on-change="stopLine_callback" id="50" />
                <edit on-editing-finished="stopLineChange_callback" id="51"/>

                <label text="Stop line processing stage <" style="* {margin-left: 20px;}"/>
                <edit on-editing-finished="stopLineProcessingStageChange_callback" id="52"/>

                <label text="Transfer start (mm)"/>
                <edit on-editing-finished="transferStartChange_callback" id="24"/>

                <label text="Transfer length (mm)"/>
                <edit on-editing-finished="transferLengthChange_callback" id="23"/>

                <label text="Conveyor belt"/>
                <combobox id="10" on-change="conveyorChange_callback">
                </combobox>

                <label text="Input"/>
                <combobox id="11" on-change="inputChange_callback">
                </combobox>

                <checkbox text="Override pallet creation" on-change="palletOverride_callback" id="19" />
                <button text="Adjust"  on-click="palletCreation_callback" id="18" />

                 <label text="Hidden during simulation" />
                <checkbox text="" on-change="hidden_callback" id="3" />

                <label text="Visualize tracked items"/>
                <checkbox text="" on-change="showPoints_callback" id="5" />

                <label text="Show debug console"/>
                <checkbox text="" on-change="console_callback" id="4" />
        ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,false,nil,false,false,false,'layout="form"')

        local c=readInfo()
        local loc=getAvailableConveyors()
        comboConveyor=simBWF.populateCombobox(ui,10,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF)),true,{{simBWF.NONE_TEXT,-1}})
        loc=getAvailableDetectionOrTrackingWindows()
        comboInput=simBWF.populateCombobox(ui,11,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_INPUT_REF)),true,{{simBWF.NONE_TEXT,-1}})

        setDlgItemContent()
    end
end

function showDlg()
    if not ui then
        createDlg()
    end
end

function removeDlg()
    if ui then
        local x,y=simUI.getPosition(ui)
        previousDlgPos={x,y}
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    -- Following for backward compatibility:
    if _info['conveyor'] then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['conveyor']))
        _info['conveyor']=nil
    end
    if _info['input'] then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_INPUT_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['input']))
        _info['input']=nil
    end
    ----------------------------------------
    writeInfo(_info)
    trackBox=sim.getObjectHandle('genericTrackingWindow_track')
    stopLineBox=sim.getObjectHandle('genericTrackingWindow_stopLine')
    transferBox=sim.getObjectHandle('genericTrackingWindow_transfer')
    
    -- Following for backward compatibility:
    createPalletPointsIfNeeded()
    updatePluginRepresentation()
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
end

showOrHideUiIfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg()
    else
        removeDlg()
    end
end

function sysCall_nonSimulation()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    setSizes() -- reset the box's shift bias
    local c=readInfo()
    c['transferItems']={}
    c['itemsToRemoveFromTracking']={}
    c['targetPositionsToMarkAsProcessed']={}
    c['trackedItemsInWindow']={}
    c['trackedTargetsInWindow']={}
    c['associatedRobotTrackingCorrectionTime']=0
    c['freezeStaticWindow']=nil
    writeInfo(c)
    sim.setModelProperty(model,0)
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    removeDlg()
    local c=readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1)==0)
    if not show then
        sim.setModelProperty(model,sim.modelproperty_not_visible)
    end
end

function sysCall_beforeInstanceSwitch()
    removeDlg()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    removeDlg()
    removeFromPluginRepresentation()
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end