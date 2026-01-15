function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='thermoformer'
    end
    if not info['velocity'] then
        info['velocity']=0.1
    end
    if not info['acceleration'] then
        info['acceleration']=0.1
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- 64: enabled
    end
    if not info['encoderDistance'] then
        info['encoderDistance']=0
    end
    if not info['stopRequests'] then
        info['stopRequests']={}
    end
    if not info['extrusionWidth'] then
        info['extrusionWidth']=0.1
    end
    if not info['extrusionLength'] then
        info['extrusionLength']=0.2
    end
    if not info['extrusionDepth'] then
        info['extrusionDepth']=0.05
    end
    if not info['plasticThickness'] then
        info['plasticThickness']=0.002
    end
    if not info['rows'] then
        info['rows']=3
    end
    if not info['rowStep'] then
        info['rowStep']=0.15
    end
    if not info['columns'] then
        info['columns']=2
    end
    if not info['columnStep'] then
        info['columnStep']=0.25
    end
    if not info['color'] then
        info['color']={1,1,0.9}
    end
    if not info['pullLength'] then
        info['pullLength']=0.5
    end
    if not info['cuttingStationIndex'] then
        info['cuttingStationIndex']=3
    end
    if not info['partName'] then
        info['partName']='<partName>'
    end
    if not info['partDestination'] then
        info['partDestination']='<defaultDestination>'
    end
    if not info['placeOffset'] then
        info['placeOffset']={0,0,0}
    end
    if not info['indexesPerMinute'] then
        info['indexesPerMinute']=5
    end
    if not info['deactivationTime'] then
        info['deactivationTime']=60
    end

    if info['palletSpacing'] then
        info['pullLength']=info['palletSpacing']+info['columns']*info['columnStep']
        info['palletSpacing']=nil
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,'')
    end
end

function readPartInfo(handle)
    local data=simBWF.readPartInfoV0(handle)

    -- Additional fields here:
    if not data['thermoformingPart'] then
        data['thermoformingPart']=true
    end

    return data
end

function writePartInfo(handle,data)
    return simBWF.writePartInfo(handle,data)
end

function computeDwellTime(velOrNil,accelOrNil,spacingOrNil,indPerMinOrNil)
    local c=readInfo()
    local v=c['velocity']
    local a=c['acceleration']
    local s=c['pullLength']
    local im=c['indexesPerMinute']
    if velOrNil then
        v=velOrNil
    end
    if accelOrNil then
        a=accelOrNil
    end
    if spacingOrNil then
        s=spacingOrNil
    end
    if indPerMinOrNil then
        im=indPerMinOrNil
    end
    local tt=2*v/a+(s-v*v/a)/v
    local dt=60/im-tt
print("dwellT, travelT",dt,tt)
    return dt
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

function getAvailableSensors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_BINARYSENSOR_INFO')
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function triggerStopChange_callback(ui,id,newIndex)
    local sens=comboStopTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,1,sens)
    if simBWF.getReferencedObjectHandle(model,2)==sens then
        simBWF.setReferencedObjectHandle(model,2,-1)
    end
    simBWF.markUndoPoint()
    updateStartStopTriggerComboboxes()
end

function triggerStartChange_callback(ui,id,newIndex)
    local sens=comboStartTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,2,sens)
    if simBWF.getReferencedObjectHandle(model,1)==sens then
        simBWF.setReferencedObjectHandle(model,1,-1)
    end
    simBWF.markUndoPoint()
    updateStartStopTriggerComboboxes()
end

function updateStartStopTriggerComboboxes()
    local c=readInfo()
    local loc=getAvailableSensors()
    comboStopTrigger=simBWF.populateCombobox(ui,13,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,1)),true,{{simBWF.NONE_TEXT,-1}})
    comboStartTrigger=simBWF.populateCombobox(ui,14,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,2)),true,{{simBWF.NONE_TEXT,-1}})
end


showSamplesAndPallets=function(show)
    local lay=0
    if show then
        lay=1
    end
    local samples=sim.getObjectsInTree(sampleHolder,sim.handle_all,1)
    for i=1,#samples,1 do
        sim.setObjectInt32Parameter(samples[i],sim.objintparam_visibility_layer,lay)
    end
    local pallets=sim.getObjectsInTree(palletHolder,sim.handle_all,1)
    for i=1,#pallets,1 do
        sim.setObjectInt32Parameter(pallets[i],sim.objintparam_visibility_layer,lay)
    end
end

function createPalletPoint(objectHandle)
    local data=readPartInfo(objectHandle)
    if #data['palletPoints']==0 then
        data['palletPoints']=simBWF.generatePalletPoints(data)
    end
    writePartInfo(objectHandle,data)
end

function updateThermoformer()
    local conf=readInfo()
    local extrusionWidth=conf['extrusionWidth']
    local extrusionLength=conf['extrusionLength']
    local extrusionDepth=conf['extrusionDepth']
    local wt=conf['plasticThickness']
    local rows=conf['rows']
    local rowStep=conf['rowStep']
    local columns=conf['columns']
    local columnStep=conf['columnStep']
    local palletSpacing=conf['pullLength']-columns*columnStep
    local cuttingStationIndex=conf['cuttingStationIndex']

    local samples=sim.getObjectsInTree(sampleHolder,sim.handle_all,1)
    for i=1,#samples,1 do
        sim.removeObject(samples[i])
    end
    --local bitCoded=conf['bitCoded']
    local objRelPos={-(rows-1)*0.5*rowStep,(columns-1)*0.5*columnStep,-0.5*(extrusionDepth+wt)}
    for col=1,columns,1 do
        objRelPos[1]=-(rows-1)*0.5*rowStep
        for row=1,rows,1 do
            local boxModel=simBWF.createOpenBox({extrusionWidth+wt*2,extrusionLength+wt*2,extrusionDepth+wt},wt,wt,200,1,true,false,conf['color'])
            sim.setObjectSpecialProperty(boxModel,0)
            local p=sim.getObjectProperty(boxModel)
            p=sim.boolOr32(p,sim.objectproperty_selectmodelbaseinstead)
            p=sim.boolOr32(p,sim.objectproperty_dontshowasinsidemodel)
            sim.setObjectProperty(boxModel,p)
            sim.setObjectPosition(boxModel,sampleHolder,objRelPos)
            sim.setObjectParent(boxModel,sampleHolder,true)
            objRelPos[1]=objRelPos[1]+rowStep
            local info=readPartInfo(boxModel)
            info['name']=conf['partName']
            info['destination']=conf['partDestination']
            info['singlePatternData']=conf['placeOffset']
            info['palletPattern']=1
            writePartInfo(boxModel,info)
            createPalletPoint(boxModel)
        end
        objRelPos[2]=objRelPos[2]-columnStep
    end
    local len=columns*columnStep*cuttingStationIndex+palletSpacing*(cuttingStationIndex-1)

    sim.setObjectPosition(sampleHolder,model,{0,-len*0.5+columns*columnStep*0.5,0})

    setObjectSize(baseShape,rows*rowStep,len,extrusionDepth+wt)
    sim.setObjectPosition(baseShape,model,{0,0,-0.5*(extrusionDepth+wt)})

    local pallets=sim.getObjectsInTree(palletHolder,sim.handle_all,1)
    for i=2,#pallets,1 do
        sim.removeObject(pallets[i])
    end
    pallets={pallets[1]}
    local pos={0,0,-0.5*(extrusionDepth+wt)}
    for i=1,cuttingStationIndex,1 do
        local h=pallets[1]
        if i~=1 then
            h=sim.copyPasteObjects({h},0)[1]
            sim.setObjectParent(h,palletHolder,true)
            pallets[i]=h
        else
            setObjectSize(h,rows*rowStep*0.99,columns*columnStep*0.99,extrusionDepth+wt)
        end
        
        sim.setObjectPosition(h,sampleHolder,pos)
        pos[2]=pos[2]+columns*columnStep+palletSpacing
    end
    local ww=math.min(extrusionWidth,extrusionLength)
    setObjectSize(sensor,ww,ww,extrusionDepth*2)
end


function velocityChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.3 then l=0.3 end
        if l~=c['velocity'] then
            local dwellTime=computeDwellTime(l,nil,nil,nil)
            if dwellTime>=0 then
                simBWF.markUndoPoint()
                c['velocity']=l
                c['dwellTime']=dwellTime
                writeInfo(c)
            else
                sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,"Negative Dwell Time","The value you entered would result in a negative dwell time.")
            end
        end
    end
    setDlgItemContent()
end

function accelerationChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.01 then l=0.01 end
        if l>10 then l=10 end
        if l~=c['acceleration'] then
            local dwellTime=computeDwellTime(nil,l,nil,nil)
            if dwellTime>=0 then
                simBWF.markUndoPoint()
                c['acceleration']=l
                c['dwellTime']=dwellTime
                writeInfo(c)
            else
                sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,"Negative Dwell Time","The value you entered would result in a negative dwell time.")
            end
        end
    end
    setDlgItemContent()
end

function indexPerMinuteChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        if l<0.01 then l=0.01 end
        if l>200 then l=200 end
        if l~=c['indexesPerMinute'] then
            local dwellTime=computeDwellTime(nil,nil,nil,l)
            if dwellTime>=0 then
                simBWF.markUndoPoint()
                c['indexesPerMinute']=l
                c['dwellTime']=dwellTime
                writeInfo(c)
            else
                sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,"Negative Dwell Time","The value you entered would result in a negative dwell time.")
            end
        end
    end
    setDlgItemContent()
end

function pullLengthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        local columns=c['columns']
        local columnStep=c['columnStep']
        if l<columns*columnStep then l=columns*columnStep end
        if l>5 then l=5 end
        if l~=c['pullLength'] then
            local dwellTime=computeDwellTime(nil,nil,l,nil)
            if dwellTime>=0 then
                simBWF.markUndoPoint()
                c['pullLength']=l
                c['dwellTime']=dwellTime
                writeInfo(c)
                updateThermoformer()
            else
                sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,"Negative Dwell Time","The value you entered would result in a negative dwell time.")
            end
        end
    end
    setDlgItemContent()
end

function deactivationTime_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        if l<1 then l=1 end
        if l>100000 then l=100000 end
        if l~=c['deactivationTime'] then
            simBWF.markUndoPoint()
            c['deactivationTime']=l
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function cuttingStationIndexChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<3 then l=3 end
        if l>10 then l=10 end
        if l~=c['cuttingStationIndex'] then
            simBWF.markUndoPoint()
            c['cuttingStationIndex']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function extrWidthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.02 then l=0.02 end
        if l>0.5 then l=0.5 end
        local wt=c['plasticThickness']
        local rowStep=c['rowStep']
        if rowStep<l+3*wt then rowStep=l+3*wt end
        if l~=c['extrusionWidth'] or rowStep~=c['rowStep'] then
            simBWF.markUndoPoint()
            c['extrusionWidth']=l
            c['rowStep']=rowStep
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function extrLengthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.02 then l=0.02 end
        if l>0.5 then l=0.5 end
        local wt=c['plasticThickness']
        local columnStep=c['columnStep']
        if columnStep<l+3*wt then columnStep=l+3*wt end
        if l~=c['extrusionLength'] or columnStep~=c['columnStep'] then
            simBWF.markUndoPoint()
            c['extrusionLength']=l
            c['columnStep']=columnStep
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function extrDepthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.01 then l=0.01 end
        if l>0.3 then l=0.3 end
        if l~=c['extrusionDepth'] then
            simBWF.markUndoPoint()
            c['extrusionDepth']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function plasticThicknessChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.01 then l=0.01 end
        if l~=c['plasticThickness'] then
            simBWF.markUndoPoint()
            c['plasticThickness']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function rowChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<1 then l=1 end
        if l>10 then l=10 end
        if l~=c['rows'] then
            simBWF.markUndoPoint()
            c['rows']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function rowStepChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.02 then l=0.02 end
        if l>0.5 then l=0.5 end
        local wt=c['plasticThickness']
        local extrW=c['extrusionWidth']
        if l<extrW+3*wt then l=extrW+3*wt end
        if l~=c['rowStep'] then
            simBWF.markUndoPoint()
            c['rowStep']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function columnChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        local columnStep=c['columnStep']
        local pullLength=c['pullLength']
        if l<1 then l=1 end
        if l>pullLength/columnStep then l=pullLength/columnStep end
        if l~=c['columns'] then
            simBWF.markUndoPoint()
            c['columns']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function columnStepChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        local columns=c['columns']
        local pullLength=c['pullLength']
        if l<0.02 then l=0.02 end
        if l>pullLength/columns then l=pullLength/columns end
        local wt=c['plasticThickness']
        local extrL=c['extrusionLength']
        if l<extrL+3*wt then l=extrL+3*wt end
        if l~=c['columnStep'] then
            simBWF.markUndoPoint()
            c['columnStep']=l
            writeInfo(c)
            updateThermoformer()
        end
    end
    setDlgItemContent()
end

function redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['color'][1]=newVal/100
    writeInfo(c)
    updateThermoformer()
end

function greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['color'][2]=newVal/100
    writeInfo(c)
    updateThermoformer()
end

function blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['color'][3]=newVal/100
    writeInfo(c)
    updateThermoformer()
end

function enabledClicked(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function partName_callback(ui,id,newVal)
    local c=readInfo()
    c['partName']=newVal
    simBWF.markUndoPoint()
    writeInfo(c)
    updateThermoformer()
    setDlgItemContent()
end

function defaultDestination_callback(ui,id,newVal)
    local c=readInfo()
    c['partDestination']=newVal
    simBWF.markUndoPoint()
    writeInfo(c)
    updateThermoformer()
    setDlgItemContent()
end


function placeOffsetChange_callback(ui,id,newVal)
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
    c['placeOffset']={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    updateThermoformer()
    setDlgItemContent()
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,10000,enabled,true)
        simUI.setEnabled(ui,10001,enabled,true)
        simUI.setEnabled(ui,10003,enabled,true)

        simUI.setEnabled(ui,13,enabled,true)
        simUI.setEnabled(ui,14,enabled,true)
        simUI.setEnabled(ui,9,enabled,true)
        simUI.setEnabled(ui,10,enabled,true)
        simUI.setEnabled(ui,19,enabled,true)
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setCheckboxValue(ui,15,(sim.boolAnd32(config['bitCoded'],64)~=0) and 2 or 0,true)
        simUI.setEditValue(ui,11,simBWF.format("%.0f",config['velocity']/0.001),true)
        simUI.setEditValue(ui,12,simBWF.format("%.0f",config['acceleration']/0.001),true)
        simUI.setEditValue(ui,9,simBWF.format("%.0f",config['pullLength']/0.001),true)
        simUI.setEditValue(ui,19,simBWF.format("%.2f",config['indexesPerMinute']),true)
        simUI.setLabelText(ui,21,simBWF.format("%.1f",config['dwellTime']),true) -- actually computed
        simUI.setEditValue(ui,10,simBWF.format("%.0f",config['cuttingStationIndex']),true)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['extrusionWidth']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['extrusionLength']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['extrusionDepth']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['plasticThickness']/0.001),true)
        simUI.setEditValue(ui,5,simBWF.format("%.0f",config['rows']),true)
        simUI.setEditValue(ui,6,simBWF.format("%.0f",config['rowStep']/0.001),true)
        simUI.setEditValue(ui,7,simBWF.format("%.0f",config['columns']),true)
        simUI.setEditValue(ui,8,simBWF.format("%.0f",config['columnStep']/0.001),true)
        simUI.setEditValue(ui,16,config['partName'],true)
        simUI.setEditValue(ui,17,config['partDestination'],true)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['deactivationTime']),true)
        local off=config['placeOffset']
        simUI.setEditValue(ui,18,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)

        updateStartStopTriggerComboboxes()
        updateEnabledDisabledItemsDlg()
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="77">
    <tab title="General" layout="form" id="10002">
                <label text="Enabled"/>
                <checkbox text="" on-change="enabledClicked" id="15"/>

                <label text="Max. velocity (mm/s)"/>
                <edit on-editing-finished="velocityChange" id="11"/>

                <label text="Acceleration (mm/s^2)"/>
                <edit on-editing-finished="accelerationChange" id="12"/>

                <label text="Stop on trigger"/>
                <combobox id="13" on-change="triggerStopChange_callback">
                </combobox>

                <label text="Restart on trigger"/>
                <combobox id="14" on-change="triggerStartChange_callback">
                </combobox>

                <label text="Indexes / minute"/>
                <edit on-editing-finished="indexPerMinuteChange" id="19"/>

                <label text="Pull length (mm)"/>
                <edit on-editing-finished="pullLengthChange" id="9"/>

                <label text="Cutting station index"/>
                <edit on-editing-finished="cuttingStationIndexChange" id="10"/>

                <label text="Calculated dwell time (s)"/>
                <label text="0" id="21"/>
    </tab>
    <tab title="Layout" layout="form" id="10000">
                <label text="Extrusion width (mm)"/>
                <edit on-editing-finished="extrWidthChange" id="1"/>

                <label text="Extrusion length (mm)"/>
                <edit on-editing-finished="extrLengthChange" id="2"/>

                <label text="Extrusion depth (mm)"/>
                <edit on-editing-finished="extrDepthChange" id="3"/>

                <label text="Plastic thickness (mm)"/>
                <edit on-editing-finished="plasticThicknessChange" id="4"/>

                <label text="Rows"/>
                <edit on-editing-finished="rowChange" id="5"/>

                <label text="Row step (mm)"/>
                <edit on-editing-finished="rowStepChange" id="6"/>

                <label text="Columns"/>
                <edit on-editing-finished="columnChange" id="7"/>

                <label text="Column step (mm)"/>
                <edit on-editing-finished="columnStepChange" id="8"/>
    </tab>
    <tab title="Name and Destination" layout="form" id="10003">

                <label text="Place offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="placeOffsetChange_callback" id="18"/>

                <label text="Location name"/>
                <edit on-editing-finished="partName_callback" id="16"/>

                <label text="Default destination"/>
                <edit on-editing-finished="defaultDestination_callback" id="17"/>

                <label text="Part deactivation time (s)"/>
                <edit on-editing-finished="deactivationTime_callback" id="20"/>
    </tab>

    <tab title="Color" layout="form" id="10001">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange" id="60"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange" id="61"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange" id="62"/>

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
    </tab>
    </tabs>
        ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

        local conf=readInfo()
        simUI.setSliderValue(ui,60,conf['color'][1]*100,true)
        simUI.setSliderValue(ui,61,conf['color'][2]*100,true)
        simUI.setSliderValue(ui,62,conf['color'][3]*100,true)

        setDlgItemContent()
        simUI.setCurrentTab(ui,77,dlgMainTabIndex,true)
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
        dlgMainTabIndex=simUI.getCurrentTab(ui,77)
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    dlgMainTabIndex=0
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    -- Following for backward compatibility:
    if _info['stopTrigger'] then
        simBWF.setReferencedObjectHandle(model,1,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['stopTrigger']))
        _info['stopTrigger']=nil
    end
    if _info['startTrigger'] then
        simBWF.setReferencedObjectHandle(model,2,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['startTrigger']))
        _info['startTrigger']=nil
    end
    ----------------------------------------
    writeInfo(_info)
    baseShape=sim.getObjectHandle('genericThermoformer_base')
    sampleHolder=sim.getObjectHandle('genericThermoformer_sampleHolder')
    palletHolder=sim.getObjectHandle('genericThermoformer_palletHolder')
    sensor=sim.getObjectHandle('genericThermoformer_sensor')
	

    -- For backward compatibility:
    local parts=sim.getObjectsInTree(sampleHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        createPalletPoint(parts[i])
    end
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

function sysCall_sensing()
    if simJustStarted then
        updateEnabledDisabledItemsDlg()
    end
    simJustStarted=nil
    showOrHideUiIfNeeded()
end

function sysCall_suspend()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    updateEnabledDisabledItemsDlg()
--    showOrHideUiIfNeeded()
    showSamplesAndPallets(true)
    local conf=readInfo()
    conf['encoderDistance']=0
    conf['stopRequests']={}
    writeInfo(conf)
end

function sysCall_beforeSimulation()
    simJustStarted=true
    showSamplesAndPallets(false)
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