model.dlg={}
model.dlg.calibrationDlg={}

function model.dlg.calibrationDlg.start_callback()
    local data={}
    data.id=model.handle
    data.robotId,data.channel=model.getConnectedRobotAndChannel()
    local res,retData=simBWF.query('conveyor_getEncoderValue',data)
    if res=='ok' then
        model.dlg.calibrationDlg.encoderStart=retData.value
    else
        if simBWF.isInTestMode() then
            model.dlg.calibrationDlg.encoderStart=100
        end
    end
    if model.dlg.calibrationDlg.encoderStart then
        simUI.setEnabled(model.dlg.calibrationDlg.ui,1,false)
        simUI.setEnabled(model.dlg.calibrationDlg.ui,2,true)
        local outputBoxHandle=simBWF.getReferencedObjectHandle(model.handle, model.objRefIdx.OUTPUTBOX)
        if outputBoxHandle~=-1 then
            if simBWF.callCustomizationScriptFunction('model.ext.getState', outputBoxHandle) then
                simUI.setEnabled(model.dlg.calibrationDlg.ui,7,true)
            else
                simUI.setEnabled(model.dlg.calibrationDlg.ui,6,true)
            end
        end
    end
end

function model.dlg.calibrationDlg.end_callback()
    local data={}
    data.id=model.handle
    data.robotId,data.channel=model.getConnectedRobotAndChannel()
    local res,retData=simBWF.query('conveyor_getEncoderValue',data)
    if res=='ok' then
        model.dlg.calibrationDlg.encoderEnd=retData.value
    else
        if simBWF.isInTestMode() then
            model.dlg.calibrationDlg.encoderEnd=568
        end
    end
    if model.dlg.calibrationDlg.encoderEnd and (model.dlg.calibrationDlg.encoderEnd~=model.dlg.calibrationDlg.encoderStart) then
        simUI.setEnabled(model.dlg.calibrationDlg.ui,2,false)
        simUI.setEnabled(model.dlg.calibrationDlg.ui,3,true)
        if simBWF.getReferencedObjectHandle(model.handle, model.objRefIdx.OUTPUTBOX) ~= -1 then
            simUI.setEnabled(model.dlg.calibrationDlg.ui,6,false)
            simUI.setEnabled(model.dlg.calibrationDlg.ui,7,false)
        end
    end
end

function model.dlg.calibrationDlg.beltStart_callback()
    local data={}
    data.id=model.handle
    data.outputBoxId=simBWF.getReferencedObjectHandle(model.handle, model.objRefIdx.OUTPUTBOX)
    local res=simBWF.query('conveyor_calBeltStart',data)
    if res=='ok' then
        simUI.setEnabled(model.dlg.calibrationDlg.ui,6,false)
        simUI.setEnabled(model.dlg.calibrationDlg.ui,7,true)
    else
        if simBWF.isInTestMode() then
            simUI.setEnabled(model.dlg.calibrationDlg.ui,6,false)
            simUI.setEnabled(model.dlg.calibrationDlg.ui,7,true)
        end
    end
end

function model.dlg.calibrationDlg.beltStop_callback()
    local data={}
    data.id=model.handle
    data.outputBoxId=simBWF.getReferencedObjectHandle(model.handle, model.objRefIdx.OUTPUTBOX)
    local res=simBWF.query('conveyor_calBeltStop',data)
    if res=='ok' then
        simUI.setEnabled(model.dlg.calibrationDlg.ui,6,true)
        simUI.setEnabled(model.dlg.calibrationDlg.ui,7,false)
    else
        if simBWF.isInTestMode() then
            simUI.setEnabled(model.dlg.calibrationDlg.ui,6,true)
            simUI.setEnabled(model.dlg.calibrationDlg.ui,7,false)
        end
    end
end

function model.dlg.calibrationDlg.distance_callback(ui,id,value)
    local v=tonumber(value)
    if v then
        if v<-100000 then v=-100000 end
        if v>100000 then v=100000 end
        model.dlg.calibrationDlg.distance=v
        if v~=0 then
            simUI.setEnabled(model.dlg.calibrationDlg.ui,5,true)
        end
    end
    simUI.setEditValue(model.dlg.calibrationDlg.ui,3,simBWF.format("%.5f",model.dlg.calibrationDlg.distance),true)
end

function model.dlg.calibrationDlg.cancel_callback()
    simUI.destroy(model.dlg.calibrationDlg.ui)
end

function model.dlg.calibrationDlg.ok_callback()
    simUI.destroy(model.dlg.calibrationDlg.ui)
    local c=model.readInfo()
    c.calibration=model.dlg.calibrationDlg.distance/(model.dlg.calibrationDlg.encoderEnd-model.dlg.calibrationDlg.encoderStart)
    model.writeInfo(c)
    simUI.setEditValue(model.dlg.ui,30,simBWF.format("%.5f",c.calibration),true)
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
end

function model.dlg.calDlg()
    local xml = [[
        <group layout="hbox" flat="true">
            <button text="Mark start"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.start_callback" id="1" />
            <button text="Mark end"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.end_callback" id="2" />
        </group>
        <group layout="hbox" flat="true">
            <button text="Start belt"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.beltStart_callback" id="6" />
            <button text="Stop belt"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.beltStop_callback" id="7" />
        </group>

        <group layout="form" flat="false">
            <label text="Distance moved (mm)"/>
            <edit on-editing-finished="model.dlg.calibrationDlg.distance_callback" id="3"/>
        </group>

        <group layout="hbox" flat="true">
            <button text="Cancel"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.cancel_callback" id="4" />
            <button text="OK"  style="* {min-width: 150px}" on-click="model.dlg.calibrationDlg.ok_callback" id="5" />
        </group>
    ]]
    model.dlg.calibrationDlg.distance=0 -- in mm
    model.dlg.calibrationDlg.ui=simBWF.createCustomUi(xml,"Conveyor calibration",'center',false,'',true,false,true)
    simUI.setEditValue(model.dlg.calibrationDlg.ui,3,simBWF.format("%.5f",model.dlg.calibrationDlg.distance),true) -- in mm
    simUI.setEnabled(model.dlg.calibrationDlg.ui,2,false)
    simUI.setEnabled(model.dlg.calibrationDlg.ui,3,false)
    simUI.setEnabled(model.dlg.calibrationDlg.ui,5,false)
    simUI.setEnabled(model.dlg.calibrationDlg.ui,6,false)
    simUI.setEnabled(model.dlg.calibrationDlg.ui,7,false)
end

function model.dlg.triggerStopChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboStopTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,sens)
    if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)==sens then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,-1)
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.triggerStartChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboStartTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,sens)
    if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)==sens then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,-1)
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.masterChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboMaster[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR,sens) -- master
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.outputboxChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboOutputbox[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX,sens)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.updateStartStopTriggerComboboxes()
    local loc=model.getAvailableSensors()
    model.dlg.comboStopTrigger=simBWF.populateCombobox(model.dlg.ui,100,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)),true,{{simBWF.NONE_TEXT,-1}})
    model.dlg.comboStartTrigger=simBWF.populateCombobox(model.dlg.ui,101,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)),true,{{simBWF.NONE_TEXT,-1}})
end

function model.dlg.updateMasterCombobox()
    local loc=model.getAvailableMasterConveyors()
    model.dlg.comboMaster=simBWF.populateCombobox(model.dlg.ui,102,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR)),true,{{simBWF.NONE_TEXT,-1}})
end

function model.dlg.updateOutputboxCombobox()
    local loc=model.getAvailableOutputboxes()
    model.dlg.comboOutputbox=simBWF.populateCombobox(model.dlg.ui,103,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX)),true,{{simBWF.NONE_TEXT,-1}})
end

function model.dlg.updatePalletCombobox()
    local loc=simBWF.getAvailablePallets()
    model.dlg.comboPallet=simBWF.populateCombobox(model.dlg.ui,3007,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)),true,{{simBWF.NONE_TEXT,-1}})
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
--        simUI.setEnabled(model.dlg.ui,4899,simStopped,true)
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2,simStopped,true)

        simUI.setEnabled(model.dlg.ui,40,simStopped,true)
        simUI.setEnabled(model.dlg.ui,41,simStopped,true)

        simUI.setEnabled(model.dlg.ui,30,simStopped,true)
        simUI.setEnabled(model.dlg.ui,31,simStopped,true)

        simUI.setEnabled(model.dlg.ui,1000,simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR)==-1,true) -- enable
        simUI.setEnabled(model.dlg.ui,10,simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR)==-1,true) -- vel
        simUI.setEnabled(model.dlg.ui,12,simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR)==-1,true) -- accel

        simUI.setEnabled(model.dlg.ui,100,simStopped,true) -- stop trigger
        simUI.setEnabled(model.dlg.ui,101,simStopped,true) -- restart trigger
        simUI.setEnabled(model.dlg.ui,102,simStopped,true) -- master
        simUI.setEnabled(model.dlg.ui,103,simStopped,true) -- outputBox

        simUI.setEnabled(model.dlg.ui,2000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2002,simStopped,true)
        
        simUI.setEnabled(model.dlg.ui,3000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3003,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3004,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3005,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3006,simStopped,true)
    end
end

function model.dlg.enabledClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.enabled=not conf.enabled
    model.writeInfo(conf)
    simBWF.markUndoPoint()
end

function model.dlg.speedChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<-0.5 then v=-0.5 end
        if v>0.5 then v=0.5 end
        if v~=c['velocity'] then
            simBWF.markUndoPoint()
            c['velocity']=v
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end


function model.dlg.stopCmdChange_callback(ui,id,newVal)
    local c=model.readInfo()
    newVal=string.gsub(newVal,'.',string.upper)
    newVal=string.gsub(newVal,' ','')
    newVal=string.match(newVal,'M8%d%d')
    if newVal then
        if c.stopCmd~=newVal then
            c.stopCmd=newVal
            model.writeInfo(c)
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.startCmdChange_callback(ui,id,newVal)
    local c=model.readInfo()
    newVal=string.gsub(newVal,'.',string.upper)
    newVal=string.gsub(newVal,' ','')
    newVal=string.match(newVal,'M8%d%d')
    if newVal then
        if c.start~=newVal then
            c.startCmd=newVal
            model.writeInfo(c)
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.calibrationChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<-10 then v=-10 end
        if v>10 then v=10 end
        if v~=c['calibration'] then
            c['calibration']=v
            model.writeInfo(c)
            model.updatePluginRepresentation()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.accelerationChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>1 then v=1 end
        if v~=c['acceleration'] then
            simBWF.markUndoPoint()
            c['acceleration']=v
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end

function model.dlg.triggerDistance(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0 then v=0 end
        if v>5 then v=5 end
        if v~=c['triggerDistance'] then
            simBWF.markUndoPoint()
            c['triggerDistance']=v
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end

function model.dlg.dwellTimeChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v>10 then v=10 end
        if v<0 then v=0 end
        if v~=c.thermo_dwellTime then
            c.thermo_dwellTime=v
            model.writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.stationCntChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v>10 then v=10 end
        if v<3 then v=3 end
        if v~=c.thermo_stationCnt then
            c.thermo_stationCnt=v
            model.writeInfo(c)
            model.updateThermoformer()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.stationSpacingChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v>0.2 then v=0.2 end
        if v<0 then v=0 end
        if v~=c.thermo_stationSpacing then
            c.thermo_stationSpacing=v
            model.writeInfo(c)
            model.updateThermoformer()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.extrusionSizeChange(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.thermo_extrusionSize
    local s={c.thermo_rowColStep[1]-2*c.thermo_wallThickness,c.thermo_rowColStep[2]-2*c.thermo_wallThickness,1}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then
            t[i]=100
        end
        t[i]=t[i]*0.001
        if t[i]<0.01 then t[i]=0.01 end
        if t[i]>s[i] then t[i]=s[i] end
        i=i+1
        if i>=4 then break end
    end
    c.thermo_extrusionSize=t
    model.writeInfo(c)
    model.updateThermoformer()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.wallThicknessChange(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    local s=math.min((c.thermo_rowColStep[1]-c.thermo_extrusionSize[1])/2,(c.thermo_rowColStep[2]-c.thermo_extrusionSize[2])/2)
    if v then
        v=v*0.001
        if v>s then v=s end
        if v<0.001 then v=0.001 end
        if v~=c.thermo_wallThickness then
            c.thermo_wallThickness=v
            model.writeInfo(c)
            model.updateThermoformer()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.rowColumnCountChange(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.thermo_rowColCnt
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then
            t[i]=2
        end
        t[i]=math.floor(t[i])
        if t[i]<1 then t[i]=1 end
        if t[i]>10 then t[i]=10 end
        i=i+1
        if i>=3 then break end
    end
    c.thermo_rowColCnt=t
    model.writeInfo(c)
    model.updateThermoformer()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.rowColumnStepChange(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.thermo_rowColStep
    local s={c.thermo_extrusionSize[1]+2*c.thermo_wallThickness,c.thermo_extrusionSize[2]+2*c.thermo_wallThickness}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then
            t[i]=0
        end
        t[i]=t[i]*0.001
        if t[i]<s[i] then t[i]=s[i] end
        if t[i]>0.3 then t[i]=0.3 end
        i=i+1
        if i>=3 then break end
    end
    c.thermo_rowColStep=t
    model.writeInfo(c)
    model.updateThermoformer()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.detectionOffsetChange_callback(ui,id,newValue)
    local index=id-4000+1
    local c=model.readInfo()
    local i=1
    local t={0,0,0}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>2 then t[i]=2 end
        if t[i]<-2 then t[i]=-2 end
        i=i+1
    end
    c.detectionOffset[index]={t[1],t[2],t[3]}
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.redChange(ui,id,newVal)
    local c=model.readInfo()
    c.thermo_color[1]=newVal/100
    model.writeInfo(c)
    model.setColor(c.thermo_color)
    simBWF.markUndoPoint()
end

function model.dlg.greenChange(ui,id,newVal)
    local c=model.readInfo()
    c.thermo_color[2]=newVal/100
    model.writeInfo(c)
    model.setColor(c.thermo_color)
    simBWF.markUndoPoint()
end

function model.dlg.blueChange(ui,id,newVal)
    local c=model.readInfo()
    c.thermo_color[3]=newVal/100
    model.writeInfo(c)
    model.setColor(c.thermo_color)
    simBWF.markUndoPoint()
end

function model.dlg.palletChange_callback(ui,id,newIndex)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PALLET,model.dlg.comboPallet[newIndex+1][2])
    simBWF.markUndoPoint()
    model.dlg.updateEnabledDisabledItems()
end

--[[
function model.dlg.deviceIdComboChange_callback(uiHandle,id,newValue)
    local newDeviceId=model.dlg.comboDeviceIds[newValue+1][1]
    local c=model.readInfo()
    c.deviceId=newDeviceId
    model.writeInfo(c)
    simBWF.markUndoPoint()
--    model.dlg.updateDeviceIdCombobox()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end
--]]

--[[
function model.dlg.updateDeviceIdCombobox()
    local c=model.readInfo()
    local resp,data
    if simBWF.isInTestMode() then
        resp='ok'
        data={}
        data.deviceIds={'RAGNAR-95426587447','RAGNAR-35426884525','CONVEYOR-35426884525-0','CONVEYOR-00:1b:63:84:45:e6-1','SENSOR-00:fa:08:46:8b:11-1','VISION-00:3b:99:34:7d:1f-0'}
    else
        resp,data=simBWF.query('get_deviceIds')
        if resp~='ok' then
            data.deviceIds={}
        end
    end

    local ids={}
    for i=1,#data.deviceIds,1 do
        if string.find(data.deviceIds[i],"CONVEYOR-")==1 then
            ids[#ids+1]=data.deviceIds[i]
        end
    end

    local selected=c.deviceId
    local isKnown=false
    local items={}
    for i=1,#ids,1 do
        if ids[i]==selected then
            isKnown=true
        end
        items[#items+1]={ids[i],i}
    end
    if not isKnown then
        table.insert(items,1,{selected,#items+1})
    end
    if selected~=simBWF.NONE_TEXT then
        table.insert(items,1,{simBWF.NONE_TEXT,#items+1})
    end
    model.dlg.comboDeviceIds=simBWF.populateCombobox(model.dlg.ui,4899,items,{},selected,false,{})
end
--]]

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml = [[
    <tabs id="77">
    <tab title="General">
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

                <label text="Connected robot"/>
                <label text="" id="13"/>

                <label text="Enabled"/>
                <checkbox text="" on-change="model.dlg.enabledClicked" id="1000"/>

                <label text="Speed (mm/s)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.speedChange" id="10"/>

                <label text="Acceleration (mm/s^2)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.accelerationChange" id="12"/>

                <label text="Trigger distance (mm)"/>
                <edit on-editing-finished="model.dlg.triggerDistance" id="14"/>

                <label text="Master conveyor"/>
                <combobox id="102" on-change="model.dlg.masterChange_callback">
                </combobox>

                <label text="Stop on trigger"/>
                <combobox id="100" on-change="model.dlg.triggerStopChange_callback">
                </combobox>

                <label text="Restart on trigger"/>
                <combobox id="101" on-change="model.dlg.triggerStartChange_callback">
                </combobox>

                <label text="Output box"/>
                <combobox id="103" on-change="model.dlg.outputboxChange_callback">
                </combobox>

                <label text="Dwell time (s)"/>
                <edit on-editing-finished="model.dlg.dwellTimeChange" id="3000"/>
                
                <label text="Associated pallet" style="* {background-color: #ccffcc}"/>
                <combobox id="3007" on-change="model.dlg.palletChange_callback"/>
                
            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>

            </group>

    </tab>
    <tab title="Dimensions">
            <group layout="form" flat="false">
                <label text="Station count"/>
                <edit on-editing-finished="model.dlg.stationCntChange" id="3001"/>
                
                <label text="Spacing between station (mm)"/>
                <edit on-editing-finished="model.dlg.stationSpacingChange" id="3002"/>
                
                <label text="Extrusion size (x, y, z, in mm)"/>
                <edit on-editing-finished="model.dlg.extrusionSizeChange" id="3003"/>
                
                <label text="Wall thickness (mm)"/>
                <edit on-editing-finished="model.dlg.wallThicknessChange" id="3004"/>
                
                <label text="Row/column count"/>
                <edit on-editing-finished="model.dlg.rowColumnCountChange" id="3005"/>
                
                <label text="Row/column step (mm)"/>
                <edit on-editing-finished="model.dlg.rowColumnStepChange" id="3006"/>
                
                <label text="Overall size (X, Y, Z, in mm)"/>
                <label text="" id="2"/>
            </group>
    </tab>
    
    <tab title="Simulation">
            <group layout="form" flat="false">
                <label text="Simulated thermoformer specific" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Pallet offset (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectionOffsetChange_callback" id="4000"/>
            </group>
    </tab>
    
    <tab title="Online">
            <group layout="form" flat="false">
                <label text="Stop command"/>
                <edit on-editing-finished="model.dlg.stopCmdChange_callback" id="40"/>

                <label text="Start command"/>
                <edit on-editing-finished="model.dlg.startCmdChange_callback" id="41"/>
            </group>

            <group flat="false">
            <group layout="form" flat="true">
                <label text="Calibration" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Conveyor calibration (mm/pulse)"/>
                <edit on-editing-finished="model.dlg.calibrationChange" id="30"/>

            </group>
            <button text="Calibrate" on-click="model.dlg.calDlg" id="31" />
            <label text=""/>
            </group>
            <group layout="form" flat="false">
                <label text="Real thermoformer specific" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Pallet offset (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectionOffsetChange_callback" id="4001"/>
                
            </group>


    </tab>
    <tab title="Color">
            <group layout="form" flat="false">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange" id="2000"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange" id="2001"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange" id="2002"/>
            </group>
    </tab>
    ]]

    xml=xml.."</tabs>"

    model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    model.dlg.refresh()
    simUI.setCurrentTab(model.dlg.ui,77,model.dlg.mainTabIndex,true)

--]]
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        local config=model.readInfo()
        local size=model.getModelSize()

        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setLabelText(model.dlg.ui,2,simBWF.format("%.0f , %.0f , %.0f",size[1]*1000,size[2]*1000,size[3]*1000),true)

        simUI.setEditValue(model.dlg.ui,40,config.stopCmd,true)
        simUI.setEditValue(model.dlg.ui,41,config.startCmd,true)

        simUI.setCheckboxValue(model.dlg.ui,1000,config.enabled and 2 or 0,true)
        simUI.setEditValue(model.dlg.ui,10,simBWF.format("%.0f",config['velocity']/0.001),true)
        simUI.setEditValue(model.dlg.ui,12,simBWF.format("%.0f",config['acceleration']/0.001),true)
        simUI.setEditValue(model.dlg.ui,14,simBWF.format("%.0f",config['triggerDistance']/0.001),true)
        local connectedRobot=simBWF.NONE_TEXT
        local rob,channel=model.getConnectedRobotAndChannel()
        if rob>=0 then
            connectedRobot=simBWF.getObjectAltName(rob)..' (on channel '..channel..')'
        end
        simUI.setLabelText(model.dlg.ui,13,connectedRobot)

        simUI.setEditValue(model.dlg.ui,30,simBWF.format("%.5f",config['calibration']),true)

        model.dlg.updateStartStopTriggerComboboxes()
        model.dlg.updateMasterCombobox()
        model.dlg.updateOutputboxCombobox()

        simUI.setSliderValue(model.dlg.ui,2000,config.thermo_color[1]*100,true)
        simUI.setSliderValue(model.dlg.ui,2001,config.thermo_color[2]*100,true)
        simUI.setSliderValue(model.dlg.ui,2002,config.thermo_color[3]*100,true)
        
        simUI.setEditValue(model.dlg.ui,3000,simBWF.format("%.2f",config.thermo_dwellTime))
        simUI.setEditValue(model.dlg.ui,3001,simBWF.format("%i",config.thermo_stationCnt))
        simUI.setEditValue(model.dlg.ui,3002,simBWF.format("%.0f",config.thermo_stationSpacing*1000))
        simUI.setEditValue(model.dlg.ui,3003,simBWF.format("%.0f , %.0f , %.0f",config.thermo_extrusionSize[1]*1000,config.thermo_extrusionSize[2]*1000,config.thermo_extrusionSize[3]*1000))
        simUI.setEditValue(model.dlg.ui,3004,simBWF.format("%.0f",config.thermo_wallThickness*1000))
        simUI.setEditValue(model.dlg.ui,3005,simBWF.format("%i , %i",config.thermo_rowColCnt[1],config.thermo_rowColCnt[2]))
        simUI.setEditValue(model.dlg.ui,3006,simBWF.format("%.0f , %.0f",config.thermo_rowColStep[1]*1000,config.thermo_rowColStep[2]*1000))
        
        model.dlg.updatePalletCombobox()

        for i=1,2,1 do
            local off=config.detectionOffset[i]
            simUI.setEditValue(model.dlg.ui,4000+i-1,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
        end

        
        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end


function model.dlg.showDlg()
    if not model.dlg.ui then
        model.dlg.createDlg()
    end
end

function model.dlg.removeDlg()
    if model.dlg.ui then
        local x,y=simUI.getPosition(model.dlg.ui)
        model.dlg.previousDlgPos={x,y}
        model.dlg.mainTabIndex=simUI.getCurrentTab(model.dlg.ui,77)
        simUI.destroy(model.dlg.ui)
        model.dlg.ui=nil
    end
end

function model.dlg.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model.handle then
        model.dlg.showDlg()
    else
        model.dlg.removeDlg()
    end
end

function model.dlg.init()
    model.dlg.mainTabIndex=0
    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
end

