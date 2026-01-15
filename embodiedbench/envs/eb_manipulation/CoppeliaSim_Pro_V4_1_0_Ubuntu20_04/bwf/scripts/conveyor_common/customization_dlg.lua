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


function model.dlg.sizeChange(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t={c.length,c.width,c.height}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0.1 then t[i]=0.1 end
            if t[i]>40 then t[i]=40 end
        end
        if i==2 then
            if t[i]<0.01 then t[i]=0.01 end
            if t[i]>5 then t[i]=5 end
        end
        if i==3 then
            if t[i]<0.01 then t[i]=0.01 end
            if t[i]>2 then t[i]=2 end
        end
        i=i+1
    end
    c.length=t[1]
    c.width=t[2]
    c.height=t[3]
    model.writeInfo(c)
    model.updateConveyor()
    simBWF.markUndoPoint()
    model.dlg.refresh()
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

        model.dlg.updateEnabledDisabledItems_specific()
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

                <label text="Connected device"/>
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

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>

            </group>

    </tab>
    <tab title="Dimensions">
            <group layout="form" flat="false">
                <label text="Size (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.sizeChange" id="2"/>
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


    </tab>
    ]]
--[[
            <group layout="form" flat="false">
                <label text="Identification" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Device ID"/>
                <combobox id="4899" on-change="model.dlg.deviceIdComboChange_callback"> </combobox>
            </group>
--]]

    xml=xml..model.dlg.getSpecificTabContent().."</tabs>"

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

        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,2,simBWF.format("%.0f , %.0f , %.0f",config.length*1000,config.width*1000,config.height*1000),true)

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
--        model.dlg.updateDeviceIdCombobox()
        model.dlg.updateOutputboxCombobox()

        model.dlg.refresh_specific()

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
