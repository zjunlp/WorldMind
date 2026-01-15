model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)

        local loc=model.getAvailableConveyors()
        model.dlg.comboConveyor=simBWF.populateCombobox(model.dlg.ui,3,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)),true,{{simBWF.NONE_TEXT,-1}})
        
        local updateFrequComboItems={
            {"every 50 ms",0},
            {"every 200 ms",1},
            {"every 1000 ms",2}
        }
        simBWF.populateCombobox(model.dlg.ui,103,updateFrequComboItems,{},updateFrequComboItems[config.plotUpdateFrequ[1]+1][1],false,nil)
        simBWF.populateCombobox(model.dlg.ui,104,updateFrequComboItems,{},updateFrequComboItems[config.plotUpdateFrequ[2]+1][1],false,nil)
--        model.dlg.updateDeviceIdCombobox()

        local loc=model.getAvailableInputs()
        model.dlg.comboInput=simBWF.populateCombobox(model.dlg.ui,232,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)),true,{{simBWF.NONE_TEXT,-1}})
        
        local d=config['calibrationBallDistance']
        simUI.setEditValue(model.dlg.ui,233,simBWF.format("%.0f",d/0.001),true)
        
        simUI.setCheckboxValue(model.dlg.ui,24,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.bitCoded,2)>0))
        
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,6,simBWF.format("%.0f",config.detectionWidth/0.001),true)
--        simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f",config.measurementLength/0.001),true)
        for i=1,2,1 do
            local off=config.detectionOffset[i]
            simUI.setEditValue(model.dlg.ui,4+i-1,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
            simUI.setCheckboxValue(model.dlg.ui,100+i,simBWF.getCheckboxValFromBool(config.showPlot[i]),true)
        end
        
        local pallets=simBWF.getAvailablePallets()
        local refPallet=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
        local selected=simBWF.NONE_TEXT
        for i=1,#pallets,1 do
            if pallets[i][2]==refPallet then
                selected=pallets[i][1]
                break
            end
        end
        comboPallet=simBWF.populateCombobox(model.dlg.ui,234,pallets,{},selected,true,{{simBWF.NONE_TEXT,-1}})
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
        model.dlg.updateEnabledDisabledItems()
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
--        simUI.setEnabled(model.dlg.ui,1,simStopped,true)
        simUI.setEnabled(model.dlg.ui,6,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3,simStopped,true)
        simUI.setEnabled(model.dlg.ui,24,simStopped and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)>=0,true)
--        simUI.setEnabled(model.dlg.ui,4899,simStopped,true)
        simUI.setEnabled(model.dlg.ui,232,simStopped,true)
        simUI.setEnabled(model.dlg.ui,233,simStopped and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)>=0,true)
        local notOnline=not simBWF.isSystemOnline()
    end
end

function model.dlg.conveyorChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboConveyor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,-1)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR,newLoc)
    sim.setObjectParent(model.handle,newLoc,true) -- attach/detach the vision system to/from the conveyor
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.detectionOffsetChange_callback(ui,id,newValue)
    local index=id-4+1
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
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.showSensorPlotClick_callback(ui,id,newVal)
    local index=id-101+1
    local c=model.readInfo()
    c.showPlot[index]=(newVal~=0)
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.plotUpdateFrequChange_callback(ui,id,newIndex)
    local index=id-103+1
    local c=model.readInfo()
    c.plotUpdateFrequ[index]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.flipped180Click_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    model.writeInfo(c)
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    simBWF.markUndoPoint()
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

function model.dlg.detectionWidthChange_callback(ui,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0.005 then newValue=0.005 end
        if newValue>0.2 then newValue=0.2 end
        if c.detectionWidth~=newValue then
            c.detectionWidth=newValue
            model.writeInfo(c)
            model.adjustSensor()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

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
        if string.find(data.deviceIds[i],"SENSOR-")==1 then
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

function model.dlg.inputChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboInput[newIndex+1][2]
    if newLoc>=0 then
        simBWF.forbidInputForTrackingWindowChainItems(newLoc)
    end
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,newLoc)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR,-1) -- no conveyor in that case
    sim.setObjectParent(model.handle,-1,true) -- detach the vision system from the conveyor
    model.avoidCircularInput(-1)
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.calibrationBallDistanceChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>5 then v=5 end
        if v~=c['calibrationBallDistance'] then
            c['calibrationBallDistance']=v
            model.writeInfo(c)
            model.alignCalibrationBallsWithInputAndReturnRedBall()
            model.updatePluginRepresentation()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.palletChange_callback(ui,id,newIndex)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PALLET,comboPallet[newIndex+1][2])
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General">
            <group layout="form" flat="false">
            
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
                <label text="Conveyor belt"/>
                <combobox id="3" on-change="model.dlg.conveyorChange_callback"/>
                
                <label text="Flipped 180 deg. w.r. to conveyor" style="* {margin-left: 20px;}"/>
                <checkbox text="" checked="false" on-change="model.dlg.flipped180Click_callback" id="24"/>
                
                <label text="Input"/>
                <combobox id="232" on-change="model.dlg.inputChange_callback">
                </combobox>

                <label text="Calibration ball distance (mm)" style="* {margin-left: 20px;}"/>
                <edit on-editing-finished="model.dlg.calibrationBallDistanceChange_callback" id="233"/>

                <label text="Associated pallet" style="* {background-color: #ccffcc}"/>
                <combobox id="234" on-change="model.dlg.palletChange_callback"/>
            </group>
            </tab>
            
            <tab title="Simulation">
            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Show sensor plot"/>
                <checkbox text="" checked="false" on-change="model.dlg.showSensorPlotClick_callback" id="101"/>
                
                <label text="Visualization update freq."/>
                <combobox id="103" on-change="model.dlg.plotUpdateFrequChange_callback"></combobox>
            </group>
            <group layout="form" flat="false">
                <label text="Simulated sensor specific" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Detection offset (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectionOffsetChange_callback" id="4"/>
                
                <label text="Detection width (in mm)"/>
                <edit on-editing-finished="model.dlg.detectionWidthChange_callback" id="6"/>
            </group>
            </tab>

            <tab title="Online">
            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Show sensor plot"/>
                <checkbox text="" checked="false" on-change="model.dlg.showSensorPlotClick_callback" id="102"/>
                
                <label text="Visualization update freq."/>
                <combobox id="104" on-change="model.dlg.plotUpdateFrequChange_callback"></combobox>
            </group>
            <group layout="form" flat="false">
                <label text="Real sensor specific" style="* {font-weight: bold;}"/>  <label text=""/>
                
                <label text="Detection offset (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectionOffsetChange_callback" id="5"/>
                
            </group>
            
            </tab>

       </tabs>
        ]]
        
--[[        
                <label text="Device ID"/>
                <combobox id="4899" on-change="model.dlg.deviceIdComboChange_callback"> </combobox>
--]]        
        
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
        
        model.dlg.refresh()
        simUI.setCurrentTab(model.dlg.ui,77,model.dlg.mainTabIndex,true)
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
