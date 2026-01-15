model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        
        local loc=model.getAvailableConveyors()
        model.dlg.comboConveyor=simBWF.populateCombobox(model.dlg.ui,11,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)),true,{{simBWF.NONE_TEXT,-1}})

        local d=config['calibrationBallDistance']
        simUI.setEditValue(model.dlg.ui,233,simBWF.format("%.0f",d/0.001),true)
        
        local loc=model.getAvailableInputs()
        model.dlg.comboInput=simBWF.populateCombobox(model.dlg.ui,232,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)),true,{{simBWF.NONE_TEXT,-1}})

        simUI.setCheckboxValue(model.dlg.ui,28,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.bitCoded,4)>0))
        simUI.setCheckboxValue(model.dlg.ui,24,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.bitCoded,2)>0))
        simUI.setCheckboxValue(model.dlg.ui,23,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.bitCoded,1)>0))
        simUI.setEditValue(model.dlg.ui,25,simBWF.format("%.0f",config.detectorDiameter/0.001),true)
        simUI.setEditValue(model.dlg.ui,27,simBWF.format("%.0f",config.detectorHeight/0.001),true)
        simUI.setEditValue(model.dlg.ui,26,simBWF.format("%.0f",config.detectorHeightOffset/0.001),true)

        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        
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
        simUI.setEnabled(model.dlg.ui,11,simStopped,true)
        simUI.setEnabled(model.dlg.ui,24,simStopped and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)>=0,true)
        simUI.setEnabled(model.dlg.ui,23,simStopped,true)
        simUI.setEnabled(model.dlg.ui,25,simStopped,true)
        simUI.setEnabled(model.dlg.ui,27,simStopped,true)
        simUI.setEnabled(model.dlg.ui,26,simStopped,true)
        simUI.setEnabled(model.dlg.ui,28,simStopped,true)
        simUI.setEnabled(model.dlg.ui,232,simStopped,true)
        simUI.setEnabled(model.dlg.ui,233,simStopped and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)>=0,true)

        local notOnline=not simBWF.isSystemOnline()
    end
end


function model.dlg.conveyorChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboConveyor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,-1)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR,newLoc)
    sim.setObjectParent(model.handle,newLoc,true) -- attach/detach the detector to/from the conveyor
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.sensorDiameterChange(ui,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0.001 then newValue=0.001 end
        if newValue>0.2 then newValue=0.2 end
        if c.detectorDiameter~=newValue then
            c.detectorDiameter=newValue
            model.writeInfo(c)
            local s=model.getObjectSize(model.handles.detectorSensor)
            model.setObjectSize(model.handles.detectorSensor,newValue,newValue,s[3])
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.detectorHeightOffsetChange(ui,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<-0.2 then newValue=-0.2 end
        if newValue>0.2 then newValue=0.2 end
        if c.detectorHeightOffset~=newValue then
            c.detectorHeightOffset=newValue
            model.writeInfo(c)
            model.setDetectorBoxSizeAndPos()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.detectorHeightChange(ui,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0.05 then newValue=0.05 end
        if newValue>0.8 then newValue=0.8 end
        if c.detectorHeight~=newValue then
            c.detectorHeight=newValue
            model.writeInfo(c)
            model.setDetectorBoxSizeAndPos()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.hideDetectorBoxClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
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

function model.dlg.showDetectionsClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    model.writeInfo(c)
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
                <combobox id="11" on-change="model.dlg.conveyorChange_callback"/>

                <label text="Flipped 180 deg. w.r. to conveyor" style="* {margin-left: 20px;}"/>
                <checkbox text="" checked="false" on-change="model.dlg.flipped180Click_callback" id="24"/>
                
                <label text="Input"/>
                <combobox id="232" on-change="model.dlg.inputChange_callback">
                </combobox>

                <label text="Calibration ball distance (mm)" style="* {margin-left: 20px;}"/>
                <edit on-editing-finished="model.dlg.calibrationBallDistanceChange_callback" id="233"/>
                
                <label text="Associated pallet"/>
                <combobox id="234" on-change="model.dlg.palletChange_callback"/>
            </group>
            </tab>

            <tab title="More">
            <group layout="form" flat="false">
                <label text="Hide when running"/>
                <checkbox text="" checked="false" on-change="model.dlg.hideDetectorBoxClick_callback" id="23"/>
                
                <label text="Show detections in scene"/>
                <checkbox text="" checked="false" on-change="model.dlg.showDetectionsClick_callback" id="28"/>
                
                <label text="Detection diameter (mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.sensorDiameterChange" id="25"/>
                
                <label text="Detection Z-axis size (mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectorHeightChange" id="27"/>
                
                <label text="Detection Z-axis offset (mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.detectorHeightOffsetChange" id="26"/>
            </group>
            </tab>
       </tabs>
        ]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos)        
       
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
