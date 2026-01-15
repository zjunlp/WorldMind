model.dlg={}
model.dlg.pickPlaceDlg={}

function model.dlg.pickPlaceDlg.pickAndPlaceSettingsClose_cb(dlgPos)
    model.dlg.pickPlaceDlg.previousPickPlaceDlgPos=dlgPos
end

function model.dlg.pickPlaceDlg.pickAndPlaceSettingsApply_cb(pickAndPlaceInfo)
    local c=model.readInfo()
    c.pickAndPlaceInfo.overrideGripperSettings=pickAndPlaceInfo.overrideGripperSettings
    c.pickAndPlaceInfo.speed=pickAndPlaceInfo.speed
    c.pickAndPlaceInfo.accel=pickAndPlaceInfo.accel
    for i=1,2,1 do
        c.pickAndPlaceInfo.dwellTime[i]=pickAndPlaceInfo.dwellTime[i]
        c.pickAndPlaceInfo.approachHeight[i]=pickAndPlaceInfo.approachHeight[i]
        c.pickAndPlaceInfo.useAbsoluteApproachHeight[i]=pickAndPlaceInfo.useAbsoluteApproachHeight[i]
        c.pickAndPlaceInfo.departHeight[i]=pickAndPlaceInfo.departHeight[i]
        c.pickAndPlaceInfo.rounding[i]=pickAndPlaceInfo.rounding[i]
        c.pickAndPlaceInfo.nullingAccuracy[i]=pickAndPlaceInfo.nullingAccuracy[i]
        for j=1,3,1 do
            c.pickAndPlaceInfo.offset[i][j]=pickAndPlaceInfo.offset[i][j]
        end
        --c.pickAndPlaceInfo.freeModeTiming[i]=pickAndPlaceInfo.freeModeTiming[i]
        --c.pickAndPlaceInfo.actionModeTiming[i]=pickAndPlaceInfo.actionModeTiming[i]
        c.pickAndPlaceInfo.relativeToBelt[i]=pickAndPlaceInfo.relativeToBelt[i]
    end
    c.pickAndPlaceInfo.actionTemplates=pickAndPlaceInfo.actionTemplates
    c.pickAndPlaceInfo.pickActions=pickAndPlaceInfo.pickActions
    c.pickAndPlaceInfo.multiPickActions=pickAndPlaceInfo.multiPickActions
    c.pickAndPlaceInfo.placeActions=pickAndPlaceInfo.placeActions
    
    model.writeInfo(c)
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
end

function model.dlg.pickAndPlaceSettings_callback()
    local c=model.readInfo()
    model.pickPlaceDlg.display(c.pickAndPlaceInfo,"'"..simBWF.getObjectAltName(model.handle).."' pick & place settings",true,model.dlg.pickPlaceDlg.pickAndPlaceSettingsApply_cb,model.dlg.pickPlaceDlg.pickAndPlaceSettingsClose_cb,model.dlg.pickPlaceDlg.previousPickPlaceDlgPos)
end

function model.dlg.stackingChange_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=math.floor(newValue)
        if newValue<1 then newValue=1 end
        if newValue>20 then newValue=20 end
        if newValue~=c['stacking'] then
            c['stacking']=newValue
            model.writeInfo(c)
            simBWF.markUndoPoint()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.stackingShiftChange_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0 then newValue=0 end
        if newValue>0.1 then newValue=0.1 end
        if newValue~=c['stackingShift'] then
            c['stackingShift']=newValue
            model.writeInfo(c)
            simBWF.markUndoPoint()
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

function model.dlg.nailComboChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['gripperType'][9]=newIndex-0
    c['subtype']=model.getGripperTypeString(c['gripperType'])
    model.writeInfo(c)
    model.updateAppearance()
    simBWF.markUndoPoint()
end

function model.dlg.refresh()
    if model.dlg.ui then
        local c=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f",c['stacking']),true)
        simUI.setEditValue(model.dlg.ui,3,simBWF.format("%.0f",c['stackingShift']*1000),true)
        simUI.setEditValue(model.dlg.ui,5,c['subtype'],true)

        local nailComboItems={
            {"none",0},
            {"steel",1},
            {"plastic",2}
        }
        simBWF.populateCombobox(model.dlg.ui,2,nailComboItems,{},nailComboItems[c['gripperType'][9]+1][1],false,nil)

        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2,simStopped,true)
        simUI.setEnabled(model.dlg.ui,5,false,true)
    end
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
                <label text="Type"/>
                <edit id="5"/>
                
                <label text="Nails"/>
                <combobox id="2" on-change="model.dlg.nailComboChange_callback"></combobox>
                
                <label text="Stacking"/>
                <edit on-editing-finished="model.dlg.stackingChange_callback" id="1"/>

                <label text="Stacking shift (mm)"/>
                <edit on-editing-finished="model.dlg.stackingShiftChange_callback" id="3"/>
                
                <label text="Pick & place settings"/>
                <button text="Edit" on-click="model.dlg.pickAndPlaceSettings_callback" id="4" />
                
            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
            </group>
        ]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil,false,false,false,'')
        
        model.dlg.refresh()
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
