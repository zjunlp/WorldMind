model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local c=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setCheckboxValue(model.dlg.ui,1,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1)~=0),true)

        model.dlg.updateEnabledDisabledItems()
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,enabled)
        simUI.setEnabled(model.dlg.ui,1,enabled)
    end
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.gripperActionColorChangeClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,1)
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml=[[
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
                <label text="Change platform color with gripper action" />
                <checkbox text="" checked="false" on-change="model.dlg.gripperActionColorChangeClick_callback" id="1"/>
                
                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
            </group>
        ]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

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
