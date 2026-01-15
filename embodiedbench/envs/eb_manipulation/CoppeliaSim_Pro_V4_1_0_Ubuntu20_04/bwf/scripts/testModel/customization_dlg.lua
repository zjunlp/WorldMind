model.dlg={}

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local config=model.readInfo()
--        simUI.setEnabled(model.dlg.ui,2,config.visionBoxServerName~=simBWF.NONE_TEXT,true)
--        simUI.setEnabled(model.dlg.ui,5,simStopped,true) -- simBWF.getReferencedObjectHandle(model,model.objRefIdx.PALLET)>=0,true)
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setCheckboxValue(model.dlg.ui,30,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)

        
        local c=model.readInfo()
        local loc=model.getAvailableConnections()
        model.dlg.comboConnection=simBWF.populateCombobox(model.dlg.ui,11,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONNECTION)),true,{{simBWF.NONE_TEXT,-1}})
        
--        model.dlg.updateVisionBoxServerNameCombobox()
        
        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.connectionChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboConnection[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONNECTION,newLoc)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.hiddenDuringSimulation_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end


function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General">
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365" style="* {min-width: 300px;}"/>
                
                <label text="Input"/>
                <combobox id="11" on-change="model.dlg.connectionChange_callback">
                </combobox>
            </group>
            </tab>
            <tab title="More">
            <group layout="form" flat="false">
                <label text="Hidden during simulation"/>
                <checkbox text="" checked="false" on-change="model.dlg.hiddenDuringSimulation_callback" id="30"/>
            </group>
            </tab>
            
       </tabs>
        ]]

        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil,false,false,false)

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
