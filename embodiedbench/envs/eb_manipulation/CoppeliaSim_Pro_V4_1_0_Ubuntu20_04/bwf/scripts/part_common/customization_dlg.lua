model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        model.dlg.refresh_specific()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.addToRepo_callback()
    local repo=simBWF.getPartRepositoryHandles()
    simBWF.callCustomizationScriptFunction('model.ext.insertPart',repo,model.handle)
    model.finalizeModel=true
    sim.removeScript(sim.handle_self)
end

function model.dlg.instanciate_callback()
    model.finalizeModel=true
    local c=simBWF.readPartInfo(model.handle)
    c.instanciated=true
    simBWF.writePartInfo(model.handle,c)
    sim.removeScript(sim.handle_self)
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml = '<tabs id="77"> '..model.dlg.getSpecificTabContent()
        xml=xml..[[ <tab title="Finalization">
                <button text="Add part to repository"  on-click="model.dlg.addToRepo_callback"/>
                <button text="Instanciate part" on-click="model.dlg.instanciate_callback"/>
            </tab>
        </tabs>
        ]]

        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil--[[,modal,resizable,activate,additionalUiAttribute--]])

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
