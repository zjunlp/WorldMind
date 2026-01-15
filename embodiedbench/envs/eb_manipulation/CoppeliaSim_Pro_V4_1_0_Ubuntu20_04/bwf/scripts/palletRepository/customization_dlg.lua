model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        
        simUI.setColumnCount(model.dlg.ui,10,1)
        simUI.setColumnWidth(model.dlg.ui,10,0,310,310)
        
        model.dlg.tablePalletHandles=model.dlg.populatePalletRepoTable()
        
        model.dlg.updateEnabledDisabledItems()
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
        
        model.dlg.updateEnabledDisabledItems()
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        
        simUI.setEnabled(model.dlg.ui,1,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2,model.dlg.selectedPalletHandle>=0,true)
        simUI.setEnabled(model.dlg.ui,3,simStopped and model.dlg.selectedPalletHandle>=0,true)
    end
end

function model.dlg.populatePalletRepoTable()
    local retVal={}
    local allPallets=model.getAllPalletHandles()
    simUI.clearTable(model.dlg.ui,10)
    simUI.setRowCount(model.dlg.ui,10,0)
    for i=1,#allPallets,1 do
        local pallet=allPallets[i]
        simUI.setRowCount(model.dlg.ui,10,i)
        simUI.setRowHeight(model.dlg.ui,10,i-1,25,25)
        simUI.setItem(model.dlg.ui,10,i-1,0,simBWF.getObjectAltName(pallet))
        retVal[i]=pallet
    end
    return retVal
end

function model.dlg.addNewPalletClick_callback()
    local palletHandle,name=model.addNewPallet()
    local rc=simUI.getRowCount(model.dlg.ui,10)
    simUI.setRowCount(model.dlg.ui,10,rc+1)
    simUI.setRowHeight(model.dlg.ui,10,rc,25,25)
    simUI.setItem(model.dlg.ui,10,rc,0,name)
    model.dlg.tablePalletHandles[rc+1]=palletHandle
    model.dlg.selectedPalletHandle=palletHandle
    simUI.setTableSelection(model.dlg.ui,10,rc,0)
    model.dlg.updateEnabledDisabledItems()
    simBWF.announcePalletWasCreated()
end

function model.dlg.onRejectPalletEdit()
    sim.auxFunc('enableRendering')
end

function model.dlg.onAcceptPalletEdit(arg1)
    sim.auxFunc('enableRendering')
    simBWF.writePalletInfo(model.dlg.selectedPalletHandle,arg1)
    model.afterReceivingPalletDataFromPlugin(model.dlg.selectedPalletHandle)
    model.updatePluginRepresentation_onePallet(model.dlg.selectedPalletHandle)
end

function model.dlg.editPalletClick_callback()
    if model.dlg.selectedPalletHandle>=0 then
        model.beforeSendingPalletDataToPlugin(model.dlg.selectedPalletHandle)
        local palletData=simBWF.readPalletInfo(model.dlg.selectedPalletHandle)
        local data={}
        data.pallet=palletData
        data.onReject='model.dlg.onRejectPalletEdit'
        data.onAccept='model.dlg.onAcceptPalletEdit'
        sim.auxFunc('disableRendering')
        local reply=simBWF.query('pallet_edit',data)
        if reply~='ok' then
            sim.auxFunc('enableRendering')
        end
    end
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.duplicatePalletClick_callback()
    if model.dlg.selectedPalletHandle>=0 then
        local name
        model.dlg.selectedPalletHandle,name=model.duplicatePallet(model.dlg.selectedPalletHandle)
        local rc=simUI.getRowCount(model.dlg.ui,10)
        simUI.setRowCount(model.dlg.ui,10,rc+1)
        simUI.setRowHeight(model.dlg.ui,10,rc,25,25)
        simUI.setItem(model.dlg.ui,10,rc,0,name)
        model.dlg.tablePalletHandles[rc+1]=model.dlg.selectedPalletHandle
        simUI.setTableSelection(model.dlg.ui,10,rc,0)
    end
    model.dlg.updateEnabledDisabledItems()
    simBWF.announcePalletWasCreated()
end

function model.dlg.onPalletRepoDlgCellActivate(uiHandle,id,row,column,value)
    if model.dlg.selectedPalletHandle>=0 then
        local valid=false
        if #value>0 then
            value=simBWF.getValidName(value,true)
            if model.getPalletWithName(value)==-1 then
                valid=true
                simBWF.setObjectAltName(model.dlg.selectedPalletHandle,value)
                value=simBWF.getObjectAltName(model.dlg.selectedPalletHandle)
                local data=simBWF.readPalletInfo(model.dlg.selectedPalletHandle)
                data.name=value
                simBWF.writePalletInfo(model.dlg.selectedPalletHandle,data)
                model.updatePluginRepresentation_onePallet(model.dlg.selectedPalletHandle)
                simUI.setItem(model.dlg.ui,10,row,0,value)
                simBWF.announcePalletWasRenamed(model.dlg.selectedPalletHandle)
            end
        end
        if not valid then
            value=simBWF.getObjectAltName(model.dlg.selectedPalletHandle)
            simUI.setItem(model.dlg.ui,10,row,0,value)
        end
    end
end

function model.dlg.onPalletRepoDlgTableSelectionChange(uiHandle,id,row,column)
    if row>=0 then
        model.dlg.selectedPalletHandle=model.dlg.tablePalletHandles[row+1]
    else
        model.dlg.selectedPalletHandle=-1
    end
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.onPalletRepoDlgTableKeyPress(uiHandle,id,key,text)
    if model.dlg.selectedPalletHandle>=0 then
        if text:byte(1,1)==27 then
            -- esc
            model.dlg.selectedPalletHandle=-1
            simUI.setTableSelection(model.dlg.ui,10,-1,-1)
            model.dlg.updateEnabledDisabledItems()
        end
        if text:byte(1,1)==13 then
            -- enter or return
        end
        if text:byte(1,1)==127 or text:byte(1,1)==8 then
            -- del or backspace
            if sim.getSimulationState()==sim.simulation_stopped then
                model.removePallet(model.dlg.selectedPalletHandle)
                model.dlg.tablePalletHandles=model.dlg.populatePalletRepoTable()
                model.dlg.selectedPalletHandle=-1
                model.dlg.updateEnabledDisabledItems()
                simBWF.announcePalletWasCreated()
            end
        end
    end
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[

            <table show-horizontal-header="false" autosize-horizontal-header="true" show-grid="false" selection-mode="row" editable="true" on-cell-activate="model.dlg.onPalletRepoDlgCellActivate" on-selection-change="model.dlg.onPalletRepoDlgTableSelectionChange" on-key-press="model.dlg.onPalletRepoDlgTableKeyPress" id="10"/>
            <button text="Add new pallet" style="* {min-width: 300px;}" on-click="model.dlg.addNewPalletClick_callback" id="1" />
            <button text="Edit selected pallet" style="* {min-width: 300px;}" on-click="model.dlg.editPalletClick_callback" id="2" />
            <button text="Duplicate selected pallet" style="* {min-width: 300px;}" on-click="model.dlg.duplicatePalletClick_callback" id="3" />

            ]]
--            <button text="Edit pallets"  style="* {min-width: 300px;}" on-click="editPalletsClick_callback" id="99" />
--            <button text="Delete selected pallet" style="* {min-width: 300px;}" on-click="deleteClick_callback" id="4" />


        model.dlg.ui=simBWF.createCustomUi(xml,'Pallet Repository',model.dlg.previousDlgPos,true,'model.dlg.onClose')

        model.dlg.selectedPalletHandle=-1
        
        model.dlg.refresh()
        
    end
end

function model.dlg.onClose()
    sim.setBoolParameter(sim.boolparam_br_palletrepository,false)
    model.dlg.removeDlg()
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
    if sim.getBoolParameter(sim.boolparam_br_palletrepository) then
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
