model.dlg={}

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local config=model.readInfo()
--        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
--        simUI.setEnabled(model.dlg.ui,5,simStopped,true) -- simBWF.getReferencedObjectHandle(model,model.objRefIdx.PALLET)>=0,true)
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setCheckboxValue(model.dlg.ui,30,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)

        model.dlg.updateIoHubSerialCombobox()
        model.dlg.updateConveyorInputBoxComboboxes()
        model.dlg.updateInputBoxComboboxes()
        model.dlg.updateOutputBoxComboboxes()
        
        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.updateConveyorInputBoxComboboxes()
    model.dlg.conveyorInputBox_comboboxItems={}
    local loc=model.getAvailableConveyors()
    for i=1,2,1 do
        model.dlg.conveyorInputBox_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,1+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updateInputBoxComboboxes()
    model.dlg.inputBox_comboboxItems={}
    local loc=model.getAvailableInputBoxes()
    for i=3,8,1 do
        model.dlg.inputBox_comboboxItems[i-2]=simBWF.populateCombobox(model.dlg.ui,1+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updateOutputBoxComboboxes()
    model.dlg.outputBox_comboboxItems={}
    local loc=model.getAvailableOutputBoxes()
    for i=1,8,1 do
        model.dlg.outputBox_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,11+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.inputChange_callback(ui,id,newIndex)
    if id<3 then
        -- Input 1&2 (conveyors)
        local newLoc=model.dlg.conveyorInputBox_comboboxItems[id][newIndex+1][2]
        if newLoc~=-1 then
            simBWF.disconnectInputOrOutputBox(newLoc)
        end
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+id-1,newLoc)
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        model.dlg.updateConveyorInputBoxComboboxes()
    else
        -- Input 3-8 (input boxes)
        local newLoc=model.dlg.inputBox_comboboxItems[id-2][newIndex+1][2]
        if newLoc~=-1 then
            simBWF.disconnectInputOrOutputBox(newLoc)
        end
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+id-1,newLoc)
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        model.dlg.updateInputBoxComboboxes()
    end
end

function model.dlg.outputChange_callback(ui,id,newIndex)
    -- Outputs 1-8 (output boxes)
    local newLoc=model.dlg.outputBox_comboboxItems[id-10][newIndex+1][2]
    if newLoc~=-1 then
        simBWF.disconnectInputOrOutputBox(newLoc)
    end
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+id-11,newLoc)
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updateOutputBoxComboboxes()
end

function model.dlg.serialComboChange_callback(uiHandle,id,newValue)
    local newSerial=model.dlg.comboSerials[newValue+1][1]
    local c=model.readInfo()
    c['iohubSerial']=newSerial
    model.writeInfo(c)
    simBWF.markUndoPoint()
--    model.dlg.updateIoHubSerialCombobox()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.updateIoHubSerialCombobox()
    local c=model.readInfo()
    local resp,data
    if simBWF.isInTestMode() then
        resp='ok'
        data={}
        data.serials={'serial-1','serial-2','serial-x'}
    else
        resp,data=simBWF.query('get_iohubSerials')
        if resp~='ok' then
            data.serials={}
        end
    end
    
    local selected=c['iohubSerial']
    local isKnown=false
    local items={}
    for i=1,#data.serials,1 do
        if data.serials[i]==selected then
            isKnown=true
        end
        items[#items+1]={data.serials[i],i}
    end
    if not isKnown then
        table.insert(items,1,{selected,#items+1})
    end
    if selected~=simBWF.NONE_TEXT then
        table.insert(items,1,{simBWF.NONE_TEXT,#items+1})
    end
    model.dlg.comboSerials=simBWF.populateCombobox(model.dlg.ui,1200,items,{},selected,false,{})
--    model.updatePluginRepresentation()
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
        model.updatePluginRepresentation()
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
                
                <label text="IO hub serial"/>
                <combobox id="1200" on-change="model.dlg.serialComboChange_callback"> </combobox>
            </group>
            </tab>
            <tab title="Input Connections">
            <group layout="form" flat="false">
                <label text="Conveyor1" />
                <combobox on-change="model.dlg.inputChange_callback" id="1" />
                
                <label text="Conveyor2" />
                <combobox on-change="model.dlg.inputChange_callback" id="2" />

                <label text="Input3" />
                <combobox on-change="model.dlg.inputChange_callback" id="3" />

                <label text="Input4" />
                <combobox on-change="model.dlg.inputChange_callback" id="4" />

                <label text="Input5" />
                <combobox on-change="model.dlg.inputChange_callback" id="5" />

                <label text="Input6" />
                <combobox on-change="model.dlg.inputChange_callback" id="6" />

                <label text="Input7" />
                <combobox on-change="model.dlg.inputChange_callback" id="7" />

                <label text="Input8" />
                <combobox on-change="model.dlg.inputChange_callback" id="8" />
            </group>
            </tab>
            <tab title="Output Connections">
            <group layout="form" flat="false">
                <label text="Output1" />
                <combobox on-change="model.dlg.outputChange_callback" id="11" />
                
                <label text="Output2" />
                <combobox on-change="model.dlg.outputChange_callback" id="12" />

                <label text="Output3" />
                <combobox on-change="model.dlg.outputChange_callback" id="13" />

                <label text="Output4" />
                <combobox on-change="model.dlg.outputChange_callback" id="14" />

                <label text="Output5" />
                <combobox on-change="model.dlg.outputChange_callback" id="15" />

                <label text="Output6" />
                <combobox on-change="model.dlg.outputChange_callback" id="16" />

                <label text="Output7" />
                <combobox on-change="model.dlg.outputChange_callback" id="17" />

                <label text="Output8" />
                <combobox on-change="model.dlg.outputChange_callback" id="18" />
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
