model.dlg={}
model.dlg.pickPlaceDlg={}

function model.dlg.pickPlaceDlg.partRobotSettingsClose_cb(dlgPos)
    model.dlg.pickPlaceDlg.previousDlgPos=dlgPos
end

function model.dlg.pickPlaceDlg.partRobotSettingsApply_cb(robotInfo)
    local c=model.getPartData(model.dlg.selectedPartId)
    c.robotInfo.overrideGripperSettings=robotInfo.overrideGripperSettings
    c.robotInfo.speed=robotInfo.speed
    c.robotInfo.accel=robotInfo.accel
    for i=1,2,1 do
        c.robotInfo.dwellTime[i]=robotInfo.dwellTime[i]
        c.robotInfo.approachHeight[i]=robotInfo.approachHeight[i]
        c.robotInfo.useAbsoluteApproachHeight[i]=robotInfo.useAbsoluteApproachHeight[i]
        c.robotInfo.departHeight[i]=robotInfo.departHeight[i]
        c.robotInfo.rounding[i]=robotInfo.rounding[i]
        c.robotInfo.nullingAccuracy[i]=robotInfo.nullingAccuracy[i]
        for j=1,3,1 do
            c.robotInfo.offset[i][j]=robotInfo.offset[i][j]
        end
        --c.robotInfo.freeModeTiming[i]=robotInfo.freeModeTiming[i]
        --c.robotInfo.actionModeTiming[i]=robotInfo.actionModeTiming[i]
        c.robotInfo.relativeToBelt[i]=robotInfo.relativeToBelt[i]
    end
    c.robotInfo.actionTemplates=robotInfo.actionTemplates
    c.robotInfo.pickActions=robotInfo.pickActions
    c.robotInfo.multiPickActions=robotInfo.multiPickActions
    c.robotInfo.placeActions=robotInfo.placeActions
    model.updatePartData(model.dlg.selectedPartId,c)
    simBWF.markUndoPoint()
end

function model.dlg.partRobotSettings_callback()
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        model.pickPlaceDlg.display(c.robotInfo,"'"..simBWF.getObjectAltName(model.dlg.selectedPartId).."' pick settings",false,model.dlg.pickPlaceDlg.partRobotSettingsApply_cb,model.dlg.pickPlaceDlg.partRobotSettingsClose_cb,model.dlg.pickPlaceDlg.previousDlgPos)
    end
end


function model.dlg.editDestinations_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
    
        local allDest=simBWF.getAllPossiblePartDestinations()
        local txt="{"
        local c=false
        for i=simBWF.PART_DESTINATIONFIRST_REF,simBWF.PART_DESTINATIONLAST_REF,1 do
            local r=simBWF.getReferencedObjectHandle(model.dlg.selectedPartId,i)
            if r>=0 then
                if c then
                    txt=txt..","
                end
                txt=txt.."'"..sim.getObjectName(r+sim.handleflag_altname).."'"
            end
            c=true
        end
        txt=txt.."}\n\n--[[tmpRem\n\n"
        txt=txt..[[
a) Usage:
   {<destination 1>,<destination 2>, ...,<destination N>}. Leave list empty for default behaviour. 

b) Example:
   {'pickLocationFrame1','trackingWindow2'}

c) Following are possible destinations:]].."\n    "
        c=false
        for key,value in pairs(allDest) do
            if c then
                txt=txt..", "
            end
            txt=txt.."'"..key.."'"
            c=true
        end
        txt=txt.."\n\n--]]"

        local s="800 400"
        local p="200 200"
        if model.dlg.destinationDlgDlgSize then
            s=model.dlg.destinationDlgDlgSize[1]..' '..model.dlg.destinationDlgDlgSize[2]
        end
        if model.dlg.destinationDlgPos then
            p=model.dlg.destinationDlgPos[1]..' '..model.dlg.destinationDlgPos[2]
        end
        local xml = [[ <editor title="]].."Part destinations"..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="false" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            

        local res=false
        while res==false do
            txt,model.dlg.destinationDlgDlgSize,model.dlg.destinationDlgPos=sim.openTextEditor(txt,xml)
            local toExecute="return "..txt
            local theTable
            res,theTable=sim.executeLuaCode(toExecute)
            if res then
                if type(theTable)=='table' then
                    local t={}
                    for i=1,#theTable,1 do
                        if type(theTable[i])~='string' then
                            res=false
                            break
                        else
                            local h=sim.getObjectHandle(theTable[i]..'@alt@silentError')
                            if h<0 then
                                res=false
                                break
                            else
                                t[#t+1]=h
                            end
                        end
                    end
                    if #t>simBWF.PART_DESTINATIONLAST_REF-simBWF.PART_DESTINATIONFIRST_REF+1 then
                        res=false
                    end
                    if res then
                        if #t==0 then
                            -- Let's check for situations like: txt={xxx} that would also result in an empty table:
                            local tmp=string.gsub(txt," ","")
                            local tmp=string.gsub(tmp,"\t","")
                            if string.find(tmp,"{}")~=1 then
                                res=false
                            end
                        end
                        if res then
                            for i=simBWF.PART_DESTINATIONFIRST_REF,simBWF.PART_DESTINATIONLAST_REF,1 do
                                simBWF.setReferencedObjectHandle(model.dlg.selectedPartId,i,-1)
                            end
                            for i=1,#t,1 do
                                simBWF.setReferencedObjectHandle(model.dlg.selectedPartId,simBWF.PART_DESTINATIONFIRST_REF+i-1,t[i])
                            end
                            break
                        end
                    end
                else
                    res=false
                end
            end
            if not res then
                local r=sim.msgBox(sim.msgbox_type_critical,sim.msgbox_buttons_yesno,'Input Error',"The destination list is ill-formated. Do you wish to correct it?")
                if r==sim.msgbox_return_no then
                    break
                end
            end
        end
        
        if res then
            model.updatePartData(model.dlg.selectedPartId,model.getPartData(model.dlg.selectedPartId))
            simBWF.markUndoPoint()
        end
    end
 end


function model.dlg.populatePartRepoTable()
    local parts=model.getPartTable()
    local retVal={}
    simUI.clearTable(model.dlg.ui,10)
    simUI.setRowCount(model.dlg.ui,10,0)
    for i=1,#parts,1 do
        local part=parts[i]
        simUI.setRowCount(model.dlg.ui,10,i)
        simUI.setRowHeight(model.dlg.ui,10,i-1,25,25)
        simUI.setItem(model.dlg.ui,10,i-1,0,part[1])
        retVal[i]=part[2]
    end
    return retVal
end

function model.dlg.refresh()
    if model.dlg.ui then
        simUI.setColumnCount(model.dlg.ui,10,1)
        simUI.setColumnWidth(model.dlg.ui,10,0,310,310)
        model.dlg.tablePartIds=model.dlg.populatePartRepoTable()
        
        if model.dlg.selectedPartId>=0 then
            local c=model.getPartData(model.dlg.selectedPartId)
            simUI.setCheckboxValue(model.dlg.ui,41,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1)~=0),true)
            simUI.setCheckboxValue(model.dlg.ui,42,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],2)~=0),true)
            simUI.setCheckboxValue(model.dlg.ui,9,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],4)~=0),true)
            simUI.setCheckboxValue(model.dlg.ui,11,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],8)~=0),true)
            simUI.setCheckboxValue(model.dlg.ui,44,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],16)~=0),true)
            
            local pallets=simBWF.getAvailablePallets()
            local selected=simBWF.NONE_TEXT
            for i=1,#pallets,1 do
                if pallets[i][2]==simBWF.getReferencedObjectHandle(model.dlg.selectedPartId,simBWF.PART_PALLET_REF) then
                    selected=pallets[i][1]
                    break
                end
            end
            
            local off=c['palletOffset']
            simUI.setEditValue(model.dlg.ui,8,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
            
            model.dlg.comboPallet=simBWF.populateCombobox(model.dlg.ui,7,pallets,{},selected,true,{{simBWF.NONE_TEXT,-1}})
            for i=1,#model.dlg.tablePartIds,1 do
                if model.dlg.tablePartIds[i]==model.dlg.selectedPartId then
                    simUI.setTableSelection(model.dlg.ui,10,i-1,0)
                    break
                end
            end
        else
            simUI.setCheckboxValue(model.dlg.ui,41,simBWF.getCheckboxValFromBool(false),true)
            simUI.setCheckboxValue(model.dlg.ui,44,simBWF.getCheckboxValFromBool(false),true)
            simUI.setCheckboxValue(model.dlg.ui,42,simBWF.getCheckboxValFromBool(false),true)
            simUI.setCheckboxValue(model.dlg.ui,9,simBWF.getCheckboxValFromBool(false),true)
            simUI.setCheckboxValue(model.dlg.ui,11,simBWF.getCheckboxValFromBool(false),true)
            simUI.setComboboxItems(model.dlg.ui,7,{},-1,true)
            simUI.setTableSelection(model.dlg.ui,10,-1,-1)
            simUI.setEditValue(model.dlg.ui,8,"0, 0, 0",true)
        end
        
        model.dlg.updateEnabledDisabledItems()
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,6,model.dlg.selectedPartId>=0 and simStopped,true)
        simUI.setEnabled(model.dlg.ui,7,model.dlg.selectedPartId>=0 and simStopped,true)
        local pall=false
        if model.dlg.selectedPartId>=0 then
            pall=simBWF.getReferencedObjectHandle(model.dlg.selectedPartId,simBWF.PART_PALLET_REF)>=0
        end
        simUI.setEnabled(model.dlg.ui,8,pall and simStopped,true)
        simUI.setEnabled(model.dlg.ui,9,pall and simStopped,true)
        simUI.setEnabled(model.dlg.ui,11,pall and simStopped,true)
        simUI.setEnabled(model.dlg.ui,41,model.dlg.selectedPartId>=0 and not pall and simStopped,true)
        simUI.setEnabled(model.dlg.ui,42,model.dlg.selectedPartId>=0 and not pall and simStopped,true)
        simUI.setEnabled(model.dlg.ui,43,model.dlg.selectedPartId>=0,true)
        simUI.setEnabled(model.dlg.ui,44,model.dlg.selectedPartId>=0 and not pall and simStopped,true)
    end
end

function model.dlg.palletOffset_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        
        local i=1
        local t={0,0,0}
        for token in (newVal..","):gmatch("([^,]*),") do
            t[i]=tonumber(token)
            if t[i]==nil then t[i]=0 end
            t[i]=t[i]*0.001
            if t[i]>0.2 then t[i]=0.2 end
            if t[i]<-0.2 then t[i]=-0.2 end
            i=i+1
        end
        c['palletOffset']={t[1],t[2],t[3]}
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.palletChange_callback(ui,id,newIndex)
    if model.dlg.selectedPartId>=0 then
        simBWF.setReferencedObjectHandle(model.dlg.selectedPartId,simBWF.PART_PALLET_REF,model.dlg.comboPallet[newIndex+1][2])
        local c=model.getPartData(model.dlg.selectedPartId)
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.invisiblePart_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
        if newVal==0 then
            c['bitCoded']=c['bitCoded']-1
        end
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.invisibleToOtherParts_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
        if newVal==0 then
            c['bitCoded']=c['bitCoded']-2
        end
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.attachPart_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
        if newVal==0 then
            c['bitCoded']=c['bitCoded']-16
        end
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.ignoreBasePart_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
        if newVal==0 then
            c['bitCoded']=c['bitCoded']-4
        end
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.usePalletColors_callback(ui,id,newVal)
    if model.dlg.selectedPartId>=0 then
        local c=model.getPartData(model.dlg.selectedPartId)
        c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
        if newVal==0 then
            c['bitCoded']=c['bitCoded']-8
        end
        model.updatePartData(model.dlg.selectedPartId,c)
        simBWF.markUndoPoint()
    end
    model.dlg.refresh()
end

function model.dlg.onPartRepoCellActivate(uiHandle,id,row,column,value)
    if model.dlg.selectedPartId>=0 then
        local valid=false
        if #value>0 and (sim.getSimulationState()==sim.simulation_stopped) then
--            value=string.match(value,"[^ ]+")
            value=simBWF.getValidName(value,true)
            if not model.doesPartWithNameExist(value) then
                valid=true
                simBWF.setObjectAltName(model.dlg.selectedPartId,value)
--                model.updatePartData(model.dlg.selectedPartId,partData)
                model.updatePluginRepresentation()
            end
        end
        simUI.setItem(uiHandle,10,row,0,simBWF.getObjectAltName(model.dlg.selectedPartId))
    end
end

function model.dlg.onPartRepoTableSelectionChange(uiHandle,id,row,column)
    if row>=0 then
        model.dlg.selectedPartId=model.dlg.tablePartIds[row+1]
    else
        model.dlg.selectedPartId=-1
    end
    model.dlg.refresh()
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.onPartRepoTableKeyPress(uiHandle,id,key,text)
    if model.dlg.selectedPartId>=0 then
        if text:byte(1,1)==27 then
            -- esc
            model.dlg.selectedPartId=-1
            simUI.setTableSelection(uiHandle,10,-1,-1)
            model.dlg.refresh()
            model.dlg.updateEnabledDisabledItems()
        end
        if text:byte(1,1)==13 then
            -- enter or return
        end
        if text:byte(1,1)==127 or text:byte(1,1)==8 then
            -- del or backspace
            if sim.getSimulationState()==sim.simulation_stopped then
                model.removePart(model.dlg.selectedPartId)
                model.dlg.tablePartIds=model.dlg.populatePartRepoTable()
                model.dlg.selectedPartId=-1
                model.dlg.refresh()
                model.dlg.updateEnabledDisabledItems()
            end
        end
    end
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
            <table show-horizontal-header="false" autosize-horizontal-header="true" show-grid="false" selection-mode="row" editable="true" on-cell-activate="model.dlg.onPartRepoCellActivate" on-selection-change="model.dlg.onPartRepoTableSelectionChange" on-key-press="model.dlg.onPartRepoTableKeyPress" id="10"/>

            <group layout="form" flat="false">
                <label text="Part properties" style="* {font-weight: bold;}"/><label text=""/>
                
                <label text="Destinations"/>
                <button text="Edit" on-click="model.dlg.editDestinations_callback" style="* {min-width: 175px;}" id="6" />

                <label text="Pallet"/>
                <combobox id="7" on-change="model.dlg.palletChange_callback"/>

                <label text="Pallet offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.palletOffset_callback" id="8"/>
                
                <label text="Ignore base part"/>
                <checkbox text="" on-change="model.dlg.ignoreBasePart_callback" id="9" />
                
                <label text="Use pallet colors"/>
                <checkbox text="" on-change="model.dlg.usePalletColors_callback" id="11" />
                
                <label text="Invisible"/>
                <checkbox text="" on-change="model.dlg.invisiblePart_callback" id="41" />

                <label text="Invisible to other parts"/>
                <checkbox text="" on-change="model.dlg.invisibleToOtherParts_callback" id="42" />
                
                <label text="Attach to other parts"/>
                <checkbox text="" on-change="model.dlg.attachPart_callback" id="44" />
                
                <label text="Pick settings"/>
                <button text="Edit" on-click="model.dlg.partRobotSettings_callback" id="43" />
                
            </group>
        ]]


        model.dlg.ui=simBWF.createCustomUi(xml,'Part Repository',model.dlg.previousDlgPos,true,'model.dlg.onCloseDlg')-- modal,resizable,activate,additionalUiAttribute--]])

    
        model.dlg.selectedPartId=-1
        
        model.dlg.refresh()
    end
end

function model.dlg.onCloseDlg()
    sim.setBoolParameter(sim.boolparam_br_partrepository,false)
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
    if sim.getBoolParameter(sim.boolparam_br_partrepository) then
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
