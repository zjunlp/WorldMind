function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='multifeeder'
    end
    if not info['frequency'] then
        info['frequency']=1
    end
    if not info['algorithm'] then
        info['algorithm']=''
    end
    if not info['conveyorDist'] then
        info['conveyorDist']=0.2
    end
    if not info['feederDistribution'] then
        info['feederDistribution']="{1,'feederName'}"
    end

    if not info['bitCoded'] then
        info['bitCoded']=3 -- 1=hidden, 2=enabled, 4-31:0=frequency, 4=sensor triggered, 8=user, 12=conveyorTriggered, 16=multi-feeder triggered
    end
    if not info['multiFeederTriggerCnt'] then
        info['multiFeederTriggerCnt']=0
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.MULTIFEEDER)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    getDefaultInfoForNonExistingFields(data)
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,simBWF.modelTags.MULTIFEEDER,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.MULTIFEEDER,'')
    end
end

function frequencyChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>10 then v=10 end
        if v~=c['frequency'] then
            simBWF.markUndoPoint()
            c['frequency']=v
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,2,simBWF.format("%.2f",c['frequency']),true)
end

function conveyorDistanceChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>2 then v=2 end
        if v~=c['conveyorDist'] then
            simBWF.markUndoPoint()
            c['conveyorDist']=v
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,62,simBWF.format("%.0f",c['conveyorDist']/0.001),true)
end

function dropAlgorithmClick_callback()

    local s="800 600"
    local p="100 100"
    if algoDlgSize then
        s=algoDlgSize[1]..' '..algoDlgSize[2]
    end
    if algoDlgPos then
        p=algoDlgPos[1]..' '..algoDlgPos[2]
    end
    local xml = [[
        <editor title="Feeder Algorithm" editable="true" searchable="true"
            tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190"
            selectionColor="128 128 255" size="]]..s..[[" position="]]..p..[["
            useVrepKeywords="true" isLua="true">
            <keywords1 color="152 0 0" >
            </keywords1>
            <keywords2 color="220 80 20" >
            </keywords2>
        </editor>
    ]]

    local c=readInfo()
    local initialText=c['algorithm']
    local modifiedText
    modifiedText,algoDlgSize,algoDlgPos=sim.openTextEditor(initialText,xml)
    c['algorithm']=modifiedText
    writeInfo(c)
    simBWF.markUndoPoint()
end

function getAvailableSensors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_BINARYSENSOR_INFO')
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function sensorComboChange_callback(ui,id,newIndex)
    local sens=comboSensor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,1,sens)
    simBWF.markUndoPoint()
end

function conveyorComboChange_callback(ui,id,newIndex)
    local conv=comboConveyor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,2,conv)
    simBWF.markUndoPoint()
end

function loadTheString()
    local f=loadstring("local counter=0\n".."return "..theString)
    return f
end

function distributionDlg(title,fieldName,tempComment)
    local prop=readInfo()
    local s="800 400"
    local p="200 200"
    if distributionDlgSize then
        s=distributionDlgSize[1]..' '..distributionDlgSize[2]
    end
    if distributionDlgPos then
        p=distributionDlgPos[1]..' '..distributionDlgPos[2]
    end
    local xml = [[ <editor title="]]..title..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            



    local initialText=prop[fieldName]
    if tempComment then
        initialText=initialText.."--[[tmpRem\n\n"..tempComment.."\n\n--]]"
    end
    local modifiedText
    modifiedText,distributionDlgSize,distributionDlgPos=sim.openTextEditor(initialText,xml)
    theString='{'..modifiedText..'}' -- variable needs to be global here
    local bla,f=pcall(loadTheString)
    local success=false
    if f then
        local res,err=xpcall(f,function(err) return debug.traceback(err) end)
        if res then
            modifiedText=simBWF.removeTmpRem(modifiedText)
            prop[fieldName]=modifiedText
            simBWF.markUndoPoint()
            writeInfo(prop)
            success=true
        end
    end
    if not success then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Input Error',"The distribution string is ill-formated.")
    end
end


function feederDistribution_callback(ui,id,newVal)
    local feeders=simBWF.getAllPossibleTriggerableFeeders(model)
    local tmpTxt="a) There are currently no valid feeders in the scene. Make sure that feeders you want to trigger from here are configured as 'Multi-feeder triggered'.\n\n"
    if feeders and #feeders>0 then
        tmpTxt="a) Following feeders can be triggered from here:"
        local lcnt=0
        for i=1,#feeders,1 do
            if lcnt~=0 then
                tmpTxt=tmpTxt..", "
            end
            if lcnt==4 then
                lcnt=0
            end
            if lcnt==0 then
                tmpTxt=tmpTxt.."\n   "
            end
            tmpTxt=tmpTxt.."'"..feeders[i][1].."'"
            lcnt=lcnt+1
        end
        tmpTxt=tmpTxt.."\n\n"
    end
    tmpTxt=tmpTxt..[[
b) Usage:
   {partialProportion1,<feederName>},{partialProportion2,<feederName>}

c) Example:
   {2.1,'genericFeeder'},{1.5,'multiFeederTrigger'} --> out of (2.1+1.5) triggers, 2.1 will go to feeder 'genericFeeder', and 1.5 will go to multi-feeder 'multiFeederTrigger' (statistically)

d) You may use the variable 'counter' as in following example:
   {counter%2,'genericFeeder'},{(counter+1)%2,'multiFeederTrigger'} --> triggers will be send in alternance to 'genericFeeder' and 'multiFeederTrigger']]
    distributionDlg('Feeder Distribution','feederDistribution',tmpTxt)
end

function updateEnabledDisabledItemsDlg1()
    if ui1 then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        local config=readInfo()
        local freq=sim.boolAnd32(config['bitCoded'],4+8+16)==0
        local sens=sim.boolAnd32(config['bitCoded'],4+8+16)==4
        local user=sim.boolAnd32(config['bitCoded'],4+8+16)==8
        local conv=sim.boolAnd32(config['bitCoded'],4+8+16)==12
        simUI.setEnabled(ui1,2,enabled and freq,true)
        simUI.setEnabled(ui1,999,enabled and sens,true)
        simUI.setEnabled(ui1,3,enabled and user,true)
        simUI.setEnabled(ui1,998,enabled and conv,true)
        simUI.setEnabled(ui1,62,enabled and conv,true)
        simUI.setEnabled(ui1,1000,enabled,true)
        simUI.setEnabled(ui1,1001,enabled,true)
        simUI.setEnabled(ui1,1002,enabled,true)
        simUI.setEnabled(ui1,1003,enabled,true)
        simUI.setEnabled(ui1,1004,enabled,true)
        simUI.setEnabled(ui1,11,enabled,true)
        simUI.setEnabled(ui1,30,enabled,true)
      --  simUI.setEnabled(ui1,40,enabled,true)
    end
end

function setDlgItemContent()
    if ui1 then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui1)
        simUI.setEditValue(ui1,62,simBWF.format("%.0f",config['conveyorDist']/0.001),true)
        simUI.setEditValue(ui1,2,simBWF.format("%.2f",config['frequency']),true)
        simUI.setCheckboxValue(ui1,30,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui1,40,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setRadiobuttonValue(ui1,1000,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==0),true)
        simUI.setRadiobuttonValue(ui1,1001,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==4),true)
        simUI.setRadiobuttonValue(ui1,1002,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==8),true)
        simUI.setRadiobuttonValue(ui1,1003,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==12),true)
        simUI.setRadiobuttonValue(ui1,1004,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==16),true)
        simBWF.setSelectedEditWidget(ui1,sel)
    end
end


function hidden_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function enabled_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function triggerTypeClick_callback(ui,id)
    local c=readInfo()
    local w={4+8+16,8+16,4+16,16,4+8}
    local v=w[id-1000+1]
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4+8+16)-v
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
    updateEnabledDisabledItemsDlg1()
end

function createDlg1()
    if (not ui1) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Distribution of feeders to trigger"/>
                <button text="Edit"  on-click="feederDistribution_callback" id="11" />

                <label text="" style="* {margin-left: 190px;}"/>
                <label text="" style="* {margin-left: 190px;}"/>

                <radiobutton text="Trigger frequency (1/s)" on-click="triggerTypeClick_callback" id="1000" />
                <edit on-editing-finished="frequencyChange_callback" id="2"/>

                <radiobutton text="Sensor triggered" on-click="triggerTypeClick_callback" id="1001" />
                <combobox id="999" on-change="sensorComboChange_callback">
                </combobox>

                <radiobutton text="Conveyor belt triggered" on-click="triggerTypeClick_callback" id="1003" />
                <combobox id="998" on-change="conveyorComboChange_callback">
                </combobox>

                <label text="Distance for trigger (mm)" style="* {margin-left: 20px;}"/>
                <edit on-editing-finished="conveyorDistanceChange_callback" id="62"/>

                <radiobutton text="User defined algorithm" on-click="triggerTypeClick_callback" id="1002" />
                <button text="Edit"  on-click="dropAlgorithmClick_callback" id="3" />

                <radiobutton text="Multi-feeder triggered" on-click="triggerTypeClick_callback" id="1004" />
                <label text=""/>

               <label text="Enabled"/>
                <checkbox text="" on-change="enabled_callback" id="40" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="hidden_callback" id="30" />
        ]]

        ui1=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlg1Pos,false,nil,false,false,false,'layout="form"')
        
        local c=readInfo()
        local loc=getAvailableConveyors()
        comboConveyor=simBWF.populateCombobox(ui1,998,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,2)),true,{{simBWF.NONE_TEXT,-1}})
        loc=getAvailableSensors()
        comboSensor=simBWF.populateCombobox(ui1,999,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,1)),true,{{simBWF.NONE_TEXT,-1}})

        setDlgItemContent()
        updateEnabledDisabledItemsDlg1()

    end
end

function showDlg1()
    if not ui1 then
        createDlg1()
    end
end

function removeDlg1()
    if ui1 then
        local x,y=simUI.getPosition(ui1)
        previousDlg1Pos={x,y}
        simUI.destroy(ui1)
        ui1=nil
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    -- Following for backward compatibility:
    if _info['sensor'] then
        simBWF.setReferencedObjectHandle(model,1,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['sensor']))
        _info['sensor']=nil
    end
    if _info['conveyor'] then
        simBWF.setReferencedObjectHandle(model,2,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['conveyor']))
        _info['conveyor']=nil
    end
    ----------------------------------------
    writeInfo(_info)
	
    updatePluginRepresentation()
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
end

showOrHideUi1IfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg1()
    else
        removeDlg1()
    end
end

function sysCall_nonSimulation()
    showOrHideUi1IfNeeded()
end

function sysCall_sensing()
    if simJustStarted then
        updateEnabledDisabledItemsDlg1()
    end
    simJustStarted=nil
    showOrHideUi1IfNeeded()
end

function sysCall_suspend()
    showOrHideUi1IfNeeded()
end

function sysCall_afterSimulation()
    updateEnabledDisabledItemsDlg1()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,1)
    local conf=readInfo()
    conf['multiFeederTriggerCnt']=0
    writeInfo(conf)
end

function sysCall_beforeSimulation()
    simJustStarted=true
    local conf=readInfo()
    conf['multiFeederTriggerCnt']=0
    writeInfo(conf)
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(conf['bitCoded'],1)==0)
    if not show then
        sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,0)
    end
end

function sysCall_beforeInstanceSwitch()
    removeDlg1()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    removeDlg1()
    removeFromPluginRepresentation()
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end