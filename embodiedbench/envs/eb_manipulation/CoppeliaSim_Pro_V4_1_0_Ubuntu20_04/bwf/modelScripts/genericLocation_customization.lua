function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function ext_getItemData_pricing()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.type='locationFrame'
    obj.frameType='pickOrPlace'
    obj.brVersion=0
    return obj
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='location'
    end
    if not info['name'] then
        info['name']='LOCATION1'
    end
    if not info['status'] then
        info['status']='free'
    end
    if not info['bitCoded'] then
        info['bitCoded']=1
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDLOCATION)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDLOCATION,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDLOCATION,'')
    end
end

function setColor(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model,nil,sim.colorcomponent_specular,{spec,spec,spec})
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function updateStatusText()
    if ui then
        local config=readInfo()
        local oldVal=simUI.getEditValue(ui,2)
        if oldVal~=config['status'] then
            simUI.setEditValue(ui,2,config['status'],true)
        end
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,config['name'],true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        local red,green,blue,spec=getColor()
        simUI.setSliderValue(ui,11,red*100,true)
        simUI.setSliderValue(ui,12,green*100,true)
        simUI.setSliderValue(ui,13,blue*100,true)
        simUI.setSliderValue(ui,14,spec*100,true)
        updateStatusText()
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,1,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,11,enabled,true)
        simUI.setEnabled(ui,12,enabled,true)
        simUI.setEnabled(ui,13,enabled,true)
        simUI.setEnabled(ui,14,enabled,true)

        simUI.setEnabled(ui,2,false,true)
    end
end

function nameChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=newVal
    if v then
        if v~=c['name'] then
            simBWF.markUndoPoint()
            c['name']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function hidden_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function redChange_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(newVal/100,g,b,s)
end

function greenChange_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,newVal/100,b,s)
end

function blueChange_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,newVal/100,s)
end

function specularChange_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,b,newVal/100)
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Location name"/>
                <edit on-editing-finished="nameChange_callback" id="1"/>

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="hidden_callback" id="3" />

                <label text="Status"/>
                <edit id="2"/>

            <label text="" style="* {margin-left: 180px;}"/>
            <label text="" style="* {margin-left: 180px;}"/>

            <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange_callback" id="11"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange_callback" id="12"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange_callback" id="13"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="specularChange_callback" id="14"/>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,false,nil,false,false,false,'layout="form"')

        setDlgItemContent()
        updateEnabledDisabledItemsDlg()
    end
end

function showDlg()
    if not ui then
        createDlg()
    end
end

function removeDlg()
    if ui then
        local x,y=simUI.getPosition(ui)
        previousDlgPos={x,y}
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    
    updatePluginRepresentation()
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
end

showOrHideUiIfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg()
    else
        removeDlg()
    end
end

function sysCall_nonSimulation()
    showOrHideUiIfNeeded()
end

function sysCall_sensing()
    if simJustStarted then
        updateEnabledDisabledItemsDlg()
    end
    simJustStarted=nil
    updateStatusText()
    showOrHideUiIfNeeded()
end


function sysCall_suspend()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    local c=readInfo()
    c['status']='free'
    writeInfo(c)
    sim.setModelProperty(model,0)
    updateStatusText()
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    simJustStarted=true
    local c=readInfo()
    if sim.boolAnd32(c['bitCoded'],1)~=0 then
        sim.setModelProperty(model,sim.modelproperty_not_visible)
    end
end

function sysCall_beforeInstanceSwitch()
    removeDlg()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    removeDlg()
    removeFromPluginRepresentation()
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end