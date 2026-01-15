function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function setObjectSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    if x<0.05 then x=0.05 end
    if y<0.05 then y=0.05 end
    if z<0.05 then z=0.05 end
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='teleporter'
    end
    if not info['width'] then
        info['width']=0.1
    end
    if not info['length'] then
        info['length']=0.1
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['bitCoded'] then
        info['bitCoded']=1 -- 1=enabled, 2=isOrigin, 4=hiddenDuringSimulation
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_PARTTELEPORTER_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_PARTTELEPORTER_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_PARTTELEPORTER_INFO','')
    end
end

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    setObjectSize(model,w,l,h)
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['height']/0.001),true)


        simUI.setCheckboxValue(ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)==0),true)

        simUI.setRadiobuttonValue(ui,10,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setRadiobuttonValue(ui,11,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],2)==0),true)
        local red,green,blue=getColor()
        simUI.setSliderValue(ui,7,red*100,true)
        simUI.setSliderValue(ui,8,green*100,true)
        simUI.setSliderValue(ui,9,blue*100,true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local c=readInfo()
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,1,enabled,true)
        simUI.setEnabled(ui,2,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
        simUI.setEnabled(ui,10,enabled,true)
        simUI.setEnabled(ui,11,enabled,true)
        simUI.setEnabled(ui,12,enabled and sim.boolAnd32(c['bitCoded'],2)>0,true)
        simUI.setEnabled(ui,7,enabled,true)
        simUI.setEnabled(ui,8,enabled,true)
        simUI.setEnabled(ui,9,enabled,true)
    end
end

function getAvailableDestinations()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_PARTTELEPORTER_INFO')
        if data then
            data=sim.unpackTable(data)
            if sim.boolAnd32(data['bitCoded'],2)==0 then
                retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function enabled_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    writeInfo(c)
    setDlgItemContent()
end

function visible_callback(ui,id,newVal)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal~=0 then
        c['bitCoded']=c['bitCoded']-4
    end
    writeInfo(c)
    setDlgItemContent()
end

function setColor(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function originPodClick_callback(ui,id)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    writeInfo(c)
    updateEnabledDisabledItemsDlg()
    setDlgItemContent()
end

function destinationPodClick_callback(ui,id)
    simBWF.markUndoPoint()
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    c['bitCoded']=c['bitCoded']-2
    writeInfo(c)
    updateEnabledDisabledItemsDlg()
    setDlgItemContent()
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>1 then v=1 end
        if v~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function lengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>1 then v=1 end
        if v~=c['length'] then
            simBWF.markUndoPoint()
            c['length']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>1 then v=1 end
        if v~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function destinationChange_callback(ui,id,newIndex)
    local newLoc=comboDestination[newIndex+1][2]
    simBWF.markUndoPoint()
    simBWF.setReferencedObjectHandle(model,simBWF.TELEPORTER_DESTINATION_REF,newLoc)
end

function redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(newVal/100,g,b,s)
end

function greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,newVal/100,b,s)
end

function blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,newVal/100,s)
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="77">
    <tab title="Properties" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="2"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="3"/>

                <radiobutton text="Pod is source pod, destination pod is" on-click="originPodClick_callback" id="10" />
                <combobox id="12" on-change="destinationChange_callback">
                </combobox>

                <radiobutton text="Pod is destination pod" on-click="destinationPodClick_callback" id="11" />
                <label text=""/>

                <label text="Enabled"/>
                <checkbox text="" on-change="enabled_callback" id="4" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="visible_callback" id="5" />

    </tab>
    <tab title="Colors" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange" id="7"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange" id="8"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange" id="9"/>

                <label text="" style="* {margin-left: 180px;}"/>
                <label text="" style="* {margin-left: 180px;}"/>
    </tab>
    </tabs>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

        local c=readInfo()
        local loc=getAvailableDestinations()
        local exceptItems={}
        exceptItems[sim.getObjectName(model)]=true
        comboDestination=simBWF.populateCombobox(ui,12,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.TELEPORTER_DESTINATION_REF)),true,{{simBWF.NONE_TEXT,-1}})

        setDlgItemContent()
        updateEnabledDisabledItemsDlg()
        simUI.setCurrentTab(ui,77,dlgMainTabIndex,true)
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
        dlgMainTabIndex=simUI.getCurrentTab(ui,77)
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    dlgMainTabIndex=0
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    -- Following for backward compatibility:
    if _info['destination'] then
        simBWF.setReferencedObjectHandle(model,simBWF.TELEPORTER_DESTINATION_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['destination']))
        _info['destination']=nil
    end
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
--    updateStatusText()
    showOrHideUiIfNeeded()
end

function sysCall_suspend()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    sim.setModelProperty(model,0)
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    local c=readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],4)~=0)
    if not show then
        sim.setModelProperty(model,sim.modelproperty_not_visible)
    end
    simJustStarted=true
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