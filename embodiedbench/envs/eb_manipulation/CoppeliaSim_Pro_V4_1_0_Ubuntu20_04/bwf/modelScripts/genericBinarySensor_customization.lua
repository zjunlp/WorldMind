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
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='binary'
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
        info['bitCoded']=1+4 -- 1=enabled, 2=detect parts only, 4=visible during simulation, 8=rising edge, 16=falling edge, 32=showStatistics
    end
    if not info['countForTrigger'] then
        info['countForTrigger']=1
    end
    if not info['delayForTrigger'] then
        info['delayForTrigger']=0
    end
    if not info['statText'] then
        info['statText']='Produced parts: '
    end
    if not info['detectionState'] then
        info['detectionState']=0
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO','')
    end
end

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    setObjectSize(model,w,l,h)
    local r,mmin=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    r,mmin=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_min_x)
    r,mmax=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_max_x)
    sx=mmax-mmin
    r,mmin=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_min_y)
    r,mmax=sim.getObjectFloatParameter(sensor,sim.objfloatparam_objbbox_max_y)
    sy=mmax-mmin
    sim.scaleObject(sensor,w/sx,l/sy,h/sz)
    sim.setObjectPosition(sensor,model,{0,0,-h*0.5})
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
        simUI.setCheckboxValue(ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],32)~=0),true)
        simUI.setEditValue(ui,7,config['statText'],true)

        simUI.setRadiobuttonValue(ui,10,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],2)==0),true)
        simUI.setRadiobuttonValue(ui,11,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setRadiobuttonValue(ui,100,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],8+16)==8),true)
        simUI.setRadiobuttonValue(ui,101,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],8+16)==16),true)
        simUI.setRadiobuttonValue(ui,102,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],8+16)==8+16),true)
        simUI.setEditValue(ui,103,simBWF.format("%.0f",config['countForTrigger']),true)
        simUI.setEditValue(ui,104,simBWF.format("%.2f",config['delayForTrigger']),true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local config=readInfo()
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,1,enabled,true)
        simUI.setEnabled(ui,2,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
        simUI.setEnabled(ui,6,enabled,true)
        simUI.setEnabled(ui,7,enabled and sim.boolAnd32(config['bitCoded'],32)~=0,true)
        simUI.setEnabled(ui,10,enabled,true)
        simUI.setEnabled(ui,11,enabled,true)
        simUI.setEnabled(ui,100,enabled,true)
        simUI.setEnabled(ui,101,enabled,true)
        simUI.setEnabled(ui,102,enabled,true)
        simUI.setEnabled(ui,103,enabled,true)
        simUI.setEnabled(ui,104,enabled,true)
    end
end

function enabled_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function visible_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal~=0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function showStats_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],32)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-32
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
    updateEnabledDisabledItemsDlg()
end

function anythingDetectClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    c['bitCoded']=c['bitCoded']-2
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function partsDetectClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function riseClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8+16)
    c['bitCoded']=c['bitCoded']-16
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function fallClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8+16)
    c['bitCoded']=c['bitCoded']-8
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function riseAndFallClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8+16)
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function showStatsChange_callback(ui,id,v)
    local c=readInfo()
    if v~=c['statText'] then
        simBWF.markUndoPoint()
        c['statText']=v
        writeInfo(c)
    end
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

function countForTriggerChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v~=c['countForTrigger'] then
            simBWF.markUndoPoint()
            c['countForTrigger']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function delayForTriggerChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>120 then v=120 end
        if v~=c['delayForTrigger'] then
            simBWF.markUndoPoint()
            c['delayForTrigger']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml=[[
        <tabs id="77">
            <tab title="General properties" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="2"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="3"/>

                <label text="Enabled"/>
                <checkbox text="" on-change="enabled_callback" id="4" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="visible_callback" id="5" />

                <checkbox text="Show statistics, text:" on-change="showStats_callback" id="6" />
                <edit on-editing-finished="showStatsChange_callback" id="7"/>
            </tab>
            <tab title="Trigger conditions">
                <group layout="form" flat="true">
                <label text="Detect"/>
                <radiobutton text="anything" on-click="anythingDetectClick_callback" auto-exclusive="false" id="10" />

                <label text=""/>
                <radiobutton text="Detect parts only" on-click="partsDetectClick_callback" auto-exclusive="false" id="11" />
                </group>
                

                
                <group layout="form" flat="true">
                <label text="Count"/>
                <radiobutton text="rising edge" on-click="riseClick_callback" auto-exclusive="false" id="100" />

                <label text=""/>
                <radiobutton text="falling edge" on-click="fallClick_callback" auto-exclusive="false" id="101" />

                <label text=""/>
                <radiobutton text="rising and falling edge" on-click="riseAndFallClick_callback" auto-exclusive="false" id="102" />
                </group>

                

                
                <group layout="form" flat="true">
                <label text="Increments for trigger"/>
                <edit on-editing-finished="countForTriggerChange_callback" id="103"/>

                <label text="Delay for trigger (s)"/>
                <edit on-editing-finished="delayForTriggerChange_callback" id="104"/>
                </group>
            </tab>
        </tabs>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

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
    sensor=sim.getObjectHandle('genericBinarySensor_sensor')
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    
    updatePluginRepresentation()
    previousDlgPos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
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
    local c=readInfo()
    c['detectionState']=0
    writeInfo(c)
    sim.setModelProperty(model,0)
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    local c=readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],4)>0)
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
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos)
end