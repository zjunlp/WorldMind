function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function ext_getItemData_pricing()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.type='ragnarVision'
    obj.visionType='default'
    obj.brVersion=0
    return obj
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
        info['subtype']='window'
    end
    if not info['width'] then
        info['width']=0.5
    end
    if not info['length'] then
        info['length']=0.5
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['maxLabelAngle'] then
        info['maxLabelAngle']=30*math.pi/180
    end
    if not info['detectionDiameter'] then
        info['detectionDiameter']=0.1
    end
    if not info['detectedItems'] then
        info['detectedItems']={}
    end
    if not info['bitCoded'] then
        info['bitCoded']=1 -- 1=hidden,2=console,4=showPts,8=showLabelPts,16=labelSensTop,32=labelSensSide1,64=labelSensSide2,128=colorLabel,256=showStatistics
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO','')
    end
end

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    local d=c['detectionDiameter']
    setObjectSize(box,w,l,h)
    local r,mmin=sim.getObjectFloatParameter(sensor1,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(sensor1,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    r,mmin=sim.getObjectFloatParameter(sensor1,sim.objfloatparam_objbbox_min_x)
    r,mmax=sim.getObjectFloatParameter(sensor1,sim.objfloatparam_objbbox_max_x)
    sd=mmax-mmin
    sim.scaleObject(sensor1,d/sd,d/sd,(h+0.01)/sz)
    r,mmin=sim.getObjectFloatParameter(sensor2,sim.objfloatparam_objbbox_min_z)
    r,mmax=sim.getObjectFloatParameter(sensor2,sim.objfloatparam_objbbox_max_z)
    sz=mmax-mmin
    sim.scaleObject(sensor2,(h+0.01)/sz,(h+0.01)/sz,(h+0.01)/sz)
    sim.setObjectPosition(box,model,{0,0,h*0.5})
    sim.setObjectPosition(sensor1,model,{0,0,h})
    sim.setObjectPosition(sensor2,model,{0,0,h})
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,22,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,23,simBWF.format("%.0f",config['detectionDiameter']/0.001),true)
        simUI.setEditValue(ui,24,simBWF.format("%.0f",180*config['maxLabelAngle']/math.pi),true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)
        simUI.setCheckboxValue(ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(ui,30,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)
        simUI.setCheckboxValue(ui,31,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],32)~=0),true)
        simUI.setCheckboxValue(ui,32,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],64)~=0),true)
        simUI.setCheckboxValue(ui,33,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],128)~=0),true)
        simUI.setCheckboxValue(ui,34,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],256)~=0),true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,20,enabled,true)
        simUI.setEnabled(ui,21,enabled,true)
        simUI.setEnabled(ui,22,enabled,true)
        simUI.setEnabled(ui,23,enabled,true)
        simUI.setEnabled(ui,24,enabled,true)
        simUI.setEnabled(ui,30,enabled,true)
        simUI.setEnabled(ui,31,enabled,true)
        simUI.setEnabled(ui,32,enabled,true)
        simUI.setEnabled(ui,33,enabled,true)
        simUI.setEnabled(ui,34,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,4,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
        simUI.setEnabled(ui,6,enabled,true)
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
    setDlgItemContent()
end

function showPoints_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function console_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function label_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function labelSensorTop_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function labelSensorSide1_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],32)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-32
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function labelSensorSide2_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function colorLabel_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function showStats_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],256)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-256
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.1 then v=0.1 end
        if v>2 then v=2 end
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
        if v<0.01 then v=0.01 end
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
        if v<0.05 then v=0.05 end
        if v>0.5 then v=0.5 end
        if v~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function diameterChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>0.5 then v=0.5 end
        if v~=c['detectionDiameter'] then
            simBWF.markUndoPoint()
            c['detectionDiameter']=v
            writeInfo(c)
            setSizes()
        end
    end
    setDlgItemContent()
end

function labelAngleChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<1 then v=1 end
        if v>80 then v=80 end
        v=v*math.pi/180
        if v~=c['maxLabelAngle'] then
            simBWF.markUndoPoint()
            c['maxLabelAngle']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General properties" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="20"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="21"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="22"/>

                <label text="Detection diameter (mm)"/>
                <edit on-editing-finished="diameterChange_callback" id="23"/>

                <label text="Max. label detection angle (deg)"/>
                <edit on-editing-finished="labelAngleChange_callback" id="24"/>

                <label text="Label sensor, from top"/>
                <checkbox text="" on-change="labelSensorTop_callback" id="30" />

                <label text="Label sensor, from side 1"/>
                <checkbox text="" on-change="labelSensorSide1_callback" id="31" />

                <label text="Label sensor, from side 2"/>
                <checkbox text="" on-change="labelSensorSide2_callback" id="32" />
            </tab>

            <tab title="More" layout="form">
                <label text="Color detected labels"/>
                <checkbox text="" on-change="colorLabel_callback" id="33" />

                <label text="Show detected labels"/>
                <checkbox text="" on-change="label_callback" id="6" />

                <label text="Visualize detected items"/>
                <checkbox text="" on-change="showPoints_callback" id="5" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="hidden_callback" id="3" />

                <label text="Show debug console"/>
                <checkbox text="" on-change="console_callback" id="4" />

                <label text="Show statistics"/>
                <checkbox text="" on-change="showStats_callback" id="34" />

                <label text="" style="* {margin-left: 190px;}"/>
                <label text="" style="* {margin-left: 190px;}"/>
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
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    box=sim.getObjectHandle('genericDetectionWindow_box')
    sensor1=sim.getObjectHandle('genericDetectionWindow_sensor1')
    sensor2=sim.getObjectHandle('genericDetectionWindow_sensor2')
    
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


function sysCall_afterSimulation()
    local c=readInfo()
    c['detectedItems']={}
    writeInfo(c)
    sim.setModelProperty(model,0)
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    removeDlg()
    local c=readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1)==0)
    if not show then
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