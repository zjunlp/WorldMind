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
        info['subtype']='staticPickWindow'
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
    if not info['bitCoded'] then
        info['bitCoded']=1+4 -- 1=hidden during simulation,4=showPts
    end
    if not info['triggerState'] then
        info['triggerState']=0
    end
    if not info['trackedItemsInWindow'] then
        info['trackedItemsInWindow']={}
    end
    if not info['itemsToRemoveFromTracking'] then
        info['itemsToRemoveFromTracking']={}
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO','')
    end
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

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    setObjectSize(box,w,l,h)
    sim.setObjectPosition(box,model,{0,0,h*0.5})
end

function updateEnabledDisabledItemsDlg()
    if ui then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,20,enabled,true)
        simUI.setEnabled(ui,21,enabled,true)
        simUI.setEnabled(ui,22,enabled,true)
        simUI.setEnabled(ui,23,enabled,true)
        simUI.setEnabled(ui,3,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,22,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)
        updateEnabledDisabledItemsDlg()
        simBWF.setSelectedEditWidget(ui,sel)
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
        if v<0.05 then v=0.05 end
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

function sensorChange_callback(ui,id,newIndex)
    local newLoc=comboSensor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.STATICPICKWINDOW_SENSOR_REF,newLoc)
    simBWF.markUndoPoint()
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="20"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="21"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="22"/>

                <label text="Activation sensor"/>
                <combobox id="23" on-change="sensorChange_callback">
                </combobox>

                 <label text="Hidden during simulation"/>
                <checkbox text="" on-change="hidden_callback" id="3" />

                <label text="Visualize tracked items" />
                <checkbox text="" on-change="showPoints_callback" id="5" />

                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
       ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,false,nil,false,false,false,'layout="form"')

        local c=readInfo()
        local sens=getAvailableSensors()
        comboSensor=simBWF.populateCombobox(ui,23,sens,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.STATICPICKWINDOW_SENSOR_REF)),true,{{simBWF.NONE_TEXT,-1}})

        setDlgItemContent()
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
    -- Following for backward compatibility:
    if _info['sensor'] then
        simBWF.setReferencedObjectHandle(model,simBWF.STATICPICKWINDOW_SENSOR_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['sensor']))
        _info['sensor']=nil
    end
    writeInfo(_info)
    box=sim.getObjectHandle('staticPickWindow_box')
    
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

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(box,sim.objintparam_visibility_layer,1)
    local c=readInfo()
    c['itemsToRemoveFromTracking']={}
    c['trackedItemsInWindow']={}
    writeInfo(c)
    updateEnabledDisabledItemsDlg()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    removeDlg()
    local c=readInfo()
    local show=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1)==0)
    if not show then
        sim.setObjectInt32Parameter(box,sim.objintparam_visibility_layer,256)
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
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos)
end