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
        info['subtype']='sink'
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
    if not info['status'] then
        info['status']='operational'
    end
    if not info['bitCoded'] then
        info['bitCoded']=1 -- 1=visibleDuringSimulation, 128=show statistics
    end
    if not info['destroyedCnt'] then
        info['destroyedCnt']=0
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTSINK)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.PARTSINK,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.PARTSINK,'')
    end
end

function setSizes()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    setObjectSize(model,w,l,h)
    setObjectSize(sensor,w,l,h)
    local p=sim.getObjectPosition(model,-1)
    --sim.setObjectPosition(model,-1,{p[1],p[2],h*0.5})
    sim.setObjectPosition(sensor,model,{0,0,-h*0.5+0.001})
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,22,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,4,simBWF.getCheckboxValFromBool(config['status']~='disabled'),true)
        simUI.setCheckboxValue(ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],128)~=0),true)
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

function showStatisticsClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function enabled_callback(ui,id,newVal)
    local c=readInfo()
    if newVal==0 then
        c['status']='disabled'
    else
        c['status']='operational'
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
        if v<0.2 then v=0.2 end
        if v>5 then v=5 end
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
        if v<0.2 then v=0.2 end
        if v>5 then v=5 end
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
        if v<0.01 then v=0.01 end
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

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="20"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="21"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="22"/>

                <label text="Enabled"/>
                <checkbox text="" on-change="enabled_callback" id="4" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="hidden_callback" id="3" />

                <label text="Show statistics"/>
                 <checkbox text="" checked="false" on-change="showStatisticsClick_callback" id="6"/>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,false,nil,false,false,false,'layout="form"')

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
    writeInfo(_info)
    sensor=sim.getObjectHandle('genericPartSink_sensor')
    
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
    if wasDisabledBeforeSimulation then
        c['status']='disabled'
    else
        c['status']='operational'
    end
   c['destroyedCnt']=0
    writeInfo(c)
    sim.setModelProperty(model,0)
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    removeDlg()
    local c=readInfo()
    wasDisabledBeforeSimulation=c['status']=='disabled'
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