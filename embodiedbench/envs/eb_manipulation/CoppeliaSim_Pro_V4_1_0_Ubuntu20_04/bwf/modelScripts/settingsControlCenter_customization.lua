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
        info['subtype']='control'
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- 0-3=0:no override, 1:force aux. vis. items not show, 2:force show, 4=override part sleep time, 8=display warning with overlapping models, 16=do not open property dialogs
    end
    if not info['deactivationTime'] then
        info['deactivationTime']=9999
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDOVERRIDE)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDOVERRIDE,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDOVERRIDE,'')
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setCheckboxValue(ui,1,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['deactivationTime']),true)
        simUI.setEnabled(ui,2,sim.boolAnd32(config['bitCoded'],4)~=0,true)
        simUI.setCheckboxValue(ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(ui,7,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)

        simUI.setRadiobuttonValue(ui,3,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],3)==0),true)
        simUI.setRadiobuttonValue(ui,4,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],3)==1),true)
        simUI.setRadiobuttonValue(ui,5,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],3)==2),true)
        local sel=simBWF.getSelectedEditWidget(ui)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function visualArtifactsNoOverrideClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1+2)
    c['bitCoded']=c['bitCoded']-3
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function visualArtifactsForceHideClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1+2)
    c['bitCoded']=c['bitCoded']-2
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function visualArtifactsForceShowClick_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1+2)
    c['bitCoded']=c['bitCoded']-1
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function deactivationTimeToggle_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolXor32(c['bitCoded'],4)
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function overlappingToggle_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolXor32(c['bitCoded'],8)
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function disableDialogs_callback(ui,id)
    local c=readInfo()
    c['bitCoded']=sim.boolXor32(c['bitCoded'],16)
    simBWF.markUndoPoint()
    writeInfo(c)
    setDlgItemContent()
end

function deactivationTimeChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<1 then v=1 end
        if v>1000000 then v=1000000 end
        if v~=c['deactivationTime'] then
            simBWF.markUndoPoint()
            c['deactivationTime']=v
            writeInfo(c)
        end
    end
    setDlgItemContent()
end

function createDlg()
    if not ui then
        local xml =[[

                <checkbox text="Override part deactivation time (s)" on-change="deactivationTimeToggle_callback" id="1" />
                <edit on-editing-finished="deactivationTimeChange_callback" id="2"/>

                <label text="Display warning with overlapping models"/>
                <checkbox text="" on-change="overlappingToggle_callback" id="6" />

                <label text="Disable model property dialogs"/>
                <checkbox text="" on-change="disableDialogs_callback" id="7" />

                <label text="Do not override aux. visualization items"/>
                <radiobutton text="" on-click="visualArtifactsNoOverrideClick_callback" id="3" />

                <label text="Force aux. visualization items to hide"/>
                <radiobutton text="" on-click="visualArtifactsForceHideClick_callback" id="4" />
 
                <label text="Force aux. visualization items to show"/>
                <radiobutton text="" on-click="visualArtifactsForceShowClick_callback" id="5" />
        ]]
        ui=simBWF.createCustomUi(xml,'Global Model Settings',previousDlgPos,false,nil,false,false,false,'layout="form"')
        setDlgItemContent()
--        updateEnabledDisabledItemsDlg()
--        simUI.setCurrentTab(ui,77,dlgMainTabIndex,true)
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
--        dlgMainTabIndex=simUI.getCurrentTab(ui,77)
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
    
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
    local objs=sim.getObjectsWithTag(simBWF.modelTags.OLDOVERRIDE,true)
    if #objs>1 then
        sim.removeObject(model)
        sim.removeObjectFromSelection(sim.handle_all)
        objs=sim.getObjectsWithTag(simBWF.modelTags.OLDOVERRIDE,true)
        sim.addObjectToSelection(sim.handle_single,objs[1])
    else
        updatePluginRepresentation()
    end
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
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,1)
end

function sysCall_beforeSimulation()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,0)
    removeDlg()
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
    if sim.isHandleValid(model)==1 then
        -- the associated object might already have been destroyed
        simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
    end
end