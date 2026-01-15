function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function ext_getItemData_pricing()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.type='ragnarGripper'
    obj.gripperType='default'
    obj.brVersion=0
    return obj
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='gripper'
    end
    if not info['size'] then
        info['size']=0
    end
    if not info['stacking'] then
        info['stacking']=1
    end
    if not info['stackingShift'] then
        info['stackingShift']=0.01
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNARGRIPPER)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.RAGNARGRIPPER,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.RAGNARGRIPPER,'')
    end
end

function sizeChange_callback(ui,id,newVal)
    local c=readInfo()
    local s=0.042624
    if newVal>=0 then
        s=0.042624*(1+newVal/4)
    else
        s=0.021312*(1+(newVal+4)/4)
    end
    local r,mmin=sim.getObjectFloatParameter(shape,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(shape,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(shape,s/sz,s/sz,s/sz)
    sim.setObjectPosition(shape,sim.handle_parent,{0,0,-s*0.021416/0.042624})

    c['size']=newVal
    writeInfo(c)
    simBWF.markUndoPoint()
end


function stackingChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=math.floor(newValue)
        if newValue<1 then newValue=1 end
        if newValue>20 then newValue=20 end
        if newValue~=c['stacking'] then
            c['stacking']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    simUI.setEditValue(ui,1,simBWF.format("%.0f",c['stacking']),true)
end

function stackingShiftChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0 then newValue=0 end
        if newValue>0.1 then newValue=0.1 end
        if newValue~=c['stackingShift'] then
            c['stackingShift']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    simUI.setEditValue(ui,3,simBWF.format("%.0f",c['stackingShift']*1000),true)
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <label text="Size"/>
                <hslider tick-position="above" tick-interval="1" minimum="-4" maximum="4" on-change="sizeChange_callback" id="2"/>

                <label text="Stacking"/>
                <edit on-editing-finished="stackingChange_callback" id="1"/>

                <label text="Stacking shift (mm)"/>
                <edit on-editing-finished="stackingShiftChange_callback" id="3"/>
            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,false,nil,false,false,false,'layout="form"')
        local c=readInfo()
        simUI.setSliderValue(ui,2,c['size'],true)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",c['stacking']),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",c['stackingShift']*1000),true)
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
    shape=sim.getObjectHandle('genericGripper_shape')
    
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

function sysCall_beforeSimulation()
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
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end