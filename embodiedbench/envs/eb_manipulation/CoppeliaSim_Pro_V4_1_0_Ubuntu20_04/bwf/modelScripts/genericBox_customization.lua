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

function setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='box'
    end
    if not info['width'] then
        info['width']=0.3
    end
    if not info['length'] then
        info['length']=0.3
    end
    if not info['height'] then
        info['height']=0.3
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- all free
    end
    if not info['mass'] then
        info['mass']=0.5
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_BOX_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_BOX_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_BOX_INFO','')
    end
end

function readPartInfo()
    local data=simBWF.readPartInfoV0(model)
    return data
end

function writePartInfo(data)
    return simBWF.writePartInfo(model,data)
end

function setColor(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function updateModel()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    local mass=c['mass']
    setObjectSize(model,w,l,h)
    setCuboidMassAndInertia(model,w,l,h,mass)
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,20,simBWF.format("%.2f",config['mass']),true)
        local red,green,blue=getColor()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.005 then v=0.005 end
        if v>2 then v=2 end
        if v~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.005 then v=0.005 end
        if v>2 then v=2 end
        if v~=c['length'] then
            simBWF.markUndoPoint()
            c['length']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.005 then v=0.005 end
        if v>2 then v=2 end
        if v~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function massChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c['mass'] then
            simBWF.markUndoPoint()
            c['mass']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(newVal/100,g,b)
end

function greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(r,newVal/100,b)
end

function blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(r,g,newVal/100)
end


function onCloseClicked()
    if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Finalizing the box',"By closing this customization dialog you won't be able to customize the box anymore. Do you want to proceed?") then
        finalizeModel=true
        sim.removeScript(sim.handle_self)
    end
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml = [[
        <tabs id="77">
            <tab title="General" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="2"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="3"/>

                <label text="Mass (Kg)"/>
                <edit on-editing-finished="massChange_callback" id="20"/>


                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
            </tab>
            <tab title="Colors" layout="form">
                    <label text="Red"/>
                    <hslider minimum="0" maximum="100" on-change="redChange" id="30"/>
                    <label text="Green"/>
                    <hslider minimum="0" maximum="100" on-change="greenChange" id="31"/>
                    <label text="Blue"/>
                    <hslider minimum="0" maximum="100" on-change="blueChange" id="32"/>
            </tab>
        </tabs>
        ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,true,'onCloseClicked'--[[,modal,resizable,activate,additionalUiAttribute--]])

        setDlgItemContent()
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
    local data=readPartInfo()
    if data['name']=='<partName>' then
        data['name']='BOX'
    end
    writePartInfo(data)
    
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
    local repo,modelHolder=simBWF.getPartRepositoryHandles()
    if (repo and (sim.getObjectParent(model)==modelHolder)) or finalizeModel then
        -- This means the box is part of the part repository or that we want to finalize the model (i.e. won't be customizable anymore)
        local c=readInfo()
        sim.writeCustomDataBlock(model,'XYZ_BOX_INFO','')
    end
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end
