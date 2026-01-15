function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function setSphereSize(h,diameter)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(h,diameter/sx,diameter/sy,diameter/sz)
end

function setSphereMassAndInertia(h,diameter,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local I=2*mass*inertiaFact*(diameter*0.5)*(diameter*0.5)/5
    local inertia={I,0,0,0,I,0,0,0,I}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='sphere'
    end
    if not info['diameter'] then
        info['diameter']=0.1
    end
    if not info['count'] then
        info['count']=1
    end
    if not info['offset'] then
        info['offset']=1
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- all free
    end
    if not info['mass'] then
        info['mass']=0.5
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_SPHERE_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_SPHERE_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_SPHERE_INFO','')
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
    for i=1,3,1 do
        sim.setShapeColor(auxSpheres[i],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    end
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function updateModel()
    local c=readInfo()
    local d=c['diameter']
    local count=c['count']
    local offset=c['offset']*d
    local mass=c['mass']
    setSphereSize(model,d)
    setSphereMassAndInertia(model,d,mass/count,2)
    for i=1,3,1 do
        setSphereSize(auxSpheres[i],d)
        setSphereMassAndInertia(auxSpheres[i],d,mass/count,2)
        sim.setObjectPosition(auxSpheres[i],model,{0,0,0})
    end
    if count>=2 then
        sim.setObjectPosition(auxSpheres[1],model,{offset,0,0})
    end
    if count>=3 then
        sim.setObjectPosition(auxSpheres[2],model,{offset*0.5,0.866*offset,0})
    end
    if count==4 then
        sim.setObjectPosition(auxSpheres[3],model,{offset*0.5,0.288*offset,0.81*offset})
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['diameter']/0.001),true)
        simUI.setSliderValue(ui,3,config['count'],true)
        simUI.setEditValue(ui,2,simBWF.format("%.2f",config['offset']),true)
        simUI.setEditValue(ui,20,simBWF.format("%.2f",config['mass']),true)
        local red,green,blue=getColor()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function diameterChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>2 then v=2 end
        if v~=c['diameter'] then
            simBWF.markUndoPoint()
            c['diameter']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end


function countChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>4 then v=4 end
        if v~=c['count'] then
            simBWF.markUndoPoint()
            c['count']=v
            if v>1 and c['offset']==0 then
                c['offset']=0.1
            end
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function offsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>1 then v=1 end
        if v~=c['offset'] then
            simBWF.markUndoPoint()
            c['offset']=v
            if v==0 and c['count']>1 then
                c['count']=1
            end
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
    if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Finalizing the sphere',"By closing this customization dialog you won't be able to customize the sphere anymore. Do you want to proceed?") then
        finalizeModel=true
        sim.removeScript(sim.handle_self)
    end
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General" layout="form">
                <label text="Diameter (mm)"/>
                <edit on-editing-finished="diameterChange_callback" id="1"/>

                <label text="Mass (Kg)"/>
                <edit on-editing-finished="massChange_callback" id="20"/>


                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
            </tab>
            <tab title="Roundness" layout="form">
                <label text="multi-sphere count"/>
                <hslider minimum="1" maximum="4" on-change="countChange_callback" id="3"/>

                <label text="offset (in % of diameter)"/>
                <edit on-editing-finished="offsetChange_callback" id="2"/>
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
    auxSpheres={}
    for i=1,3,1 do
        auxSpheres[i]=sim.getObjectHandle('genericSphere_auxSphere'..i)
    end
    local data=readPartInfo()
    if data['name']=='<partName>' then
        data['name']='SPHERE'
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
        -- This means the sphere is part of the part repository or that we want to finalize the model (i.e. won't be customizable anymore)
        local c=readInfo()
        sim.writeCustomDataBlock(model,'XYZ_SPHERE_INFO','')
        local fs=sim.getObjectsInTree(model,sim.object_forcesensor_type,1+2)
        for i=1,#fs,1 do
            sim.removeObject(fs[i])
        end
        if c['count']<4 then
            sim.removeObject(auxSpheres[3])
        end
        if c['count']<3 then
            sim.removeObject(auxSpheres[2])
        end
        if c['count']<2 then
            sim.removeObject(auxSpheres[1])
        else
            local dummy=sim.createDummy(0.01)
            sim.setObjectOrientation(dummy,model,{0,0,0})
            local oss=sim.getObjectsInTree(model,sim.object_shape_type,1)
            oss[#oss+1]=model
            local r=sim.groupShapes(oss)
            sim.reorientShapeBoundingBox(r,dummy)
            sim.removeObject(dummy)
        end
    end
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end
