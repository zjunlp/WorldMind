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

function setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,massPerVolume,inertiaFact)
    local transf=sim.getObjectMatrix(h,-1)
    local mass=sizeX*sizeY*sizeZ*massPerVolume
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function setShapeActive(h,active)
    if active then
        sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1+256) -- make it visible
        sim.setObjectSpecialProperty(h,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable) -- make it collidable, measurable, detectable, etc.
        sim.setObjectInt32Parameter(h,sim.shapeintparam_static,0) -- make it non-static
        sim.setObjectInt32Parameter(h,sim.shapeintparam_respondable,1) -- make it respondable
        local p=sim.boolOr32(sim.getObjectProperty(h),sim.objectproperty_dontshowasinsidemodel)
        sim.setObjectProperty(h,p-sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,0) -- make it invisible
        sim.setObjectSpecialProperty(h,0) -- make it not collidable, measurable, detectable, etc.
        sim.setObjectInt32Parameter(h,sim.shapeintparam_static,1) -- make it static
        sim.setObjectInt32Parameter(h,sim.shapeintparam_respondable,0) -- make it non-respondable
        local p=sim.boolOr32(sim.getObjectProperty(h),sim.objectproperty_dontshowasinsidemodel)
        sim.setObjectProperty(h,p)
    end
end

setMass=function(m)
    local currentMass=0
    local objects={model}
    while #objects>0 do
        handle=objects[#objects]
        table.remove(objects,#objects)
        local i=0
        while true do
            local h=sim.getObjectChild(handle,i)
            if h>=0 then
                objects[#objects+1]=h
                i=i+1
            else
                break
            end
        end
        if sim.getObjectType(handle)==sim.object_shape_type then
            local r,p=sim.getObjectInt32Parameter(handle,sim.shapeintparam_static)
            if p==0 then
                local m0,i0,com0=sim.getShapeMassAndInertia(handle)
                currentMass=currentMass+m0
            end
        end
    end

    local massScaling=m/currentMass

    local objects={model}
    while #objects>0 do
        handle=objects[#objects]
        table.remove(objects,#objects)
        local i=0
        while true do
            local h=sim.getObjectChild(handle,i)
            if h>=0 then
                objects[#objects+1]=h
                i=i+1
            else
                break
            end
        end
        if sim.getObjectType(handle)==sim.object_shape_type then
            local r,p=sim.getObjectInt32Parameter(handle,sim.shapeintparam_static)
            if p==0 then
                local transf=sim.getObjectMatrix(handle,-1)
                local m0,i0,com0=sim.getShapeMassAndInertia(handle,transf)
                for i=1,9,1 do
                    i0[i]=i0[i]*massScaling
                end
                sim.setShapeMassAndInertia(handle,m0*massScaling,i0,com0,transf)
            end
        end
    end
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='packingBox'
    end
    if not info['width'] then
        info['width']=0.3
    end
    if not info['length'] then
        info['length']=0.4
    end
    if not info['height'] then
        info['height']=0.3
    end
    if not info['thickness'] then
        info['thickness']=0.003
    end
    if not info['bitCoded'] then
        info['bitCoded']=1+2+4 -- 1=partA, 2=partB, 4=textured
    end
    if not info['closePartALength'] then
        info['closePartALength']=0.5
    end
    if not info['closePartAWidth'] then
        info['closePartAWidth']=1
    end
    if not info['closePartBLength'] then
        info['closePartBLength']=0.5
    end
    if not info['closePartBWidth'] then
        info['closePartBWidth']=0.9
    end
    if not info['mass'] then
        info['mass']=0.5
    end
    if not info['inertiaFactor'] then
        info['inertiaFactor']=1
    end
    if not info['lidTorque'] then
        info['lidTorque']=0.1
    end
    if not info['lidSpring'] then
        info['lidSpring']=1
    end
    if not info['lidDamping'] then
        info['lidDamping']=0
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_PACKINGBOX_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_PACKINGBOX_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_PACKINGBOX_INFO','')
    end
end

function setColor(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(sides,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(sides,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(bb,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(bb,nil,sim.colorcomponent_specular,{spec,spec,spec})
    for i=1,4,1 do
        sim.setShapeColor(lids[i],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
        sim.setShapeColor(lids[i],nil,sim.colorcomponent_specular,{spec,spec,spec})
    end
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function updateModel()
    local c=readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    local th=c['thickness']
    local bitC=c['bitCoded']
    local h2=h-th
    local defMassPerVolume=200
    local inertiaFactor=c['inertiaFactor']
    local maxTorque=c['lidTorque']
    local springK=c['lidSpring']
    local springC=c['lidDamping']
    setObjectSize(model,w,l,th)
    setCuboidMassAndInertia(model,w,l,th,defMassPerVolume,inertiaFactor)
    sim.removeObject(sides)
    local p={}
    p[1]=sim.copyPasteObjects({bb},0)[1]
    setObjectSize(p[1],th,l,h)
    setCuboidMassAndInertia(p[1],th,l,h,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(p[1],model,{(w-th)/2,0,(h+th)/2})
    p[2]=sim.copyPasteObjects({p[1]},0)[1]
    sim.setObjectPosition(p[2],model,{(-w+th)/2,0,(h+th)/2})
    p[3]=sim.copyPasteObjects({bb},0)[1]
    setObjectSize(p[3],w-th*2,th,h2)
    setCuboidMassAndInertia(p[3],w-th*2,th,h2,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(p[3],model,{0,(l-th)/2,(h2+th)/2})
    p[4]=sim.copyPasteObjects({p[3]},0)[1]
    sim.setObjectPosition(p[4],model,{0,(-l+th)/2,(h2+th)/2})

    local textureId=sim.getShapeTextureId(bb)
    for i=1,4,1 do
        if sim.boolAnd32(bitC,4)>0 then
            sim.setShapeTexture(p[i],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        else
            sim.setShapeTexture(p[i],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        end
    end


    sides=sim.groupShapes(p)
    setShapeActive(sides,true)
    sim.setObjectInt32Parameter(sides,sim.shapeintparam_respondable_mask,65535-1)
    sim.setObjectParent(sides,sideConnection,true)
    sim.setObjectPosition(joints[1],model,{w/2,0,h+th/2})
    sim.setObjectPosition(joints[2],model,{-w/2,0,h+th/2})
    sim.setObjectPosition(joints[3],model,{0,l/2,h2+th/2})
    sim.setObjectPosition(joints[4],model,{0,-l/2,h2+th/2})
    
    for i=1,4,1 do
        sim.setJointForce(joints[i],maxTorque)
        sim.setObjectFloatParameter(joints[i],sim.jointfloatparam_kc_k,springK)
        sim.setObjectFloatParameter(joints[i],sim.jointfloatparam_kc_c,springC)
    end

    local lidL=c['closePartALength']*w
    local lidW=c['closePartAWidth']*l
    setObjectSize(lids[1],th,lidW,lidL)
    setCuboidMassAndInertia(lids[1],th,lidW,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(lids[1],joints[1],{0,lidL*0.5,0})
    setObjectSize(lids[2],th,lidW,lidL)
    setCuboidMassAndInertia(lids[2],th,lidW,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(lids[2],joints[2],{0,lidL*0.5,0})

    lidL=c['closePartBLength']*l
    lidW=c['closePartBWidth']*w
    setObjectSize(lids[3],lidW,th,lidL)
    setCuboidMassAndInertia(lids[3],lidW,th,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(lids[3],joints[3],{0,lidL*0.5,0})
    setObjectSize(lids[4],lidW,th,lidL)
    setCuboidMassAndInertia(lids[4],lidW,th,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(lids[4],joints[4],{0,lidL*0.5,0})

    setShapeActive(lids[1],sim.boolAnd32(bitC,1)>0)
    setShapeActive(lids[2],sim.boolAnd32(bitC,1)>0)
    setShapeActive(lids[3],sim.boolAnd32(bitC,2)>0)
    setShapeActive(lids[4],sim.boolAnd32(bitC,2)>0)
    sim.setObjectInt32Parameter(lids[1],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(lids[2],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(lids[3],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(lids[4],sim.shapeintparam_respondable_mask,65535-254)

    if sim.boolAnd32(bitC,4)>0 then
        -- textured
        sim.setShapeTexture(model,textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[1],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[2],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[3],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[4],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
    else
        -- without texture
        sim.setShapeTexture(model,-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[1],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[2],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[3],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(lids[4],-1,sim.texturemap_cube,4+8,{0.3,0.3})
    end

    setMass(c['mass'])
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['thickness']/0.001),true)

        simUI.setCheckboxValue(ui,10,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setEditValue(ui,11,simBWF.format("%.0f",config['closePartALength']*100),true)
        simUI.setEditValue(ui,12,simBWF.format("%.0f",config['closePartAWidth']*100),true)
        simUI.setCheckboxValue(ui,13,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setEditValue(ui,14,simBWF.format("%.0f",config['closePartBLength']*100),true)
        simUI.setEditValue(ui,15,simBWF.format("%.0f",config['closePartBWidth']*100),true)

        simUI.setCheckboxValue(ui,888,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)

        simUI.setEditValue(ui,20,simBWF.format("%.2f",config['mass']),true)
        simUI.setEditValue(ui,21,simBWF.format("%.2f",config['inertiaFactor']),true)
        simUI.setEditValue(ui,22,simBWF.format("%.2f",config['lidTorque']),true)
        simUI.setEditValue(ui,23,simBWF.format("%.2f",config['lidSpring']),true)
        simUI.setEditValue(ui,24,simBWF.format("%.2f",config['lidDamping']),true)

        local red,green,blue,spec=getColor()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simUI.setSliderValue(ui,33,spec*100,true)

        simUI.setEnabled(ui,11,sim.boolAnd32(config['bitCoded'],1)~=0,true)
        simUI.setEnabled(ui,12,sim.boolAnd32(config['bitCoded'],1)~=0,true)
        simUI.setEnabled(ui,14,sim.boolAnd32(config['bitCoded'],2)~=0,true)
        simUI.setEnabled(ui,15,sim.boolAnd32(config['bitCoded'],2)~=0,true)

        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
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
        if v<0.05 then v=0.05 end
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
        if v<0.05 then v=0.05 end
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

function thicknessChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>0.02 then v=0.02 end
        if v~=c['thickness'] then
            simBWF.markUndoPoint()
            c['thickness']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lidA_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    updateModel()
    setDlgItemContent()
end

function lidB_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    updateModel()
    setDlgItemContent()
end

function lidALengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c['closePartALength'] then
            simBWF.markUndoPoint()
            c['closePartALength']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lidAWidthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c['closePartAWidth'] then
            simBWF.markUndoPoint()
            c['closePartAWidth']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lidBLengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c['closePartBLength'] then
            simBWF.markUndoPoint()
            c['closePartBLength']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lidBWidthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c['closePartBWidth'] then
            simBWF.markUndoPoint()
            c['closePartBWidth']=v
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

function inertiaFactorChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.1 then v=0.1 end
        if v>10 then v=10 end
        if v~=c['inertiaFactor'] then
            simBWF.markUndoPoint()
            c['inertiaFactor']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function torqueChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c['lidTorque'] then
            simBWF.markUndoPoint()
            c['lidTorque']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function springConstantChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>100 then v=100 end
        if v~=c['lidSpring'] then
            simBWF.markUndoPoint()
            c['lidSpring']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function dampingChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>10 then v=10 end
        if v~=c['lidDamping'] then
            simBWF.markUndoPoint()
            c['lidDamping']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
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

function specularChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,b,newVal/100)
end

function texture_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    updateModel()
    setDlgItemContent()
end

function onCloseClicked()
    if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Finalizing the box',"By closing this customization dialog you won't be able to customize the box anymore. Do you want to proceed?") then
        finalizeModel=true
        sim.removeScript(sim.handle_self)
    end
end


function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="2"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="3"/>

                <label text="Thickness (mm)"/>
                <edit on-editing-finished="thicknessChange_callback" id="4"/>

            </tab>
            <tab title="Closing lids">
                <group layout="form" flat="true">
                <checkbox text="Lid A, length (%)" on-change="lidA_callback" id="10" />
                <edit on-editing-finished="lidALengthChange_callback" id="11"/>

                <label text="Lid A, width (%)"/>
                <edit on-editing-finished="lidAWidthChange_callback" id="12"/>
                </group>

                <group layout="form" flat="true">
                <checkbox text="Lid B, length (%)" on-change="lidB_callback" id="13" />
                <edit on-editing-finished="lidBLengthChange_callback" id="14"/>

                <label text="Lid B, width (%)"/>
                <edit on-editing-finished="lidBWidthChange_callback" id="15"/>
                </group>
            </tab>
            <tab title="Colors/Texture" layout="form">
                    <label text="Textured"/>
                    <checkbox text="" on-change="texture_callback" id="888" />
                    
                    <label text="Red"/>
                    <hslider minimum="0" maximum="100" on-change="redChange" id="30"/>
                    <label text="Green"/>
                    <hslider minimum="0" maximum="100" on-change="greenChange" id="31"/>
                    <label text="Blue"/>
                    <hslider minimum="0" maximum="100" on-change="blueChange" id="32"/>
                    <label text="Specular"/>
                    <hslider minimum="0" maximum="100" on-change="specularChange" id="33"/>
            </tab>
            <tab title="More" layout="form">
                <label text="Mass (Kg)"/>
                <edit on-editing-finished="massChange_callback" id="20"/>

                <label text="Inertia adjustment factor"/>
                <edit on-editing-finished="inertiaFactorChange_callback" id="21"/>

                <label text="Lid max. torque"/>
                <edit on-editing-finished="torqueChange_callback" id="22"/>

                <label text="Lid spring constant"/>
                <edit on-editing-finished="springConstantChange_callback" id="23"/>

                <label text="Lid damping"/>
                <edit on-editing-finished="dampingChange_callback" id="24"/>

                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
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
    local data=simBWF.readPartInfoV0(model)
    if data['name']=='<partName>' then
        data['name']='PACKINGBOX'
    end
    simBWF.writePartInfo(model,data)

    bb=sim.getObjectHandle('genericPackingBox_bb')
    sideConnection=sim.getObjectHandle('genericPackingBox_sideConnection')
    sides=sim.getObjectChild(sideConnection,0)
    joints={}
    lids={}
    for i=1,4,1 do
        joints[i]=sim.getObjectHandle('genericPackingBox_j'..i)
        lids[i]=sim.getObjectChild(joints[i],0)
    end

    
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
        sim.writeCustomDataBlock(model,'XYZ_PACKINGBOX_INFO','')
        if sim.boolAnd32(c['bitCoded'],1)==0 then
            sim.removeObject(lids[1]) 
            sim.removeObject(lids[2]) 
            sim.removeObject(joints[1]) 
            sim.removeObject(joints[2]) 
        end
        if sim.boolAnd32(c['bitCoded'],2)==0 then
            sim.removeObject(lids[3]) 
            sim.removeObject(lids[4]) 
            sim.removeObject(joints[3]) 
            sim.removeObject(joints[4]) 
        end
        sim.removeObject(bb)
    end
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end
