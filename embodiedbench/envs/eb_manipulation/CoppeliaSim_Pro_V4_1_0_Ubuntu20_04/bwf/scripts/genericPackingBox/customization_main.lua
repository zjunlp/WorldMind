function model.setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,massPerVolume,inertiaFact)
    local transf=sim.getObjectMatrix(h,-1)
    local mass=sizeX*sizeY*sizeZ*massPerVolume
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function model.setShapeActive(h,active)
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

function model.setMass(m)
    local currentMass=0
    local objects={model.handle}
    while #objects>0 do
        local handle=objects[#objects]
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

    local objects={model.handle}
    while #objects>0 do
        local handle=objects[#objects]
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

function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(model.specHandles.sides,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.sides,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(model.specHandles.bb,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.bb,nil,sim.colorcomponent_specular,{spec,spec,spec})
    for i=1,4,1 do
        sim.setShapeColor(model.specHandles.lids[i],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
        sim.setShapeColor(model.specHandles.lids[i],nil,sim.colorcomponent_specular,{spec,spec,spec})
    end
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model.handle,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function model.updateModel()
    local c=model.readInfo()
    local w=c.partSpecific['width']
    local l=c.partSpecific['length']
    local h=c.partSpecific['height']
    local th=c.partSpecific['thickness']
    local bitC=c.partSpecific['bitCoded']
    local h2=h-th
    local defMassPerVolume=200
    local inertiaFactor=c.partSpecific['inertiaFactor']
    local maxTorque=c.partSpecific['lidTorque']
    local springK=c.partSpecific['lidSpring']
    local springC=c.partSpecific['lidDamping']
    model.setObjectSize(model.handle,w,l,th)
    model.setCuboidMassAndInertia(model.handle,w,l,th,defMassPerVolume,inertiaFactor)
    sim.removeObject(model.specHandles.sides)
    local p={}
    p[1]=sim.copyPasteObjects({model.specHandles.bb},0)[1]
    model.setObjectSize(p[1],th,l,h)
    model.setCuboidMassAndInertia(p[1],th,l,h,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(p[1],model.handle,{(w-th)/2,0,(h+th)/2})
    p[2]=sim.copyPasteObjects({p[1]},0)[1]
    sim.setObjectPosition(p[2],model.handle,{(-w+th)/2,0,(h+th)/2})
    p[3]=sim.copyPasteObjects({model.specHandles.bb},0)[1]
    model.setObjectSize(p[3],w-th*2,th,h2)
    model.setCuboidMassAndInertia(p[3],w-th*2,th,h2,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(p[3],model.handle,{0,(l-th)/2,(h2+th)/2})
    p[4]=sim.copyPasteObjects({p[3]},0)[1]
    sim.setObjectPosition(p[4],model.handle,{0,(-l+th)/2,(h2+th)/2})

    local textureId=sim.getShapeTextureId(model.specHandles.bb)
    for i=1,4,1 do
        if sim.boolAnd32(bitC,4)>0 then
            sim.setShapeTexture(p[i],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        else
            sim.setShapeTexture(p[i],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        end
    end


    model.specHandles.sides=sim.groupShapes(p)
    model.setShapeActive(model.specHandles.sides,true)
    sim.setObjectInt32Parameter(model.specHandles.sides,sim.shapeintparam_respondable_mask,65535-1)
    sim.setObjectParent(model.specHandles.sides,model.specHandles.sideConnection,true)
    sim.setObjectPosition(model.specHandles.joints[1],model.handle,{w/2,0,h+th/2})
    sim.setObjectPosition(model.specHandles.joints[2],model.handle,{-w/2,0,h+th/2})
    sim.setObjectPosition(model.specHandles.joints[3],model.handle,{0,l/2,h2+th/2})
    sim.setObjectPosition(model.specHandles.joints[4],model.handle,{0,-l/2,h2+th/2})
    
    for i=1,4,1 do
        sim.setJointForce(model.specHandles.joints[i],maxTorque)
        sim.setObjectFloatParameter(model.specHandles.joints[i],sim.jointfloatparam_kc_k,springK)
        sim.setObjectFloatParameter(model.specHandles.joints[i],sim.jointfloatparam_kc_c,springC)
    end

    local lidL=c.partSpecific['closePartALength']*w
    local lidW=c.partSpecific['closePartAWidth']*l
    model.setObjectSize(model.specHandles.lids[1],th,lidW,lidL)
    model.setCuboidMassAndInertia(model.specHandles.lids[1],th,lidW,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(model.specHandles.lids[1],model.specHandles.joints[1],{0,lidL*0.5,0})
    model.setObjectSize(model.specHandles.lids[2],th,lidW,lidL)
    model.setCuboidMassAndInertia(model.specHandles.lids[2],th,lidW,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(model.specHandles.lids[2],model.specHandles.joints[2],{0,lidL*0.5,0})

    lidL=c.partSpecific['closePartBLength']*l
    lidW=c.partSpecific['closePartBWidth']*w
    model.setObjectSize(model.specHandles.lids[3],lidW,th,lidL)
    model.setCuboidMassAndInertia(model.specHandles.lids[3],lidW,th,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(model.specHandles.lids[3],model.specHandles.joints[3],{0,lidL*0.5,0})
    model.setObjectSize(model.specHandles.lids[4],lidW,th,lidL)
    model.setCuboidMassAndInertia(model.specHandles.lids[4],lidW,th,lidL,defMassPerVolume,inertiaFactor)
    sim.setObjectPosition(model.specHandles.lids[4],model.specHandles.joints[4],{0,lidL*0.5,0})

    model.setShapeActive(model.specHandles.lids[1],sim.boolAnd32(bitC,1)>0)
    model.setShapeActive(model.specHandles.lids[2],sim.boolAnd32(bitC,1)>0)
    model.setShapeActive(model.specHandles.lids[3],sim.boolAnd32(bitC,2)>0)
    model.setShapeActive(model.specHandles.lids[4],sim.boolAnd32(bitC,2)>0)
    sim.setObjectInt32Parameter(model.specHandles.lids[1],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(model.specHandles.lids[2],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(model.specHandles.lids[3],sim.shapeintparam_respondable_mask,65535-254)
    sim.setObjectInt32Parameter(model.specHandles.lids[4],sim.shapeintparam_respondable_mask,65535-254)

    if sim.boolAnd32(bitC,4)>0 then
        -- textured
        sim.setShapeTexture(model.handle,textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[1],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[2],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[3],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[4],textureId,sim.texturemap_cube,4+8,{0.3,0.3})
    else
        -- without texture
        sim.setShapeTexture(model.handle,-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[1],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[2],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[3],-1,sim.texturemap_cube,4+8,{0.3,0.3})
        sim.setShapeTexture(model.specHandles.lids[4],-1,sim.texturemap_cube,4+8,{0.3,0.3})
    end

    model.setMass(c.partSpecific['mass'])
end


function sysCall_cleanup_specific()
    local c=model.readInfo()
    if sim.boolAnd32(c.partSpecific['bitCoded'],1)==0 then
        sim.removeObject(model.specHandles.lids[1]) 
        sim.removeObject(model.specHandles.lids[2]) 
        sim.removeObject(model.specHandles.joints[1]) 
        sim.removeObject(model.specHandles.joints[2]) 
    end
    if sim.boolAnd32(c.partSpecific['bitCoded'],2)==0 then
        sim.removeObject(model.specHandles.lids[3]) 
        sim.removeObject(model.specHandles.lids[4]) 
        sim.removeObject(model.specHandles.joints[3]) 
        sim.removeObject(model.specHandles.joints[4]) 
    end
    sim.removeObject(model.specHandles.bb)
end

