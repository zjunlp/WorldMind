function model.setSphereSize(h,diameter)
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

function model.setSphereMassAndInertia(h,diameter,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local I=2*mass*inertiaFact*(diameter*0.5)*(diameter*0.5)/5
    local inertia={I,0,0,0,I,0,0,0,I}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    for i=1,3,1 do
        sim.setShapeColor(model.specHandles.auxSpheres[i],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    end
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function model.updateModel()
    local c=model.readInfo()
    local d=c.partSpecific['diameter']
    local count=c.partSpecific['count']
    local offset=c.partSpecific['offset']*d
    local mass=c.partSpecific['mass']
    model.setSphereSize(model.handle,d)
    model.setSphereMassAndInertia(model.handle,d,mass/count,2)
    for i=1,3,1 do
        model.setSphereSize(model.specHandles.auxSpheres[i],d)
        model.setSphereMassAndInertia(model.specHandles.auxSpheres[i],d,mass/count,2)
        sim.setObjectPosition(model.specHandles.auxSpheres[i],model.handle,{0,0,0})
    end
    if count>=2 then
        sim.setObjectPosition(model.specHandles.auxSpheres[1],model.handle,{offset,0,0})
    end
    if count>=3 then
        sim.setObjectPosition(model.specHandles.auxSpheres[2],model.handle,{offset*0.5,0.866*offset,0})
    end
    if count==4 then
        sim.setObjectPosition(model.specHandles.auxSpheres[3],model.handle,{offset*0.5,0.288*offset,0.81*offset})
    end
end

function sysCall_cleanup_specific()
    local c=model.readInfo()
    local fs=sim.getObjectsInTree(model.handle,sim.object_forcesensor_type,1+2)
    for i=1,#fs,1 do
        sim.removeObject(fs[i])
    end
    if c.partSpecific['count']<4 then
        sim.removeObject(model.specHandles.auxSpheres[3])
    end
    if c.partSpecific['count']<3 then
        sim.removeObject(model.specHandles.auxSpheres[2])
    end
    if c.partSpecific['count']<2 then
        sim.removeObject(model.specHandles.auxSpheres[1])
    else
        local dummy=sim.createDummy(0.01)
        sim.setObjectOrientation(dummy,model.handle,{0,0,0})
        local oss=sim.getObjectsInTree(model.handle,sim.object_shape_type,1)
        oss[#oss+1]=model.handle
        local r=sim.groupShapes(oss)
        sim.reorientShapeBoundingBox(r,dummy)
        sim.removeObject(dummy)
    end
end

