function model.setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function model.updateModel()
    local c=model.readInfo()
    local w=c.partSpecific.width
    local l=c.partSpecific.length
    local h=c.partSpecific.height
    local mass=c.partSpecific.mass
    model.setObjectSize(model.handle,w,l,h)
    model.setCuboidMassAndInertia(model.handle,w,l,h,mass)
end

function sysCall_cleanup_specific()
end