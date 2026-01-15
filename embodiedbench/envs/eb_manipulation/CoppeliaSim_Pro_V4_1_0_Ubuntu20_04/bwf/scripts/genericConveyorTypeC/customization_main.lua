function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.specHandles.pad,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.pad,nil,sim.colorcomponent_specular,{spec,spec,spec})
    i=0
    while true do
        local h=sim.getObjectChild(model.specHandles.path,i)
        if h>=0 then
            local ch=sim.getObjectChild(h,0)
            if ch>=0 then
                sim.setShapeColor(ch,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
                sim.setShapeColor(ch,nil,sim.colorcomponent_specular,{spec,spec,spec})
            end
            i=i+1
        else
            break
        end
    end
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.specHandles.pad,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model.specHandles.pad,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function model.getActualPadSpacing()
    local conf=model.readInfo()
    local l=sim.getPathLength(model.specHandles.path)
    local cnt=math.floor(l/conf.conveyorSpecific.padSpacing)+1
    local dx=l/cnt
    return dx
end

function model.updateConveyor()
    local conf=model.readInfo()
    local length=conf['length']
    local width=conf['width']
    local padHeight=conf.conveyorSpecific.padHeight
    local padSpacing=conf.conveyorSpecific.padSpacing
    local bitCoded=conf.conveyorSpecific.bitCoded
    local padThickness=conf.conveyorSpecific.padThickness
    local baseThickness=conf['height']
    local wt=conf.conveyorSpecific.wallThickness
---[[
    model.setShapeSize(model.specHandles.base,width,length,baseThickness)
    model.setShapeSize(model.specHandles.baseBack,baseThickness,baseThickness,width)
    model.setShapeSize(model.specHandles.baseFront,baseThickness,baseThickness,width)
    model.setShapeSize(model.specHandles.leftSide,wt,length,baseThickness+2*(padHeight+wt))
    model.setShapeSize(model.specHandles.rightSide,wt,length,baseThickness+2*(padHeight+wt))
    model.setShapeSize(model.specHandles.backSide,width+2*wt,baseThickness*0.5+1*(padHeight+wt),baseThickness+2*(padHeight+wt))
    model.setShapeSize(model.specHandles.frontSide,width+2*wt,baseThickness*0.5+1*(padHeight+wt),baseThickness+2*(padHeight+wt))
    model.setShapeSize(model.specHandles.pad,width,padThickness,padHeight)
    sim.setObjectPosition(model.specHandles.path,model.handle,{0,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.base,model.handle,{0,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.baseBack,model.handle,{-length*0.5,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.baseFront,model.handle,{length*0.5,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.backSide,model.handle,{-(length+baseThickness*0.5+padHeight+wt)*0.5,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.frontSide,model.handle,{(length+baseThickness*0.5+padHeight+wt)*0.5,0,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.leftSide,model.handle,{0,(width+wt)*0.5,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.rightSide,model.handle,{0,-(width+wt)*0.5,-baseThickness*0.5})
    sim.setObjectPosition(model.specHandles.pad,sim.handle_parent,{0,padHeight*0.5,0})

    if sim.boolAnd32(bitCoded,1)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.leftSide,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.leftSide,sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.leftSide,0)
        sim.setObjectProperty(model.specHandles.leftSide,sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.leftSide,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.leftSide,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.leftSide,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.leftSide,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,2)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.rightSide,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.rightSide,sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.rightSide,0)
        sim.setObjectProperty(model.specHandles.rightSide,sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.rightSide,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.rightSide,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.rightSide,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.rightSide,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,4)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.frontSide,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.frontSide,sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.frontSide,0)
        sim.setObjectProperty(model.specHandles.frontSide,sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.frontSide,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.frontSide,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.frontSide,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.frontSide,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,8)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.backSide,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.backSide,sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.backSide,0)
        sim.setObjectProperty(model.specHandles.backSide,sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.backSide,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.backSide,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.backSide,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.backSide,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end

    while true do
        local h=sim.getObjectChild(model.specHandles.path,0)
        if h>=0 then
            sim.removeObject(h)
        else
            break
        end
    end

    sim.cutPathCtrlPoints(model.specHandles.path,-1,0)
    local pts={}
    for i=0,8,1 do
        pts[i*11+1]=0
        pts[i*11+2]=-length*0.5-baseThickness*0.5*math.sin(i*math.pi/8)
        pts[i*11+3]=-baseThickness*0.5*math.cos(i*math.pi/8)
        pts[i*11+4]=0 --math.pi-i*math.pi/8
        pts[i*11+5]=0
        pts[i*11+6]=0
        pts[i*11+7]=1
        pts[i*11+8]=0
        pts[i*11+9]=3
        pts[i*11+10]=0.5
        pts[i*11+11]=0.5
    end
    for i=0,8,1 do
        pts[(i+9)*11+1]=0
        pts[(i+9)*11+2]=length*0.5+baseThickness*0.5*math.sin(i*math.pi/8)
        pts[(i+9)*11+3]=baseThickness*0.5*math.cos(i*math.pi/8)
        pts[(i+9)*11+4]=0 --i*math.pi/8
        pts[(i+9)*11+5]=0
        pts[(i+9)*11+6]=0
        pts[(i+9)*11+7]=1
        pts[(i+9)*11+8]=0
        pts[(i+9)*11+9]=3
        pts[(i+9)*11+10]=0.5
        pts[(i+9)*11+11]=0.5
    end
    sim.insertPathCtrlPoints(model.specHandles.path,1,0,18,pts)
    local l=sim.getPathLength(model.specHandles.path)
    local cnt=math.floor(l/padSpacing)+1
    local dx=l/cnt
    for i=0,cnt-1,1 do
        local pb=sim.copyPasteObjects({model.specHandles.padBase},0)[1]
        local p=sim.copyPasteObjects({model.specHandles.pad},0)[1]
        sim.setObjectParent(p,pb,true)
        sim.setObjectParent(pb,model.specHandles.path,true)
        sim.setObjectInt32Parameter(p,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(p,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(p,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectFloatParameter(pb,sim.dummyfloatparam_follow_path_offset,i*dx)
    end
--]]
end
