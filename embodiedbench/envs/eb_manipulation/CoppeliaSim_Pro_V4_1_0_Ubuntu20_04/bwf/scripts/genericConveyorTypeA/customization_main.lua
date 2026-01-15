function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.specHandles.middleParts[2],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.middleParts[2],nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(model.specHandles.endParts[1],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.endParts[1],nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(model.specHandles.endParts[2],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.endParts[2],nil,sim.colorcomponent_specular,{spec,spec,spec})
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.specHandles.middleParts[2],nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model.specHandles.middleParts[2],nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function model.updateConveyor()
    local conf=model.readInfo()
    local length=conf['length']
    local width=conf['width']
    local height=conf['height']
    local borderHeight=conf.conveyorSpecific.borderHeight
    local bitCoded=conf.conveyorSpecific.bitCoded
    local wt=conf.conveyorSpecific.wallThickness
    local re=sim.boolAnd32(bitCoded,16)==0

    sim.setObjectPosition(model.specHandles.rotJoints[1],model.handle,{-length*0.5,0,-height*0.5})
    sim.setObjectPosition(model.specHandles.rotJoints[2],model.handle,{length*0.5,0,-height*0.5})

    model.setShapeSize(model.specHandles.middleParts[1],width,length,height)
    model.setShapeSize(model.specHandles.middleParts[2],width,length,0.001)
    model.setShapeSize(model.specHandles.middleParts[3],width,length,height)
    sim.setObjectPosition(model.specHandles.middleParts[1],model.handle,{0,0,-height*0.5})
    sim.setObjectPosition(model.specHandles.middleParts[2],model.handle,{0,0,-0.0005})
    sim.setObjectPosition(model.specHandles.middleParts[3],model.handle,{0,0,-height*0.5})

    model.setShapeSize(model.specHandles.endParts[1],width,0.083148*height/0.2,0.044443*height/0.2)
    sim.setObjectPosition(model.specHandles.endParts[1],model.handle,{-length*0.5-0.5*0.083148*height/0.2,0,-0.044443*height*0.5/0.2})

    model.setShapeSize(model.specHandles.endParts[2],width,0.083148*height/0.2,0.044443*height/0.2)
    sim.setObjectPosition(model.specHandles.endParts[2],model.handle,{length*0.5+0.5*0.083148*height/0.2,0,-0.044443*height*0.5/0.2})

    model.setShapeSize(model.specHandles.endParts[3],width,height*0.5,height)
    sim.setObjectPosition(model.specHandles.endParts[3],model.handle,{-length*0.5-0.25*height,0,-height*0.5})

    model.setShapeSize(model.specHandles.endParts[4],width,height*0.5,height)
    sim.setObjectPosition(model.specHandles.endParts[4],model.handle,{length*0.5+0.25*height,0,-height*0.5})

    for i=5,6,1 do
        model.setShapeSize(model.specHandles.endParts[i],height,height,width)
    end

    model.setShapeSize(model.specHandles.sides[1],wt,length,height+2*borderHeight)
    model.setShapeSize(model.specHandles.sides[2],wt,length,height+2*borderHeight)
    model.setShapeSize(model.specHandles.sides[4],width+2*wt,height*0.5+1*borderHeight,height+2*borderHeight)
    model.setShapeSize(model.specHandles.sides[3],width+2*wt,height*0.5+1*borderHeight,height+2*borderHeight)
    sim.setObjectPosition(model.specHandles.sides[4],model.handle,{-(length+height*0.5+borderHeight)*0.5,0,-height*0.5})
    sim.setObjectPosition(model.specHandles.sides[3],model.handle,{(length+height*0.5+borderHeight)*0.5,0,-height*0.5})
    sim.setObjectPosition(model.specHandles.sides[1],model.handle,{0,(width+wt)*0.5,-height*0.5})
    sim.setObjectPosition(model.specHandles.sides[2],model.handle,{0,-(width+wt)*0.5,-height*0.5})

    if re then
        sim.setObjectInt32Parameter(model.specHandles.endParts[1],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(model.specHandles.endParts[2],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(model.specHandles.endParts[3],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(model.specHandles.endParts[4],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(model.specHandles.endParts[5],sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.specHandles.endParts[6],sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.specHandles.endParts[5],sim.shapeintparam_respondable,1)
        sim.setObjectInt32Parameter(model.specHandles.endParts[6],sim.shapeintparam_respondable,1)
    else
        sim.setObjectInt32Parameter(model.specHandles.endParts[1],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[2],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[3],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[4],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[5],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[6],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[5],sim.shapeintparam_respondable,0)
        sim.setObjectInt32Parameter(model.specHandles.endParts[6],sim.shapeintparam_respondable,0)
    end

    if sim.boolAnd32(bitCoded,1)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.sides[1],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.sides[1],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.sides[1],0)
        sim.setObjectProperty(model.specHandles.sides[1],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.sides[1],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.sides[1],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.sides[1],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.sides[1],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,2)~=0 then
        sim.setObjectInt32Parameter(model.specHandles.sides[2],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.sides[2],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.sides[2],0)
        sim.setObjectProperty(model.specHandles.sides[2],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.sides[2],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.sides[2],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.sides[2],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.sides[2],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,4)~=0 or (not re) then
        sim.setObjectInt32Parameter(model.specHandles.sides[3],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.sides[3],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.sides[3],0)
        sim.setObjectProperty(model.specHandles.sides[3],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.sides[3],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.sides[3],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.sides[3],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.sides[3],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,8)~=0 or (not re) then
        sim.setObjectInt32Parameter(model.specHandles.sides[4],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.specHandles.sides[4],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(model.specHandles.sides[4],0)
        sim.setObjectProperty(model.specHandles.sides[4],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(model.specHandles.sides[4],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(model.specHandles.sides[4],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(model.specHandles.sides[4],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(model.specHandles.sides[4],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end

    if sim.boolAnd32(bitCoded,32)==0 then
        local textureID=sim.getShapeTextureId(model.specHandles.textureHolder)
        sim.setShapeTexture(model.specHandles.middleParts[2],textureID,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(model.specHandles.endParts[1],textureID,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(model.specHandles.endParts[2],textureID,sim.texturemap_plane,12,{0.04,0.04})
    else
        sim.setShapeTexture(model.specHandles.middleParts[2],-1,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(model.specHandles.endParts[1],-1,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(model.specHandles.endParts[2],-1,sim.texturemap_plane,12,{0.04,0.04})
    end
end
