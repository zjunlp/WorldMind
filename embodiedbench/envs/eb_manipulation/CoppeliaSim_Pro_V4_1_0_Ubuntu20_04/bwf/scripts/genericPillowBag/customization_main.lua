function model.setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end


function model.setColor(red,green,blue,spec)
    sim.setShapeColor(model.specHandles.convex,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
end

function model.getColor()
    local r,rgb=sim.getShapeColor(model.specHandles.convex,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function model.updateModel()
    local c=model.readInfo()
    local partInfo=model.readPartInfo()
    local partLabelC=partInfo['labelData']
    local w=c.partSpecific['width']
    local l=c.partSpecific['length']
    local h=c.partSpecific['height']
    local bitCText=c.partSpecific['bitCoded']
    local bitC=partLabelC['bitCoded']
    local mass=c.partSpecific['mass']
    local boxSize={w,l,h}
    local smallLabelSize=partLabelC['smallLabelSize']
    local largeLabelSize=partLabelC['largeLabelSize']

    partLabelC['boxSize']={w,l,h}
    partInfo['labelData']=partLabelC
    model.writePartInfo(partInfo)

    model.setObjectSize(model.handle,w*0.7145,l*0.7145,h+0.002)
    model.setCuboidMassAndInertia(model.handle,w*0.7145,l*0.7145,h,mass*0.5)

    model.setObjectSize(model.specHandles.convex,w,l,h)
    model.setCuboidMassAndInertia(model.specHandles.convex,w*0.7145,l*0.7145,h,mass*0.5)

    model.setObjectSize(model.specHandles.smallLabel,smallLabelSize[1],smallLabelSize[2])
    -- Scale also the texture:
    sim.setObjectFloatParameter(model.specHandles.smallLabel,sim.shapefloatparam_texture_scaling_y,0.11*smallLabelSize[1]/0.075)
    sim.setObjectFloatParameter(model.specHandles.smallLabel,sim.shapefloatparam_texture_scaling_x,0.11*smallLabelSize[2]/0.0375)
    sim.setObjectFloatParameter(model.specHandles.smallLabel,sim.shapefloatparam_texture_y,0.037*smallLabelSize[2]/0.0375)

    model.setObjectSize(model.specHandles.largeLabel,largeLabelSize[1],largeLabelSize[2])
    -- Scale also the texture:
    sim.setObjectFloatParameter(model.specHandles.largeLabel,sim.shapefloatparam_texture_scaling_y,0.11*largeLabelSize[1]/0.075)
    sim.setObjectFloatParameter(model.specHandles.largeLabel,sim.shapefloatparam_texture_scaling_x,0.11*largeLabelSize[2]/0.1125)

    local textureId=sim.getShapeTextureId(model.specHandles.texture)

    if sim.boolAnd32(bitCText,4)>0 then
        -- textured
        sim.setShapeTexture(model.specHandles.convex,textureId,sim.texturemap_plane,4+8,{0.3,0.3})
    else
        -- without texture
        sim.setShapeTexture(model.specHandles.convex,-1,sim.texturemap_plane,4+8,{0.3,0.3})
    end
    
    -- Now the label:

    -- Remove the current labels:
    local objs=sim.getObjectsInTree(model.handle,sim.object_shape_type,1+2)
    for i=1,#objs,1 do
        local h=objs[i]
        if h~=model.specHandles.texture and h~=model.specHandles.smallLabel and h~=model.specHandles.largeLabel then
            sim.removeObject(h)
        end
    end
    
    -- Now process the label:
    if sim.boolAnd32(bitC,8)>0 then
        local useLargeLabel=(sim.boolAnd32(bitC,64)>0)
        local labelSize=smallLabelSize
        local modelLabelHandle=model.specHandles.smallLabel
        if useLargeLabel then
            labelSize=largeLabelSize
            modelLabelHandle=model.specHandles.largeLabel
        end
        local h=sim.copyPasteObjects({modelLabelHandle},0)[1]
        sim.setObjectParent(h,model.handle,true)
        sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,255) -- make it visible
        sim.setObjectSpecialProperty(h,sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable) -- make renderable and detectable
        local code=partLabelC['placementCode'][1]
        local toExecute='local boxSizeX='..boxSize[1]..'\n'
        toExecute=toExecute..'local boxSizeY='..boxSize[2]..'\n'
        toExecute=toExecute..'local boxSizeZ='..boxSize[3]..'\n'
        toExecute=toExecute..'local labelSizeX='..labelSize[1]..'\n'
        toExecute=toExecute..'local labelSizeY='..labelSize[2]..'\n'
        toExecute=toExecute..'local labelRadius='..(0.5*math.sqrt(labelSize[1]*labelSize[1]+labelSize[2]*labelSize[2]))..'\n'

        toExecute=toExecute..'return {'..code..'}'
        local res,theTable=sim.executeLuaCode(toExecute)
        sim.setObjectPosition(h,model.handle,theTable[1])
        sim.setObjectOrientation(h,model.handle,theTable[2])
        local labelData={}
        labelData['labelIndex']=1
        sim.writeCustomDataBlock(h,simBWF.modelTags.LABEL_PART,sim.packTable(labelData))
    end
end


function sysCall_cleanup_specific()
    sim.removeObject(model.specHandles.texture)
    sim.removeObject(model.specHandles.smallLabel)
    sim.removeObject(model.specHandles.largeLabel)
end

