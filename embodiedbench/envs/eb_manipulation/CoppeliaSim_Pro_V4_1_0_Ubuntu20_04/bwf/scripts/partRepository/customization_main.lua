function model.embedPartGeometry(partHandle)
    -- 1. Get the vertices and indices of the part (in coords. relative to the shape frame first): 
    local p=sim.getModelProperty(partHandle)
    local vertices=nil
    local indices=nil
    local normals=nil
    if sim.boolAnd32(p,sim.modelproperty_not_model)>0 then
        -- We have a shape here
        vertices,indices,normals=sim.getShapeMesh(partHandle)
    else
        -- We have a model here
        vertices={}
        indices={}
        normals={}
        local shapes=sim.getObjectsInTree(partHandle,sim.object_shape_type,0)
        for i=1,#shapes,1 do
            local r,l=sim.getObjectInt32Parameter(shapes[i],sim.objintparam_visibility_layer)
            if sim.boolAnd32(l,255)>0 then
                -- For all visible shapes, get the data..
                local v,ind,norm=sim.getShapeMesh(shapes[i])
                -- Make the vertices relative to the model...
                local m=sim.getObjectMatrix(shapes[i],partHandle)
                local mr=sim.getObjectMatrix(shapes[i],partHandle)
                mr[4]=0
                mr[8]=0
                mr[12]=0
                for j=0,(#v/3)-1,1 do
                    local pt={v[3*j+1],v[3*j+2],v[3*j+3]}
                    pt=sim.multiplyVector(m,pt)
                    v[3*j+1]=pt[1]
                    v[3*j+2]=pt[2]
                    v[3*j+3]=pt[3]
                end
                for j=0,(#norm/3)-1,1 do
                    local n={norm[3*j+1],norm[3*j+2],norm[3*j+3]}
                    n=sim.multiplyVector(mr,n)
                    norm[3*j+1]=n[1]
                    norm[3*j+2]=n[2]
                    norm[3*j+3]=n[3]
                end
                -- Append the data to the existing mesh data:
                local vOff=#vertices
                local iOff=#indices
                local iiOff=#vertices/3
                local nOff=#normals
                for j=1,#v,1 do
                    vertices[vOff+j]=v[j]
                end
                for j=1,#ind,1 do
                    indices[iOff+j]=ind[j]+iiOff
                end
                for j=1,#norm,1 do
                    normals[nOff+j]=norm[j]
                end
            end
        end
    end
    
    -- Check the vertices min/max, relative to the part frame:
    local minMaxX={0,0}
    local minMaxY={0,0}
    local minMaxZ={0,0}
    for i=0,(#vertices/3)-1,1 do
        local pt={vertices[3*i+1],vertices[3*i+2],vertices[3*i+3]}
        if i==0 then
            minMaxX[1]=pt[1]
            minMaxX[2]=pt[1]
            minMaxY[1]=pt[2]
            minMaxY[2]=pt[2]
            minMaxZ[1]=pt[3]
            minMaxZ[2]=pt[3]
        else
            if pt[1]<minMaxX[1] then minMaxX[1]=pt[1] end
            if pt[1]>minMaxX[2] then minMaxX[2]=pt[1] end
            if pt[2]<minMaxY[1] then minMaxY[1]=pt[2] end
            if pt[2]>minMaxY[2] then minMaxY[2]=pt[2] end
            if pt[3]<minMaxZ[1] then minMaxZ[1]=pt[3] end
            if pt[3]>minMaxZ[2] then minMaxZ[2]=pt[3] end
        end
    end
    
    -- 2. Write the geom's offsets relative to the shape's frame:
    local pData=simBWF.readPartInfo(partHandle)--sim.readCustomDataBlock(partHandle,simBWF.modelTags.PART)
    pData.vertMinMax={minMaxX,minMaxY,minMaxZ}
    simBWF.writePartInfo(partHandle,pData)
    
    -- 3. Embed the vertices and indices into the part. But first transform the vertices to have the origin centered at the bottom center of the geometry:
    local geomData={}
    for i=0,#vertices/3-1,1 do
        vertices[3*i+1]=vertices[3*i+1]-(minMaxX[2]+minMaxX[1])/2
        vertices[3*i+2]=vertices[3*i+2]-(minMaxY[2]+minMaxY[1])/2
        vertices[3*i+3]=vertices[3*i+3]-minMaxZ[1]
    end
    geomData.vertices=vertices
    geomData.indices=indices
    geomData.normals=normals
    sim.writeCustomDataBlock(partHandle,simBWF.modelTags.GEOMETRY_PART,sim.packTable(geomData))
end

function model.getPartTable()
    local l=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
 --       print(sim.getObjectName(l[i]),sim.getObjectName(l[i]+sim.handleflag_altname),simBWF.getObjectAltName(l[i]))
        retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
    end
    return retL
end

function model.doesPartWithNameExist(name)
    local l=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
        if simBWF.getObjectAltName(l[i])==name then
            return true
        end
    end
    return false
end

function model.getPartData(partHandle)
    local l=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
        if l[i]==partHandle then
            local data=sim.readCustomDataBlock(partHandle,simBWF.modelTags.PART)
            if data then
                data=simBWF.readPartInfo(partHandle)
                return data
            end
        end
    end
end

function model.updatePartData(partHandle,data)
    local l=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
        if l[i]==partHandle then
            sim.writeCustomDataBlock(partHandle,simBWF.modelTags.PART,sim.packTable(data))
            model.updatePluginRepresentation_template(partHandle)
            break
        end
    end
end

function model.removePart(partHandle)
    local l=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
        if l[i]==partHandle then
            local pData=sim.readCustomDataBlock(partHandle,simBWF.modelTags.PART)
            pData=sim.unpackTable(pData)
            
            -- 1. Remove its plugin representation:
            model.removeFromPluginRepresentation_template(partHandle)

            -- 2. Remove the part:
            model.dlg.selectedPartId=-1
            local p=sim.getModelProperty(partHandle)
            if sim.boolAnd32(p,sim.modelproperty_not_model)>0 then
                sim.removeObject(partHandle)
            else
                sim.removeModel(partHandle)
            end
            
            simBWF.markUndoPoint()
            break
        end
    end
end

function model.insertPart(partHandle)
    local allNames=model.getAllPartNameMap()
    local data=simBWF.readPartInfo(partHandle)
    local nm=simBWF.getObjectAltName(partHandle)
    simBWF.writePartInfo(partHandle,data)

    nm=simBWF.getValidName(nm,true)
    local cnt
    local baseNm
    baseNm,cnt=simBWF.getNameAndNumber(nm)
    if baseNm=='' then
        baseNm='_'
    end
    nm=baseNm
    cnt=0
    while true do
        if not allNames[nm] then
            break
        end
        nm=baseNm..cnt
        cnt=cnt+1
    end

    simBWF.setObjectAltName(partHandle,nm)

    sim.setObjectPosition(partHandle,model.handle,{0,0,0}) -- keep the orientation as it is

    -- Make the model static, non-respondable, non-collidable, non-measurable, non-visible, etc.
    if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_model)>0 then
        -- Shape
        local p=sim.boolOr32(sim.getObjectProperty(partHandle),sim.objectproperty_dontshowasinsidemodel)
        sim.setObjectProperty(partHandle,p)
    else
        -- Model
        local p=sim.boolOr32(sim.getModelProperty(partHandle),sim.modelproperty_not_showasinsidemodel)
        sim.setModelProperty(partHandle,p)
    end

    model.removeAssociatedCustomizationScriptIfAvailable(partHandle)
    sim.setObjectParent(partHandle,model.handles.originalPartHolder,true)

    -- Destinations of that part (stored as references to loc frames and tracking windows
    for i=simBWF.PART_DESTINATIONFIRST_REF,simBWF.PART_DESTINATIONLAST_REF,1 do
        simBWF.setReferencedObjectHandle(partHandle,i,-1)
    end
    
    -- We embed into each part its geometry:
    model.embedPartGeometry(partHandle)
    model.updatePluginRepresentation_template(partHandle)
end

function model.getAllPartNameMap()
    local allNames={}
    local parts=sim.getObjectsInTree(model.handles.originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        allNames[simBWF.getObjectAltName(parts[i])]=parts[i]
    end
    return allNames
end

function model.removeAssociatedCustomizationScriptIfAvailable(h)
    local sh=sim.getCustomizationScriptAssociatedWithObject(h)
    if sh>0 then
        sim.removeScript(sh)
    end
end

function model.handleBackCompatibility()
    local allParts=model.getPartTable()
    for i=1,#allParts,1 do
        local h=allParts[i][2]
        local name=simBWF.getObjectAltName(h)
        if string.find(name,'__PART__')==1 then
            -- old pallet names had a hidden "__PART__" prefix. Try to correct for that:
            name=string.sub(name,9)
            simBWF.setObjectAltName(h,name)
        end
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()
    
    model.handleBackCompatibility()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_sensing()
    if not model.notFirstDuringSimulation then
        model.dlg.updateEnabledDisabledItems()
        model.notFirstDuringSimulation=true
    end
end

function sysCall_afterSimulation()
    model.dlg.updateEnabledDisabledItems()
    model.notFirstDuringSimulation=nil
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
