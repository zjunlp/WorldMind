function model.canScaleObjectNonIsometrically(objHandle,scaleAxisX,scaleAxisY,scaleAxisZ)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        return true -- iso scaling in this case
    end
    local t=sim.getObjectType(objHandle)
    if t==sim.object_joint_type then
        return true
    end
    if t==sim.object_dummy_type then
        return true
    end
    if t==sim.object_camera_type then
        return true
    end
    if t==sim.object_mirror_type then
        return true
    end
    if t==sim.object_light_type then
        return true
    end
    if t==sim.object_forcesensor_type then
        return true
    end
    if t==sim.object_path_type then
        return true
    end
    if t==sim.object_pointcloud_type then
        return false
    end
    if t==sim.object_octree_type then
        return false
    end
    if t==sim.object_graph_type then
        return false
    end
    if t==sim.object_proximitysensor_type then
        local r,p=sim.getObjectInt32Parameter(objHandle,sim.proxintparam_volume_type)
        if p==sim.volume_cylinder then
            return xIsY
        end
        if p==sim.volume_disc then
            return xIsZ
        end
        if p==sim.volume_cone then
            return false
        end
        if p==sim.volume_randomizedray then
            return false
        end
        return true
    end
    if t==sim.object_mill_type then
        local r,p=sim.getObjectInt32Parameter(objHandle,sim.millintparam_volume_type)
        if p==sim.volume_cylinder then
            return xIsY
        end
        if p==sim.volume_disc then
            return xIsZ
        end
        if p==sim.volume_cone then
            return false
        end
        return true
    end
    if t==sim.object_visionsensor_type then
        return xIsY
    end
    if t==sim.object_shape_type then
        local r,pt=sim.getShapeGeomInfo(objHandle)
        if sim.boolAnd32(r,1)~=0 then
            return false -- compound
        end
        if pt==sim.pure_primitive_spheroid then
            return false
        end
        if pt==sim.pure_primitive_disc then
            return xIsY
        end
        if pt==sim.pure_primitive_cylinder then
            return xIsY
        end
        if pt==sim.pure_primitive_cone then
            return xIsY
        end
        if pt==sim.pure_primitive_heightfield then
            return xIsY
        end
        return true
    end
end

function model.canScaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,ignoreNonScalableItems)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local yIsZ=(math.abs(1-math.abs(scaleAxisY/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        return true -- iso scaling in this case
    end
    local allDescendents=sim.getObjectsInTree(modelHandle,sim.handle_all,1)
    -- First the model base:
    local t=sim.getObjectType(modelHandle)
    if (t==sim.object_pointcloud_type) or (t==sim.object_pointcloud_type) or (t==sim.object_pointcloud_type) then
        if not ignoreNonScalableItems then
            if not model.canScaleObjectNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) then
                return false
            end
        end
    else
        if not model.canScaleObjectNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) then
            return false
        end
    end
    -- Ok, we can scale the base, now check the descendents:
    local baseFrameScalingFactors={scaleAxisX,scaleAxisY,scaleAxisZ}
    for i=1,#allDescendents,1 do
        local h=allDescendents[i]
        t=sim.getObjectType(h)
        if ( (t~=sim.object_pointcloud_type) and (t~=sim.object_pointcloud_type) and (t~=sim.object_pointcloud_type) ) or (not ignoreNonScalableItems) then
            local m=sim.getObjectMatrix(h,modelHandle)
            local axesMapping={-1,-1,-1} -- -1=no mapping
            local matchingAxesCnt=0
            local objFrameScalingFactors={nil,nil,nil}
            local singleMatchingAxisIndex
            for j=1,3,1 do
                local newAxis={m[j],m[j+4],m[j+8]}
                local x={math.abs(newAxis[1]),math.abs(newAxis[2]),math.abs(newAxis[3])}
                local v=math.max(math.max(x[1],x[2]),x[3])
                if v>0.99 then
                    matchingAxesCnt=matchingAxesCnt+1
                    if x[1]>0.9 then
                        axesMapping[j]=1
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                    if x[2]>0.9 then
                        axesMapping[j]=2
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                    if x[3]>0.9 then
                        axesMapping[j]=3
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                end
            end
            if matchingAxesCnt==0 then
                -- the child frame is not aligned at all with the model frame. And scaling is not iso-scaling
                -- Dummies, cameras, lights and force sensors do not mind:
                local t=sim.getObjectType(h)
                if (t~=sim.object_dummy_type) and (t~=sim.object_camera_type) and (t~=sim.object_light_type) and (t~=sim.object_forcesensor_type) then
                    return false
                end
            else
                if matchingAxesCnt==3 then
                    if not model.canScaleObjectNonIsometrically(h,objFrameScalingFactors[1],objFrameScalingFactors[2],objFrameScalingFactors[3]) then
                        return false
                    end
                else
                    -- We have only one axis that matches. We can scale the object only if the two non-matching axes have the same scaling factor:
                    local otherFactors={nil,nil}
                    for j=1,3,1 do
                        if j~=axesMapping[singleMatchingAxisIndex] then
                            if otherFactors[1] then
                                otherFactors[2]=baseFrameScalingFactors[j]
                            else
                                otherFactors[1]=baseFrameScalingFactors[j]
                            end
                        end
                    end
                    if (math.abs(1-math.abs(otherFactors[1]/otherFactors[2]))<0.001) then
                        local fff={otherFactors[1],otherFactors[1],otherFactors[1]}
                        fff[singleMatchingAxisIndex]=objFrameScalingFactors[singleMatchingAxisIndex]
                        if not model.canScaleObjectNonIsometrically(h,fff[1],fff[2],fff[3]) then
                            return false
                        end
                    else
                        return false
                    end
                end
            end
        end
    end
    return true
end

function model.scaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        sim.scaleObjects({modelHandle},scaleAxisX,false) -- iso scaling in this case
    else
        local avgScaling=(scaleAxisX+scaleAxisY+scaleAxisZ)/3
        local allDescendents=sim.getObjectsInTree(modelHandle,sim.handle_all,1)
        -- First the model base:
        sim.scaleObject(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,0)
        -- Now scale all the descendents:
        local baseFrameScalingFactors={scaleAxisX,scaleAxisY,scaleAxisZ}
        for i=1,#allDescendents,1 do
            local h=allDescendents[i]
            -- First scale the object itself:
            local m=sim.getObjectMatrix(h,modelHandle)
            local axesMapping={-1,-1,-1} -- -1=no mapping
            local matchingAxesCnt=0
            local objFrameScalingFactors={nil,nil,nil}
            for j=1,3,1 do
                local newAxis={m[j],m[j+4],m[j+8]}
                local x={math.abs(newAxis[1]),math.abs(newAxis[2]),math.abs(newAxis[3])}
                local v=math.max(math.max(x[1],x[2]),x[3])
                if v>0.99 then
                    matchingAxesCnt=matchingAxesCnt+1
                    if x[1]>0.9 then
                        axesMapping[j]=1
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                    if x[2]>0.9 then
                        axesMapping[j]=2
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                    if x[3]>0.9 then
                        axesMapping[j]=3
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                end
            end
            if matchingAxesCnt==0 then
                -- the child frame is not aligned at all with the model frame.
                sim.scaleObject(h,avgScaling,avgScaling,avgScaling,0)
            end

            if matchingAxesCnt==3 then
                -- the child frame is orthogonally aligned with the model frame
                sim.scaleObject(h,objFrameScalingFactors[1],objFrameScalingFactors[2],objFrameScalingFactors[3],0)
            else
                -- We have only one axis that is aligned with the model frame
                local objFactor,objIndex
                for j=1,3,1 do
                    if objFrameScalingFactors[j]~=nil then
                        objFactor=objFrameScalingFactors[j]
                        objIndex=j
                        break
                    end
                end
                local otherFactors={nil,nil}
                for j=1,3,1 do
                    if baseFrameScalingFactors[j]~=objFactor then
                        if otherFactors[1]==nil then
                            otherFactors[1]=baseFrameScalingFactors[j]
                        else
                            otherFactors[2]=baseFrameScalingFactors[j]
                        end
                    end
                end
                if (math.abs(1-math.abs(otherFactors[1]/otherFactors[2]))<0.001) then
                    local fff={otherFactors[1],otherFactors[1],otherFactors[1]}
                    fff[objIndex]=objFactor
                    sim.scaleObject(h,fff[1],fff[2],fff[3],0)
                else
                    local of=(otherFactors[1]+otherFactors[2])/2
                    local fff={of,of,of}
                    fff[objIndex]=objFactor
                    sim.scaleObject(h,fff[1],fff[2],fff[3],0)
                end
            end
            -- Now scale also the position of that object:
            local parentObjH=sim.getObjectParent(h)
            local m=sim.getObjectMatrix(parentObjH,modelHandle)
            m[4]=0
            m[8]=0
            m[12]=0
            local mi={}
            for j=1,12,1 do
                mi[j]=m[j]
            end
            sim.invertMatrix(mi)
            local p=sim.getObjectPosition(h,parentObjH)
            p=sim.multiplyVector(m,p)
            p[1]=p[1]*scaleAxisX
            p[2]=p[2]*scaleAxisY
            p[3]=p[3]*scaleAxisZ
            p=sim.multiplyVector(mi,p)
            sim.setObjectPosition(h,parentObjH,p)
        end
    end
end

function model.makeInvisibleOrNonRespondableToOtherParts(handle,invisible,nonRespondableToOtherParts)
    if invisible then
        local objs=sim.getObjectsInTree(handle)
        for i=1,#objs,1 do
            sim.setObjectInt32Parameter(objs[i],sim.objintparam_visibility_layer,0)
            local p=sim.getObjectSpecialProperty(objs[i])
            local p=sim.boolOr32(p,sim.objectspecialproperty_renderable)-sim.objectspecialproperty_renderable
            sim.setObjectSpecialProperty(objs[i],p)
        end
    end
    objs=sim.getObjectsInTree(handle,sim.object_shape_type)
    for i=1,#objs,1 do
        local r,m=sim.getObjectInt32Parameter(objs[i],sim.shapeintparam_respondable_mask)
        if nonRespondableToOtherParts then
            sim.setObjectInt32Parameter(objs[i],sim.shapeintparam_respondable_mask,sim.boolOr32(m,255)-255)
        end
    end
end

function model.getLabels(partH)
    -- There can be up to 3 labels in this part:
    local possibleLabels=sim.getObjectsInTree(partH,sim.object_shape_type,1)
    local labels={}
    for objInd=1,#possibleLabels,1 do
        local h=possibleLabels[objInd]
        local data=sim.readCustomDataBlock(h,simBWF.modelTags.LABEL_PART)
        if data then
            labels[#labels+1]=h
        end
    end
    return labels
end

function model.adjustSizeData(partH,sx,sy,sz)
    local data=sim.unpackTable(sim.readCustomDataBlock(partH,simBWF.modelTags.PART))
    local labelData=data['labelData']
    if labelData then
        local s=labelData['smallLabelSize']
        labelData['smallLabelSize']={s[1]*sx,s[2]*sy}
        local s=labelData['largeLabelSize']
        labelData['largeLabelSize']={s[1]*sx,s[2]*sy}
        local s=labelData['boxSize']
        labelData['boxSize']={s[1]*sx,s[2]*sy,s[3]*sz}
        data['labelData']=labelData
        sim.writeCustomDataBlock(partH,simBWF.modelTags.PART,sim.packTable(data))
    end
end

function model.setItemMass(handle,m)
    if m~=nil then -- Mass can be nil (for a default mass)
        -- Remember, the item can be a shape, or a model containing several shapes
        local currentMass=0
        local objects={handle}
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

        local objects={handle}
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
end

function model.regenerateOrRemoveLabels(partH,enabledLabels)
    -- There can be up to 3 labels in this part:
    local possibleLabels=sim.getObjectsInTree(partH,sim.object_shape_type,1)
    local labelData=sim.unpackTable(sim.readCustomDataBlock(partH,simBWF.modelTags.PART))['labelData']
    for ind=1,3,1 do
        for objInd=1,#possibleLabels,1 do
            local h=possibleLabels[objInd]
            if h>=0 then
                local data=sim.readCustomDataBlock(h,simBWF.modelTags.LABEL_PART)
                if data then
                    data=sim.unpackTable(data)
                    if data['labelIndex']==ind then
                        local bits={1,2,4}
                        if (sim.boolAnd32(bits[ind],enabledLabels)>0) then
                            -- We want to regenerate the position of this label
                            if labelData then
                                local bitC=labelData['bitCoded']
                                local smallLabelSize=labelData['smallLabelSize']
                                local largeLabelSize=labelData['largeLabelSize']
                                local useLargeLabel=(sim.boolAnd32(bitC,64*(2^(ind-1)))>0)
                                local labelSize=smallLabelSize
                                if useLargeLabel then
                                    labelSize=largeLabelSize
                                end
                                local code=labelData['placementCode'][ind]
                                local toExecute='local boxSizeX='..labelData['boxSize'][1]..'\n'
                                toExecute=toExecute..'local boxSizeY='..labelData['boxSize'][2]..'\n'
                                toExecute=toExecute..'local boxSizeZ='..labelData['boxSize'][3]..'\n'
                                toExecute=toExecute..'local labelSizeX='..labelSize[1]..'\n'
                                toExecute=toExecute..'local labelSizeY='..labelSize[2]..'\n'
                                toExecute=toExecute..'local labelRadius='..(0.5*math.sqrt(labelSize[1]*labelSize[1]+labelSize[2]*labelSize[2]))..'\n'

                                toExecute=toExecute..'return {'..code..'}'
                                local res,theTable=sim.executeLuaCode(toExecute)
                                sim.setObjectPosition(h,partH,theTable[1])
                                sim.setObjectOrientation(h,partH,theTable[2])
                            end
                        else
                            sim.removeObject(h) -- we do not want this label
                            possibleLabels[objInd]=-1
                        end
                    end
                end
            end
        end
    end
end

function model.instanciatePart(partHandle,itemPosition,itemOrientation,itemMass,itemScaling,allowChildItemsIfApplicable)
    local auxPartHandles=nil
    local p=sim.getModelProperty(partHandle)
    local tble=sim.copyPasteObjects({partHandle},1)
    local basePartCopy=tble[1]
    sim.writeCustomDataBlock(basePartCopy,simBWF.modelTags.GEOMETRY_PART,'') -- remove the embedded part geometry
    sim.setObjectParent(basePartCopy,model.handle,true)
    local basePartCopyData=sim.readCustomDataBlock(basePartCopy,simBWF.modelTags.PART)
    basePartCopyData=sim.unpackTable(basePartCopyData)
    local invisible=sim.boolAnd32(basePartCopyData['bitCoded'],1)>0
    local nonRespondableToOtherParts=sim.boolAnd32(basePartCopyData['bitCoded'],2)>0
    local ignoreBasePart=sim.boolAnd32(basePartCopyData['bitCoded'],4)>0
    local usePalletColors=sim.boolAnd32(basePartCopyData['bitCoded'],8)>0
    model.makeInvisibleOrNonRespondableToOtherParts(basePartCopy,invisible,nonRespondableToOtherParts)
    
    -- Destination:
--    if itemDestination then
--        basePartCopyData['destination']=itemDestination
--    end

    -- Size scaling:
    if itemScaling then
        local itemLabels=model.getLabels(basePartCopy)
        for j=1,#itemLabels,1 do
            sim.setObjectParent(itemLabels[j],-1,true)
        end
        if type(itemScaling)~='table' then
            -- iso-scaling
            model.adjustSizeData(basePartCopy,itemScaling,itemScaling,itemScaling)
            sim.scaleObjects({basePartCopy},itemScaling,false)
        else
            -- non-iso-scaling
            model.adjustSizeData(basePartCopy,itemScaling[1],itemScaling[2],itemScaling[3])
            if model.canScaleModelNonIsometrically(basePartCopy,itemScaling[1],itemScaling[2],itemScaling[3]) then
                model.scaleModelNonIsometrically(basePartCopy,itemScaling[1],itemScaling[2],itemScaling[3])
            end
        end
        for j=1,#itemLabels,1 do
            sim.setObjectParent(itemLabels[j],basePartCopy,true)
        end
    end
    
    -- Mass:
    if itemMass then
        model.setItemMass(basePartCopy,itemMass)
    end

   -- Labels:
    if invisible then
        labelsToEnable=0
    end
    if labelsToEnable and labelsToEnable>=0 then
        model.regenerateOrRemoveLabels(basePartCopy,labelsToEnable)
    end

    -- Position:
    sim.setObjectPosition(basePartCopy,-1,itemPosition)

    -- Orientation:
    sim.setObjectOrientation(basePartCopy,-1,itemOrientation)

    basePartCopyData['instanciated']=true
    basePartCopyData['type']=partHandle
    sim.writeCustomDataBlock(basePartCopy,simBWF.modelTags.PART,sim.packTable(basePartCopyData))
    
    -- Now check if that parts has a pallet with other parts attached:
    local attachedPalletHandle=simBWF.getReferencedObjectHandle(partHandle,simBWF.PART_PALLET_REF) -- we have to read it from the original base part
    if attachedPalletHandle>=0 and allowChildItemsIfApplicable then
        -- Yes!
        local baseM=sim.getObjectMatrix(basePartCopy,-1)
        local pallet=sim.unpackTable(sim.readCustomDataBlock(attachedPalletHandle,simBWF.modelTags.PALLET))
        local palletM=sim.buildMatrix(basePartCopyData.palletOffset,{pallet.yaw,pallet.pitch,pallet.roll})
        for i=1,#pallet.palletItemList,1 do
            local palletItem=pallet.palletItemList[i]
            if palletItem.model>=0 then
                local palletItemM=sim.buildMatrix({palletItem.locationX,palletItem.locationY,palletItem.locationZ},{palletItem.orientationY,palletItem.orientationP,palletItem.orientationR})
                local palletItemM=sim.multiplyMatrices(palletM,palletItemM)
                local palletItemM=sim.multiplyMatrices(baseM,palletItemM)
                local childPart=palletItem.model
                    
                local p=sim.getModelProperty(childPart)
                local tble=sim.copyPasteObjects({childPart},1)
                
                childPartCopy=tble[1]
                sim.writeCustomDataBlock(childPartCopy,simBWF.modelTags.GEOMETRY_PART,'') -- remove the embedded part geometry

                
                if usePalletColors then
                    local l2=sim.getObjectsInTree(childPartCopy,sim.object_shape_type)
                    for i=1,#l2,1 do
                        sim.setShapeColor(l2[i],nil,sim.colorcomponent_ambient_diffuse,{palletItem.colorR,palletItem.colorG,palletItem.colorB})
                    end
                end
                
                sim.setObjectParent(childPartCopy,model.handle,true)
                local data=sim.readCustomDataBlock(childPartCopy,simBWF.modelTags.PART)
                data=sim.unpackTable(data)
                local invisible=sim.boolAnd32(data['bitCoded'],1)>0
                local nonRespondableToOtherParts=sim.boolAnd32(data['bitCoded'],2)>0
                model.makeInvisibleOrNonRespondableToOtherParts(childPartCopy,invisible,nonRespondableToOtherParts)
                -- Correct for the part frame location (the template has its origine centered x/y, and at the bottom of z):
                local minMaxX=data.vertMinMax[1]
                local minMaxY=data.vertMinMax[2]
                local minMaxZ=data.vertMinMax[3]
                local xShift=(minMaxX[2]+minMaxX[1])/2
                local yShift=(minMaxY[2]+minMaxY[1])/2
                local zShift=-minMaxZ[1]
                palletItemM[4]=palletItemM[4]+palletItemM[1]*xShift+palletItemM[2]*yShift+palletItemM[3]*zShift
                palletItemM[8]=palletItemM[8]+palletItemM[5]*xShift+palletItemM[6]*yShift+palletItemM[7]*zShift
                palletItemM[12]=palletItemM[12]+palletItemM[9]*xShift+palletItemM[10]*yShift+palletItemM[11]*zShift
                --------
                sim.setObjectMatrix(childPartCopy,-1,palletItemM)
                data['instanciated']=true
                data['type']=childPart
                sim.writeCustomDataBlock(childPartCopy,simBWF.modelTags.PART,sim.packTable(data))
                if auxPartHandles==nil then
                    auxPartHandles={}
                end
                auxPartHandles[#auxPartHandles+1]=childPartCopy
            end
        end
    end
    local baseHandle=-1
    if ignoreBasePart and auxPartHandles and #auxPartHandles>0 then
        sim.removeModel(basePartCopy)
    else
        baseHandle=basePartCopy
    end
    
    local t=sim.getSimulationTime()
    if auxPartHandles and #auxPartHandles>0 then
        for i=1,#auxPartHandles,1 do
            local childPartCopy=auxPartHandles[i]
            p=sim.getObjectPosition(childPartCopy,-1)
            local partData={childPartCopy,t,p,true,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
            allProducedParts[#allProducedParts+1]=partData
            local d=simBWF.readPartInfo(childPartCopy)
            d['vel']={0,0,0}
            simBWF.writePartInfo(childPartCopy,d)
        end
    end
    if baseHandle>=0 then
        p=sim.getObjectPosition(baseHandle,-1)
        local partData={baseHandle,t,p,true,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
        allProducedParts[#allProducedParts+1]=partData
        local d=simBWF.readPartInfo(baseHandle)
        d['vel']={0,0,0}
        simBWF.writePartInfo(baseHandle,d)
    end
end

function model.tryToAttach(part)
    local index=0
    while (true) do
        objectsInContact,contactPt,forceDirectionAndAmplitude=sim.getContactInfo(0,part,index)
        if objectsInContact then
            local p=sim.getObjectPosition(part,-1)
            if p[3]>contactPt[3] then
                local part2=objectsInContact[1]
                if part2==part then
                    part2=objectsInContact[2]
                end
                -- Check if that object is a part or linked to a part:
                local pp=part2
                while pp>=0 do
                    local data=sim.readCustomDataBlock(pp,simBWF.modelTags.PART)
                    if data then
                        local sens=sim.createForceSensor(0,{0,1,1,0,0},{0.001,1,1,0,0})
                        sim.setObjectInt32Parameter(sens,sim.objintparam_visibility_layer,0) -- hidden
                        sim.setObjectPosition(sens,part,{0,0,0})
                        sim.setObjectParent(part,sens,true)
                        sim.setObjectParent(sens,part2,true)
                        return true
                    else
                        pp=sim.getObjectParent(pp)
                    end
                end
            end
            index=index+1
        else
            break
        end
    end
    return false
end

function model.handleCreatedParts()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()
    local i=1
    while i<=#allProducedParts do
        local h=allProducedParts[i][1]
        if sim.isHandleValid(h)>0 then
            local dataName=simBWF.modelTags.PART
            local data=sim.readCustomDataBlock(h,dataName)
            data=sim.unpackTable(data)
            local p=sim.getObjectPosition(h,-1)
            if allProducedParts[i][5] then
                -- The part is still active
                local deactivate=data['deactivate']
                local attach=false
                if data.attachStartCmd then
                    attach=sim.getSimulationTime()-data.attachStartCmd<3
                    if not attach then
                        data.attachStartCmd=nil
                    end
                end
                local dp={p[1]-allProducedParts[i][3][1],p[2]-allProducedParts[i][3][2],p[3]-allProducedParts[i][3][3]}
                local l=math.sqrt(dp[1]*dp[1]+dp[2]*dp[2]+dp[3]*dp[3])
                if (l>0.01*dt) then
                    allProducedParts[i][2]=t
                end
                allProducedParts[i][3]=p
                if (t-allProducedParts[i][2]>timeForIdlePartToDeactivate) then
                    deactivate=true
                end
                if deactivate then
                    local prop=sim.getModelProperty(h)
                    prop=sim.boolOr32(prop,sim.modelproperty_not_dynamic)
                    sim.setModelProperty(h,prop)
                    sim.resetDynamicObject(h) -- important, otherwise the dynamics engine doesn't notice the change!
                    allProducedParts[i][5]=false
                    data['vel']={0,0,0}
                else
                    if attach then
                        if model.tryToAttach(h) then
                            data['giveUpOwnership']=true
                        end
                    else
                        data['vel']={dp[1]/dt,dp[2]/dt,dp[3]/dt}
                    end
                end
                sim.writeCustomDataBlock(h,dataName,sim.packTable(data))
            end
            -- Does it want to be destroyed?
            if data['destroy'] or p[3]<-1000 or data['giveUpOwnership'] then
                if not data['giveUpOwnership'] then
                    sim.removeModel(h)
                else
                    sim.writeCustomDataBlock(h,simBWF.modelTags.PART,nil)
                end
                table.remove(allProducedParts,i)
            else
                i=i+1
            end
        else
            table.remove(allProducedParts,i)
        end
    end
end

function sysCall_init()
    model.codeVersion=1
    allProducedParts={}
    local l=sim.getObjectsWithTag(simBWF.modelTags.BLUEREALITYAPP,true)
    brAppObj=l[1]
end

function sysCall_sensing()
    timeForIdlePartToDeactivate=60
    local data=sim.readCustomDataBlock(brAppObj,simBWF.modelTags.BLUEREALITYAPP)
    data=sim.unpackTable(data)
    if data.deactivationTime then
        timeForIdlePartToDeactivate=data.deactivationTime
    end

    model.handleCreatedParts()
end

function sysCall_afterSimulation()
    allProducedParts={}
end

function sysCall_beforeSimulation()
    allProducedParts={}
    -- Handle all parts that are already instanciated (i.e. not created via a feeder):
    local l=simBWF.getAllInstanciatedParts()
    for i=1,#l,1 do
        model.makeInvisibleOrNonRespondableToOtherParts(l[i],false,false)
        sim.setObjectParent(l[i],model.handle,true)
        p=sim.getObjectPosition(l[i],-1)
        local partData={l[i],0,p,true,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
        allProducedParts[#allProducedParts+1]=partData
    end
end

