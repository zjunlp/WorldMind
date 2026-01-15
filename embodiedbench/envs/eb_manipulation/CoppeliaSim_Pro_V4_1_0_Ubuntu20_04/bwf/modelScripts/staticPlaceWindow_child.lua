getAxesWithOrderingAccordingToSize=function(partHandle)
    local modProp=sim.getModelProperty(partHandle)
    local sx=0
    local sy=0
    local sz=0
    if sim.boolAnd32(modProp,sim.modelproperty_not_model)==0 then
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_x )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_x )
        sx=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_y )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_y )
        sy=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_z )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_z )
        sz=mmax-mmin
    else
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_x )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_x )
        sx=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_y )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_y )
        sy=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_z )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_z )
        sz=mmax-mmin
    end
    local m=sim.getObjectMatrix(partHandle,-1)
    local axes={{sx,{m[1],m[5],m[9]}},{sy,{m[2],m[6],m[10]}},{sz,{m[3],m[7],m[11]}}}
    if axes[1][1]>axes[2][1] then
        local tmp=axes[1]
        axes[1]=axes[2]
        axes[2]=tmp
    end
    if axes[2][1]>axes[3][1] then
        local tmp=axes[2]
        axes[2]=axes[3]
        axes[3]=tmp
    end
    if axes[1][1]>axes[2][1] then
        local tmp=axes[1]
        axes[1]=axes[2]
        axes[2]=tmp
    end
    return {axes[1][2],axes[2][2],axes[3][2]}
end

getPartMass=function(partHandle)
    local m=0
    if partHandle>=0 then
        local modProp=sim.getModelProperty(partHandle)
        if sim.boolAnd32(modProp,sim.modelproperty_not_model)==0 then
            local objects={partHandle}
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
                        m=m+sim.getShapeMassAndInertia(handle)
                    end
                end
            end
        else
            m=m+sim.getShapeMassAndInertia(partHandle)
        end
    end
    return m
end

---[[
displayParts=function(trackingWindowParts)
    sim.addDrawingObjectItem(sphere1Container,nil)
    sim.addDrawingObjectItem(lineContainerR,nil)
    sim.addDrawingObjectItem(lineContainerG,nil)
    sim.addDrawingObjectItem(lineContainerB,nil)
    local al=0.06
    for key,value in pairs(trackingWindowParts) do
        local p=value['pickPos']
        local a=value['axes']
        sim.addDrawingObjectItem(sphere1Container,{p[1],p[2],p[3],0,0,1})
        sim.addDrawingObjectItem(lineContainerR,{p[1],p[2],p[3]+0.001,p[1]+a[1][1]*al,p[2]+a[1][2]*al,p[3]+a[1][3]*al+0.001})
        sim.addDrawingObjectItem(lineContainerG,{p[1],p[2],p[3]+0.001,p[1]+a[2][1]*al,p[2]+a[2][2]*al,p[3]+a[2][3]*al+0.001})
        sim.addDrawingObjectItem(lineContainerB,{p[1],p[2],p[3]+0.001,p[1]+a[3][1]*al,p[2]+a[3][2]*al,p[3]+a[3][3]*al+0.001})
    end
end
--]]
--[[
displayParts=function(trackingWindowParts)
    sim.addDrawingObjectItem(sphere1Container,nil)
    for i=1,4,1 do
        sim.addDrawingObjectItem(decorationContainers[i],nil)
    end
    sim.addDrawingObjectItem(lineContainerR,nil)
    sim.addDrawingObjectItem(lineContainerG,nil)
    sim.addDrawingObjectItem(lineContainerB,nil)
    local al=0.06
    for key,value in pairs(trackingWindowParts) do
        local decoration=value['decorationInfo']
        if decoration then
            for i=1,#decoration,1 do
                local h=decoration[i]['dummyHandle']
                local stage=sim.boolAnd32(decoration[i]['processingStage'],1+2)
                local p=sim.getObjectPosition(h,-1)
                sim.addDrawingObjectItem(decorationContainers[stage+1],{p[1],p[2],p[3],0,0,1})
            end
        else
            local p=value['pickPos']
            local a=value['axes']
            sim.addDrawingObjectItem(sphere1Container,{p[1],p[2],p[3],0,0,1})
            sim.addDrawingObjectItem(lineContainerR,{p[1],p[2],p[3]+0.001,p[1]+a[1][1]*al,p[2]+a[1][2]*al,p[3]+a[1][3]*al+0.001})
            sim.addDrawingObjectItem(lineContainerG,{p[1],p[2],p[3]+0.001,p[1]+a[2][1]*al,p[2]+a[2][2]*al,p[3]+a[2][3]*al+0.001})
            sim.addDrawingObjectItem(lineContainerB,{p[1],p[2],p[3]+0.001,p[1]+a[3][1]*al,p[2]+a[3][2]*al,p[3]+a[3][3]*al+0.001})
        end
    end
end
--]]
displayTargets=function(targets)
    for i=1,4,1 do
        sim.addDrawingObjectItem(decorationContainers[i],nil)
    end
    for key,value in pairs(targets) do
        local stage=sim.boolAnd32(value['processingStage'],1+2)
        local p=sim.getObjectPosition(key,-1)
        sim.addDrawingObjectItem(decorationContainers[stage+1],{p[1],p[2],p[3],0,0,1})
    end
end

displayGreyTargets=function(targets)
    sim.addDrawingObjectItem(decorationContainerGrey,nil)
    for key,value in pairs(targets) do
        local p=sim.getObjectPosition(key,-1)
        sim.addDrawingObjectItem(decorationContainerGrey,{p[1],p[2],p[3],0,0,1})
    end
end

getSensorState=function()
    if sensorHandle>=0 then
        local data=sim.readCustomDataBlock(sensorHandle,'XYZ_BINARYSENSOR_INFO')
        if data then
            data=sim.unpackTable(data)
            return data['detectionState']
        end
    end
    return 0
end


getAllPartsInWindow=function()
    local retVal={}
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    for i=1,#l,1 do
        local partHandle=l[i]
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(partHandle)
        if isInstanciated then
            local relPos=sim.getObjectPosition(partHandle,model)
            if (math.abs(relPos[1])<width*0.5) and (math.abs(relPos[2])<length*0.5) and (relPos[3]>0) and (relPos[3]<height) then
                local partData={}
                
                partData['partName']=data['name']
                partData['destinationName']=data['destination']
                partData['pickPos']=sim.getObjectPosition(partHandle,-1)
                partData['velocityVect']={0,0,0}
                partData['axes']=getAxesWithOrderingAccordingToSize(partHandle)
                local m=sim.getObjectMatrix(model,-1)
                partData['normalVect']={m[3],m[7],m[11]} -- the normal is the one from the model, not the object
                partData['hasLabel']=false
                partData['transform']={sim.getObjectPosition(partHandle,-1),sim.getObjectQuaternion(partHandle,-1)}
                partData['mass']=getPartMass(partHandle)
                local h=sim.createDummy(0.005)
                sim.setObjectParent(h,model,true)
                sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1024)
                sim.setObjectPosition(h,-1,partData['transform'][1])
                sim.setObjectQuaternion(h,-1,partData['transform'][2])
                partData['dummyHandle']=h
                retVal[partHandle]=partData
            end
        end
    end
    return retVal
end

removeTrackedPart=function(partHandle)
    local h=trackedParts[partHandle]['dummyHandle']
    local objs=sim.getObjectsInTree(h) -- if the part is decorated, it could have several dummy children
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    trackedParts[partHandle]=nil
end

attachDummiesAndDecorate=function(part,partData)
    local h=sim.createDummy(0.005)
    sim.setObjectParent(h,model,true)
    sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1024)
    
--    orientDummyAccordingToAxes(h,partData['axes'])
    sim.setObjectPosition(h,-1,partData['transform'][1])
    sim.setObjectQuaternion(h,-1,partData['transform'][2])

    partData['dummyHandle']=h
    if not partData['decorationInfo'] then
        local data=nil
        if overridePallet then
            data=sim.readCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW)
        else
            data=sim.readCustomDataBlock(part,simBWF.modelTags.PART)
        end
        data=sim.unpackTable(data)
        if #data['palletPoints']>0 then
            partData['decorationInfo']=data['palletPoints']
        end
    end
    if partData['decorationInfo'] then
        local allItems=partData['decorationInfo']
        local modelP=sim.getObjectPosition(model,-1)
        for i=1,#allItems,1 do
            local h2=sim.createDummy(0.005)
            sim.setObjectParent(h2,h,true)

            sim.setObjectPosition(h2,h,allItems[i]['pos'])
            sim.setObjectOrientation(h2,h,allItems[i]['orient'])
            sim.setObjectInt32Parameter(h2,sim.objintparam_visibility_layer,1024)
            allItems[i]['dummyHandle']=h2
        end
    end
end

removeTrackedLocation=function(associatedDummyHandle)
    for key,value in pairs(trackedParts) do
        local decoration=value['decorationInfo']
        if decoration then
            for i=1,#decoration,1 do
                if decoration[i]['dummyHandle']==associatedDummyHandle then
                    decoration[i]['processingStage']=decoration[i]['processingStage']+1
                    return
                end
            end
        end
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDSTATICPLACEWINDOW)
    data=sim.unpackTable(data)
    sensorHandle=simBWF.getReferencedObjectHandle(model,simBWF.STATICPLACEWINDOW_SENSOR_REF)
    width=data['width']
    length=data['length']
    height=data['height']
    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(data['bitCoded'],4)>0)
    sphere1Container=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,1,0})
    lineContainerR=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{1,0,0})
    lineContainerG=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,1,0})
    lineContainerB=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,0,1})
    decorationContainers={}
    decorationContainers[1]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,1})
    decorationContainers[2]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,0,1})
    decorationContainers[3]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,1,0})
    decorationContainers[4]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,0})
    decorationContainerGrey=sim.addDrawingObject(sim.drawing_spherepoints,0.005,0,-1,9999,{0.1,0.1,0.1})
    
    liftOffset=data.liftOffset
    doLiftMovements=(liftOffset[1]~=0) or (liftOffset[2]~=0) or (liftOffset[3]~=0)

    trackedParts={}
    previousSensorState=0
    previousTimeWhenOwnTriggered=0
    mode=0 -- 0=fill mode (waiting for sensor trigger), 1=empty mode (waiting for trackedParts to be empty)
    lastCurrLay=1
    initM=sim.getObjectMatrix(model,-1)
    desiredOffset={0,0,0}
    catchupOffset={0,0,0}
end



function sysCall_actuation()
    if doLiftMovements then
        if not waitTime then
            local jmp={0,0,0}
            local yep=false
            for i=1,3,1 do
                if math.abs(desiredOffset[i])>0.00002 then
                    if math.abs(desiredOffset[i])>=0.003 then
                        jmp[i]=0.003*math.abs(desiredOffset[i])/desiredOffset[i]
                        desiredOffset[i]=desiredOffset[i]-jmp[i]
                    else
                        jmp[i]=desiredOffset[i]
                        desiredOffset[i]=0
                    end
                    yep=true
                end
            end
            if yep then
                local p=sim.getObjectPosition(model,-1)
                p[1]=p[1]+jmp[1]*initM[1]+jmp[2]*initM[2]+jmp[3]*initM[3]
                p[2]=p[2]+jmp[1]*initM[5]+jmp[2]*initM[6]+jmp[3]*initM[7]
                p[3]=p[3]+jmp[1]*initM[9]+jmp[2]*initM[10]+jmp[3]*initM[11]
                sim.setObjectPosition(model,-1,p)
            end
        end
    end
    if waitTime then
        if sim.getSimulationTime()-waitTime>3 then
            trackedParts={}
            waitTime=nil
            local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDSTATICPLACEWINDOW)
            data=sim.unpackTable(data)
            data['detectionState']=data['detectionState']+1
            sim.writeCustomDataBlock(model,simBWF.modelTags.OLDSTATICPLACEWINDOW,sim.packTable(data))
        end
    end
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDSTATICPLACEWINDOW)
    data=sim.unpackTable(data)
    local sensorState=getSensorState()

    if mode==0 then
        if sensorState~=previousSensorState then
            trackedParts=getAllPartsInWindow()
            local cnt=0
            for key,value in pairs(trackedParts) do
                attachDummiesAndDecorate(key,value)
--                trackedParts[key]=value
                cnt=cnt+1
            end
            if cnt>0 then
                mode=1
            end
        else
            if t-previousTimeWhenOwnTriggered>0.251 then
                data['triggerState']=data['triggerState']+1
                previousTimeWhenOwnTriggered=t
            end
        end
    end
    local toRemove=data['itemsToRemoveFromTracking']
    for i=1,#toRemove,1 do
        removeTrackedPart(toRemove[i])
    end
    data['itemsToRemoveFromTracking']={}
    if mode>0 then
        local cnt=0
        for key,value in pairs(trackedParts) do
            cnt=cnt+1
        end
        if cnt==0 then
            mode=0
        end
    end
    local toIncrement=data['targetPositionsToMarkAsProcessed']
    for i=1,#toIncrement,1 do
        removeTrackedLocation(toIncrement[i])
    end
    data['targetPositionsToMarkAsProcessed']={}

    
    -- Now update the tracked targets:
    local trackedTargetsInWindow_currentLayer={}
    local trackedTargetsInWindow_otherLayers={}

    -- Check the lowest processing stage:
    local lowestOverallLayer=99
    for key,value in pairs(trackedParts) do
        if value['decorationInfo'] then
            local allItems=value['decorationInfo']
            for i=1,#allItems,1 do
                if allItems[i]['processingStage']==0 then
                    local lay=allItems[i]['layer']
                    if lay<lowestOverallLayer then
                        lowestOverallLayer=lay
                    end
                    break
                end
            end
        end
    end

    for key,value in pairs(trackedParts) do
        if value['decorationInfo'] then
            local allItems=value['decorationInfo']

            local currLay=99
            for i=1,#allItems,1 do
                if allItems[i]['processingStage']==0 then
                    local l=allItems[i]['layer']
                    if l<currLay then
                        currLay=l
                    end
                end
            end
            
            if doLiftMovements then
                currLay=lowestOverallLayer
                if lastCurrLay<currLay then
                    if currLay==99 then
                        for i=1,3,1 do
                            desiredOffset[i]=catchupOffset[i]
                            catchupOffset[i]=0
                        end
                        waitTime=sim.getSimulationTime()
                    else
                        for i=1,3,1 do
                            desiredOffset[i]=liftOffset[i]
                            catchupOffset[i]=catchupOffset[i]-liftOffset[i]
                        end
                    end
                end
            else
                if lowestOverallLayer==99 then
                    if not waitTime then
                        waitTime=sim.getSimulationTime()
                    end
                end
            end
            
            lastCurrLay=currLay
            -- Go through all items...
            for i=1,#allItems,1 do
                local h=allItems[i]['dummyHandle']
                local p=sim.getObjectPosition(h,model)
 --               print(h,p[1],p[2],p[3])
                -- ... in the window...
                if (math.abs(p[1])<width*0.5) and (math.abs(p[2])<length*0.5) and (p[3]>0) and (p[3]<height) then
                    -- ... and that are on the same layer as the lowest layer:
                    local dat={}
                    dat['processingStage']=allItems[i]['processingStage']
                    dat['ser']=allItems[i]['ser']
                    dat['partHandle']=key
                    dat['velocityVect']=value['velocityVect']
                    dat['partName']=value['partName']
                    if allItems[i]['layer']==currLay then
                        -- Those items are being transmitted to robots
                        trackedTargetsInWindow_currentLayer[h]=dat
                    else
                        -- items on other layers are just being displayed, but not transmitted:
                        trackedTargetsInWindow_otherLayers[h]=dat
                    end
                end
            end
        end
    end
    data['trackedTargetsInWindow']=trackedTargetsInWindow_currentLayer
    data['trackedItemsInWindow']=trackedTargetsInWindow_currentLayer -- trackedPartsInTrackingWindow
    data['transferItems']=trackedPartsInTransferWindow
--    sim.writeCustomDataBlock(model,OLDSTATICPLACEWINDOW_TAG.TRACKINGWINDOW_TAG,sim.packTable(data))




            
    
    if showPoints then
--        simpleDisplayParts(trackedParts)
--        displayParts(trackedPartsInTrackingWindow)
        displayTargets(trackedTargetsInWindow_currentLayer)
        displayGreyTargets(trackedTargetsInWindow_otherLayers)
    end

--    data['trackedItemsInWindow']=trackedParts
    sim.writeCustomDataBlock(model,simBWF.modelTags.OLDSTATICPLACEWINDOW,sim.packTable(data))

    previousSensorState=sensorState
end