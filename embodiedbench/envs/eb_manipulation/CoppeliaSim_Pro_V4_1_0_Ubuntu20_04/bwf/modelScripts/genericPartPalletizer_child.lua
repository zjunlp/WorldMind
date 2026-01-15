displayTrackingLocations=function(yDistOffset,trackingLocations)
    sim.addDrawingObjectItem(decorationContainer,nil)
    local m=sim.getObjectMatrix(model,-1)
    for i=1,#trackingLocations,1 do
        local p=trackingLocations[i]['pos']
        local _p={p[1],p[2]+yDistOffset,p[3]}
        _p=sim.multiplyVector(m,_p)
        sim.addDrawingObjectItem(decorationContainer,{_p[1],_p[2],_p[3],0,0,1})
    end
end

removeOutOfSightLocations=function(yDistOffset,trackingLocations)
    local m=sim.getObjectMatrix(model,-1)
    i=1
    while i<=#trackingLocations do
        local p=trackingLocations[i]['pos']
        if math.abs(p[2]+yDistOffset)>length*0.5 then
            table.remove(trackingLocations,i)
        else
            i=i+1
        end
    end
end

displayConsoleIfNeeded=function(info)
    if console then
        sim.auxiliaryConsolePrint(console,nil)
        for key,value in pairs(info) do
            local str='<DESTROYED OBJECT>:\n'
            if sim.isHandleValid(key)>0 then
                str=sim.getObjectName(key)..':\n'
            end
            str=str..'    handle: '..key..', partName: '..value['partName']..', destinationName: '..value['destinationName']..'\n'
            str=str..'    pick position: ('
            str=str..simBWF.format("%.0f",value['pickPos'][1]*1000)..','
            str=str..simBWF.format("%.0f",value['pickPos'][2]*1000)..','
            str=str..simBWF.format("%.0f",value['pickPos'][3]*1000)..')\n'
            str=str..'    velocity vector: ('
            str=str..simBWF.format("%.0f",value['velocityVect'][1]*1000)..','
            str=str..simBWF.format("%.0f",value['velocityVect'][2]*1000)..','
            str=str..simBWF.format("%.0f",value['velocityVect'][3]*1000)..')\n'
            str=str..'    mass: '..simBWF.format("%.2f",value['mass'])..'\n'
            str=str..'    label detected: '..tostring(value['hasLabel'])..'\n----------------------------------------------------------------\n'
            sim.auxiliaryConsolePrint(console,str)
        end
    end
end

getAllPartsInWindowExceptProcessed=function(exceptions)
    local dat={}
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type)
    for i=1,#l,1 do
        if not exceptions[l[i]] then
            local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
            if isInstanciated then
                local p=sim.getObjectPosition(l[i],model)
                if math.abs(p[1])<width*0.5 and math.abs(p[2])<length*0.5 and p[3]>0 and p[3]<height then
                    dat[#dat+1]=l[i]
                end
            end
        end
    end
    return dat
end

getCrossProduct=function(u,v)
    return {u[2]*v[3]-u[3]*v[2],u[3]*v[1]-u[1]*v[3],u[1]*v[2]-u[2]*v[1]}
end

orientDummyAccordingToAxes=function(h,axes)
    local smallestInd=1
    local smallest=math.abs(axes[1][3])
    local largestInd=1
    local largest=math.abs(axes[1][3])
    for i=2,3,1 do
        if math.abs(axes[i][3])<smallest then
            smallestInd=i
            smallest=math.abs(axes[i][3])
        end
        if math.abs(axes[i][3])>=largest then
            largestInd=i
            largest=math.abs(axes[i][3])
        end
    end
    local z=axes[largestInd]
    if z[3]<0 then
        z[1]=z[1]*-1
        z[2]=z[2]*-1
        z[3]=z[3]*-1
    end
    local tb={}
    for i=1,3,1 do
        if i~=largestInd then
            tb[#tb+1]=axes[i]
        end
    end
    local m={0,0,0,0,0,0,0,0,0,0,0,0}
    m[1]=tb[1][1]
    m[5]=tb[1][2]
    m[9]=tb[1][3]
    m[3]=z[1]
    m[7]=z[2]
    m[11]=z[3]
    local y=getCrossProduct(z,{m[1],m[5],m[9]})
    m[2]=y[1]
    m[6]=y[2]
    m[10]=y[3]
    sim.setObjectMatrix(h,-1,m)
end

getConveyorEncoderDistance=function()
    if conveyorHandle~=-1 then
        local data=sim.readCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR)
        if data then
            data=sim.unpackTable(data)
            return data['encoderDistance']
        end
    end
    return 0
end

removeTrackedPart=function(partHandle)
    local h=trackedParts[partHandle]['dummyHandle']
    local objs=sim.getObjectsInTree(h) -- if the part is decorated, it could have several dummy children
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    trackedParts[partHandle]=nil
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

placeItem=function(part,location,yOffset)
    local p=location['pos']
    p[2]=p[2]+yOffset
    sim.setObjectPosition(part,model,p)
    local prop=sim.getModelProperty(part)
    if sim.boolAnd32(prop,sim.modelproperty_not_model)==0 then
        -- We have a model
        local l=sim.getObjectsInTree(part)
        for i=1,#l,1 do
            sim.resetDynamicObject(l[i])
        end
    else
        -- We have a shape
        sim.resetDynamicObject(part)
    end
end

generateTrackingLocations=function()
    local retTable={}
    if #palletizerData['palletPoints']>0 then
        local pp=palletizerData['palletPoints']
        for i=1,#pp,1 do
            local itm={}
            itm['pos']={pp[i]['pos'][1],pp[i]['pos'][2],pp[i]['pos'][3]}
            retTable[i]=itm
        end
    end
    return retTable
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    palletizerData=sim.readCustomDataBlock(model,'XYZ_PARTPALLETIZER_INFO')
    palletizerData=sim.unpackTable(palletizerData)
    conveyorHandle=simBWF.getReferencedObjectHandle(model,simBWF.PALLETIZER_CONVEYOR_REF)
    conveyorVector={1,0,0}
    previousConveyorEncoderDistance=0
    if conveyorHandle>=0 then
        local m=sim.getObjectMatrix(conveyorHandle,model)
        conveyorVector[1]=m[2]
        conveyorVector[2]=m[6]
        conveyorVector[3]=m[10]
    end
    width=palletizerData['width']
    length=palletizerData['length']
    height=palletizerData['height']
    enabled=(sim.boolAnd32(palletizerData['bitCoded'],2)>0)
    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(palletizerData['bitCoded'],4)>0)
    decorationContainer=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,1,0})
    previousTime=0
    processedParts={}
    trackingLocations={}
    trackDy=0
end

function sysCall_actuation()
    if enabled then
        local t=sim.getSimulationTime()
        local dt=t-previousTime
        local encoderDistance=getConveyorEncoderDistance()
        local conveyorDl=encoderDistance-previousConveyorEncoderDistance
        previousConveyorEncoderDistance=encoderDistance
        trackDy=trackDy+conveyorVector[2]*conveyorDl
        local parts=getAllPartsInWindowExceptProcessed(processedParts)
        for i=1,#parts,1 do
            if #trackingLocations==0 then
                trackDy=0
                trackingLocations=generateTrackingLocations()
            end
            if #trackingLocations>0 then -- With imported type, it can happen that we don't have any pallet points at all
                placeItem(parts[i],trackingLocations[#trackingLocations],trackDy)
                table.remove(trackingLocations,#trackingLocations)
            end
            processedParts[parts[i]]=t
        end
        local toRemove={}
        for key,value in pairs(processedParts) do
            if t-value>60 then
                toRemove[#toRemove+1]=key
            end
        end
        for i=1,#toRemove,1 do
            processedParts[toRemove[i]]=nil
        end
        removeOutOfSightLocations(trackDy,trackingLocations)
        if showPoints then
            displayTrackingLocations(trackDy,trackingLocations)
        end
        previousParts=parts
        previousTime=t
    end
end
