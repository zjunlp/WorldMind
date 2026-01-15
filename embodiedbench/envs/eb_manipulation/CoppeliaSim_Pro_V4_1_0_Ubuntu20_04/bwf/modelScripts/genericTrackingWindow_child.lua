simpleDisplayParts=function(parts)
    sim.addDrawingObjectItem(sphere3Container,nil)
    for key,value in pairs(parts) do
        local p=value['pickPos']
        sim.addDrawingObjectItem(sphere3Container,{p[1],p[2],p[3],0,0,1})
    end
end

displayParts=function(trackingWindowParts)
    sim.addDrawingObjectItem(sphere1Container,nil)
--    sim.addDrawingObjectItem(sphere2Container,nil)
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

getAllPartsFromInput=function()
    local dat={}
    if inputHandle~=-1 then
        local data=sim.readCustomDataBlock(inputHandle,'XYZ_DETECTIONWINDOW_INFO')
        if data then
            data=sim.unpackTable(data)
            dat=data['detectedItems']
        else
            data=sim.readCustomDataBlock(inputHandle,simBWF.modelTags.TRACKINGWINDOW)
            if data then
                data=sim.unpackTable(data)
                dat=data['transferItems']
            end
        end
    end
    
    if conveyorHandle>=0 then
        local m=sim.getObjectMatrix(model,-1)
        sim.invertMatrix(m)
        local ret={}
        for key,value in pairs(dat) do
            local p=value['pickPos']
            p=sim.multiplyVector(m,p)
            if (p[2]<0) and (math.abs(p[1])<width*0.5) then
                ret[key]=value -- track only parts coming from the upstream side and inside the window's width
            end
        end
        return ret
    end
    return dat -- working without conveyor (non-moving items)
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

attachDummiesAndDecorate=function(part,partData)
    local h=sim.createDummy(0.005)
    sim.setObjectParent(h,model,true)
    sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1024)
    
--    orientDummyAccordingToAxes(h,partData['axes'])
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

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    trackingWindowShape=sim.getObjectHandle('genericTrackingWindow_track')
    stopLineShape=sim.getObjectHandle('genericTrackingWindow_stopLine')
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW)
    data=sim.unpackTable(data)
    local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local suff=sim.getNameSuffix(nil)
    sim.setNameSuffix(-1)
    conveyorHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF)
    inputHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDTRACKINGWINDOW_INPUT_REF)
    sim.setNameSuffix(suff)
    sim.setInt32Parameter(sim.intparam_error_report_mode,err)
    conveyorVector={1,0,0}
    previousConveyorEncoderDistance=0
    if conveyorHandle>=0 then
        local m=sim.getObjectMatrix(conveyorHandle,-1)
        conveyorVector[1]=m[2]
        conveyorVector[2]=m[6]
        conveyorVector[3]=m[10]
    end
    width=data['width']
    length=data['length']
    stopLinePos=data['stopLinePos']
    stopLineProcessingStage=data['stopLineProcessingStage']
    transferLength=data['transferLength']
    transferStart=data['transferStart']
    height=data['height']
    if sim.boolAnd32(data['bitCoded'],2)>0 then
        console=sim.auxiliaryConsoleOpen('Parts/targets in tracking window',1000,4,nil,{600,300},nil,{0.9,1,0.9})
    end
    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(data['bitCoded'],4)>0)
    overridePallet=(sim.boolAnd32(data['bitCoded'],8)>0)
    stopLine=(sim.boolAnd32(data['bitCoded'],16)>0)
    sphere1Container=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,1,0})
    sphere3Container=sim.addDrawingObject(sim.drawing_spherepoints,0.01,0,-1,9999,{1,1,1})
    decorationContainers={}
    decorationContainers[1]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,1})
    decorationContainers[2]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,0,1})
    decorationContainers[3]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,1,0})
    decorationContainers[4]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,0})
    decorationContainerGrey=sim.addDrawingObject(sim.drawing_spherepoints,0.005,0,-1,9999,{0.1,0.1,0.1})

    lineContainerR=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{1,0,0})
    lineContainerG=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,1,0})
    lineContainerB=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,0,1})
    allPreviousInputParts={}
    trackedParts={}
    trackedPartsInTrackingWindow={}
    trackedPartsInTransferWindow={}
    previousParts={}
    previousTime=0
    staticWindowFrozen=false
end

function sysCall_sensing()
    if conveyorHandle>=0 then
        sensing_withConveyor()
    else
        sensing_withoutConveyor()
    end
end
    
function sensing_withConveyor()
    local t=sim.getSimulationTime()
    local dt=t-previousTime
    local encoderDistance=getConveyorEncoderDistance()
    local conveyorDl=encoderDistance-previousConveyorEncoderDistance
    previousConveyorEncoderDistance=encoderDistance
    local trackDx={conveyorVector[1]*conveyorDl,conveyorVector[2]*conveyorDl,conveyorVector[3]*conveyorDl}
    local allInputParts=getAllPartsFromInput()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW)
    data=sim.unpackTable(data)
    for key,value in pairs(allPreviousInputParts) do
        if not allInputParts[key] then
            -- This part has disappeared from the input window
            -- We need to track it
            attachDummiesAndDecorate(key,value)
            trackedParts[key]=value
        end
    end
    for key,value in pairs(allInputParts) do
        if trackedParts[key] then
            -- This part has reappeared in the input window. Can happen with the detection window (e.g. parts sliding)
            -- Stop tracking it
            removeTrackedPart(key)
        end
    end
    local toRemove=data['itemsToRemoveFromTracking']
    for i=1,#toRemove,1 do
        removeTrackedPart(toRemove[i])
    end
    data['itemsToRemoveFromTracking']={}

    local toIncrement=data['targetPositionsToMarkAsProcessed']
    for i=1,#toIncrement,1 do
        removeTrackedLocation(toIncrement[i])
    end
    data['targetPositionsToMarkAsProcessed']={}

    allPreviousInputParts=allInputParts
    local m=sim.getObjectMatrix(model,-1)
    sim.invertMatrix(m)
    trackedPartsInTrackingWindow={}
    trackedPartsInTransferWindow={}
    toRemove={}
    local bias=0
    if dt~=0 then
        bias=math.abs(conveyorDl)*data['associatedRobotTrackingCorrectionTime']/dt
    end
    sim.setObjectPosition(trackingWindowShape,model,{0,-bias,height/2})

    local stopConveyor=false
    for key,value in pairs(trackedParts) do
        -- update the pick pos and velocity vector of the tracked parts:
        value['velocityVect']={trackDx[1]/dt,trackDx[2]/dt,trackDx[3]/dt}
        local p=value['pickPos']
        p[1]=p[1]+trackDx[1]
        p[2]=p[2]+trackDx[2]
        p[3]=p[3]+trackDx[3]
        value['pickPos']=p

        p=value['transform'][1]
        p[1]=p[1]+trackDx[1]
        p[2]=p[2]+trackDx[2]
        p[3]=p[3]+trackDx[3]
        value['transform'][1]=p

        local dum=value['dummyHandle']
        p=value['pickPos']
        sim.setObjectPosition(dum,-1,p)
        -- Orientation was already set at dummy creation

        -- We track the pickPos, not the transform pos!
        p=sim.multiplyVector(m,p)

        -- Update the parts in the tracking window:
        if (math.abs(p[1])<width*0.5) and (p[2]>-length*0.5-bias) and (p[2]<length*0.5-bias) and (p[3]>0) and (p[3]<height) then
            trackedPartsInTrackingWindow[key]=value
        end

        -- Take care of parts after the stop line:
        if stopLine and not value['decorationInfo'] then
            if (p[2]>stopLinePos) then
                stopConveyor=true
            end
        end

        -- Update the parts in the transfer window (ignore the width and height):
        if p[2]>(transferStart+length*0.5) then
            trackedPartsInTransferWindow[key]=value
        end

        -- Stop tracking all items that arr downstream the transfer window edge:
        if p[2]>(transferStart+transferLength+length*0.5) then
            toRemove[#toRemove+1]=key -- when out of the transfer window --> no not track anymore
        end
    end
    for i=1,#toRemove,1 do
        removeTrackedPart(toRemove[i])
    end

    -- Now update the tracked targets:
    local trackedTargetsInWindow_currentLayer={}
    local trackedTargetsInWindow_otherLayers={}
    for key,value in pairs(trackedParts) do
        if value['decorationInfo'] then
            local allItems=value['decorationInfo']

            -- Check the lowest processing stage:
            local currStage=99
            for i=1,#allItems,1 do
                if allItems[i]['processingStage']<currStage then
                    currStage=allItems[i]['processingStage']
                end
            end

            -- Check the lowest layer of the lowest processing stage:
            local currLay=99
            for i=1,#allItems,1 do
                if allItems[i]['processingStage']==currStage then
                    local l=allItems[i]['layer']
                    if l<currLay then
                        currLay=l
                    end
                end
            end
            -- Go through all items...
            for i=1,#allItems,1 do
                local h=allItems[i]['dummyHandle']
                local p=sim.getObjectPosition(h,model)
                -- ... in the window...
                if (math.abs(p[1])<width*0.5) and (p[2]>-length*0.5-bias) and (p[2]<length*0.5-bias) and (p[3]>0) and (p[3]<height) then
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

                -- Take care of parts after the stop line
                if stopLine and allItems[i]['processingStage']<stopLineProcessingStage then
                    if (p[2]>stopLinePos) then
                        stopConveyor=true
                    end
                end
                
            end
        end
    end
    data['trackedTargetsInWindow']=trackedTargetsInWindow_currentLayer
    data['trackedItemsInWindow']=trackedPartsInTrackingWindow
    data['transferItems']=trackedPartsInTransferWindow
    sim.writeCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
    if stopConveyor then
        if conveyorHandle>=0 and (not stopRequest) then
            local data=sim.readCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR)
            if data then
                data=sim.unpackTable(data)
                local stopRequests=data['stopRequests']
                stopRequests[model]=true -- we have a stop request from this tracking window
                data['stopRequests']=stopRequests
                sim.writeCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR,sim.packTable(data))
                stopRequest=true
            end
        end
    else
        if stopRequest and conveyorHandle>=0 then
            local data=sim.readCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR)
            if data then
                data=sim.unpackTable(data)
                local stopRequests=data['stopRequests']
                stopRequests[model]=nil
                data['stopRequests']=stopRequests
                sim.writeCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR,sim.packTable(data))
            end
            stopRequest=nil
        end
    end

    if showPoints then
        simpleDisplayParts(trackedParts)
        displayParts(trackedPartsInTrackingWindow)
        displayTargets(trackedTargetsInWindow_currentLayer)
        displayGreyTargets(trackedTargetsInWindow_otherLayers)
    end
    displayConsoleIfNeeded(trackedPartsInTrackingWindow)
    previousParts=parts
    previousTime=t
end

function sensing_withoutConveyor()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW)
    data=sim.unpackTable(data)
    if data['freezeStaticWindow'] then
        staticWindowFrozen=true
        data['freezeStaticWindow']=nil
    end
    if not staticWindowFrozen then
        local m=sim.getObjectMatrix(model,-1)
        sim.invertMatrix(m)
        local toRem={}
        for key,value in pairs(trackedParts) do
            toRem[#toRem+1]=key
        end
        for i=1,#toRem,1 do
            removeTrackedPart(toRem[i])
        end
        trackedPartsInTrackingWindow={}
        trackedParts=getAllPartsFromInput()
        for key,value in pairs(trackedParts) do
            attachDummiesAndDecorate(key,value)
            
            value['velocityVect']={0,0,0}
            local dum=value['dummyHandle']
            local p=value['pickPos']
            sim.setObjectPosition(dum,-1,p)
            
            -- We track the pickPos, not the transform pos!
            p=sim.multiplyVector(m,p)

            -- Update the parts in the tracking window:
            if (math.abs(p[1])<width*0.5) and (p[2]>-length*0.5) and (p[2]<length*0.5) and (p[3]>0) and (p[3]<height) then
                trackedPartsInTrackingWindow[key]=value
            end
        end


        
     --[[   
        allPreviousInputParts=allInputParts
        local m=sim.getObjectMatrix(model,-1)
        sim.invertMatrix(m)
        trackedPartsInTrackingWindow={}
        toRemove={}
        local bias=0
        if dt~=0 then
            bias=math.abs(conveyorDl)*data['associatedRobotTrackingCorrectionTime']/dt
        end
        sim.setObjectPosition(trackingWindowShape,model,{0,-bias,height/2})

        local stopConveyor=false
        for i=1,#toRemove,1 do
            removeTrackedPart(toRemove[i])
        end

        -- Now update the tracked targets:
        local trackedTargetsInWindow_currentLayer={}
        local trackedTargetsInWindow_otherLayers={}
        for key,value in pairs(trackedParts) do
            if value['decorationInfo'] then
                local allItems=value['decorationInfo']

                -- Check the lowest processing stage:
                local currStage=99
                for i=1,#allItems,1 do
                    if allItems[i]['processingStage']<currStage then
                        currStage=allItems[i]['processingStage']
                    end
                end

                -- Check the lowest layer of the lowest processing stage:
                local currLay=99
                for i=1,#allItems,1 do
                    if allItems[i]['processingStage']==currStage then
                        local l=allItems[i]['layer']
                        if l<currLay then
                            currLay=l
                        end
                    end
                end
                -- Go through all items...
                for i=1,#allItems,1 do
                    local h=allItems[i]['dummyHandle']
                    local p=sim.getObjectPosition(h,model)
                    -- ... in the window...
                    if (math.abs(p[1])<width*0.5) and (p[2]>-length*0.5-bias) and (p[2]<length*0.5-bias) and (p[3]>0) and (p[3]<height) then
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
        
        --]]
        data['trackedTargetsInWindow']=trackedTargetsInWindow_currentLayer
        data['trackedItemsInWindow']=trackedPartsInTrackingWindow

        if showPoints then
            displayParts(trackedPartsInTrackingWindow)
        end
        displayConsoleIfNeeded(trackedPartsInTrackingWindow)
    
    else
        local toRemove=data['itemsToRemoveFromTracking']
        local toIncrement=data['targetPositionsToMarkAsProcessed']
        if #toRemove>0 then -- or #toIncrement>0 then
            staticWindowFrozen=false
            local toRem={}
            for key,value in pairs(trackedParts) do
                toRem[#toRem+1]=key
            end
            for i=1,#toRem,1 do
                removeTrackedPart(toRem[i])
            end
            data['itemsToRemoveFromTracking']={}
            data['targetPositionsToMarkAsProcessed']={}

            data['trackedTargetsInWindow']={}
            data['trackedItemsInWindow']={}
            trackedParts={}
            trackedPartsInTrackingWindow={}
        end
        if showPoints then
            displayParts(trackedPartsInTrackingWindow)
        end
        displayConsoleIfNeeded(trackedPartsInTrackingWindow)
    end
    sim.writeCustomDataBlock(model,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
end