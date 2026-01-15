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

function sysCall_init()
    model=sim.getObjectHandle('genericPingPongPacker')
    trackingWindowModel=sim.getObjectAssociatedWithScript(sim.handle_self)
    local modelData=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    modelData=sim.unpackTable(modelData)
    locationName=modelData['locationName']

    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(modelData['bitCoded'],128)>0)

    decorationContainers={}
    decorationContainers[1]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,1})
    decorationContainers[2]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,0,1})
    decorationContainers[3]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,1,0})
    decorationContainers[4]=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,0})
    decorationContainerGrey=sim.addDrawingObject(sim.drawing_spherepoints,0.005,0,-1,9999,{0.1,0.1,0.1})

    allTrackedLocations=modelData['palletItems']
    currentCartridgeIndex=1
end

function sysCall_sensing()
    trackedLocations=allTrackedLocations[currentCartridgeIndex]
    local data=sim.readCustomDataBlock(trackingWindowModel,simBWF.modelTags.TRACKINGWINDOW)
    data=sim.unpackTable(data)
    data['itemsToRemoveFromTracking']={}

    local toIncrement=data['targetPositionsToMarkAsProcessed']
    local smallestStage=999
    for i=1,#toIncrement,1 do
        for j=1,#trackedLocations,1 do
            if trackedLocations[j]['dummyHandle']==toIncrement[i] then
                trackedLocations[j]['processingStage']=trackedLocations[j]['processingStage']+1
            end
            if trackedLocations[j]['processingStage']<smallestStage then
                smallestStage=trackedLocations[j]['processingStage']
            end
        end
    end
    data['targetPositionsToMarkAsProcessed']={}
    if 1==smallestStage then -- full
        local dat=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
        dat=sim.unpackTable(dat)
        dat['putCartridgeDown'][currentCartridgeIndex]=true
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,sim.packTable(dat))
        for i=1,#trackedLocations,1 do
            trackedLocations[i]['processingStage']=0
        end
        if currentCartridgeIndex==1 then
            currentCartridgeIndex=2
            trackedLocations=allTrackedLocations[2]
        else
            currentCartridgeIndex=1
            trackedLocations=allTrackedLocations[1]
        end
    end

    -- Now update the tracked targets:
    local trackedTargetsInWindow_currentLayer={}
    local trackedTargetsInWindow_otherLayers={}
    -- Check the lowest processing stage:
    local currStage=99
    for i=1,#trackedLocations,1 do
        if trackedLocations[i]['processingStage']<currStage then
            currStage=trackedLocations[i]['processingStage']
        end
    end

    -- Check the lowest layer of the lowest processing stage:
    local currLay=99
    for i=1,#trackedLocations,1 do
        if trackedLocations[i]['processingStage']==currStage then
            local l=trackedLocations[i]['layer']
            if l<currLay then
                currLay=l
            end
        end
    end
    -- Go through all items...
    for i=1,#trackedLocations,1 do
        local h=trackedLocations[i]['dummyHandle']
        -- ... that are on the same layer as the lowest layer:
        local dat={}
        dat['processingStage']=trackedLocations[i]['processingStage']
        dat['ser']=trackedLocations[i]['ser']
        dat['partHandle']=-1
        dat['velocityVect']={0,0,0}
        dat['partName']=locationName
        if trackedLocations[i]['layer']==currLay then
            -- Those items are being transmitted to robots
            trackedTargetsInWindow_currentLayer[h]=dat
        else
            -- items on other layers are just being displayed, but not transmitted:
            trackedTargetsInWindow_otherLayers[h]=dat
        end
    end

    local modelData=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    modelData=sim.unpackTable(modelData)
    if modelData['putCartridgeDown'][currentCartridgeIndex]==true then
        data['trackedTargetsInWindow']={} -- that cartridge is not yet ready again
    else
        data['trackedTargetsInWindow']=trackedTargetsInWindow_currentLayer
    end
    data['trackedItemsInWindow']={}
    sim.writeCustomDataBlock(trackingWindowModel,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))

    if showPoints then
        displayTargets(trackedTargetsInWindow_currentLayer)
        displayGreyTargets(trackedTargetsInWindow_otherLayers)
    end
end