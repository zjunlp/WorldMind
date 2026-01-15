function getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
    local errString=nil
    local dropName=getDistributionValue(partDistribution)
    local thePartD=partsData[dropName]
    local partToDrop=nil
    if thePartD then
        local destinationName=getDistributionValue(destinationDistribution)
        local dropShift=getDistributionValue(shiftDistribution)
        if not dropShift then dropShift={0,0,0} end
        dropShift[1]=dropShift[1]*distributionExtent[1]
        dropShift[2]=dropShift[2]*distributionExtent[2]
        dropShift[3]=dropShift[3]*distributionExtent[3]
        local dropRotation=getDistributionValue(rotationDistribution)
        local dropMass=getDistributionValue(massDistribution)
        local dropScaling=nil
        if scalingDistribution then
            dropScaling=getDistributionValue(scalingDistribution)
        end
        partToDrop={dropName,destinationName,dropShift,dropRotation,dropMass,dropScaling}
    else
        errString="Warning: part '"..dropName.."' was not found in the part repository (or part repository was not found)."
    end
    return partToDrop,errString
end

setItemMass=function(handle,m)
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

deactivatePart=function(handle,isModel)
    if isModel then
        local p=sim.getModelProperty(handle)
        p=sim.boolOr32(p,sim.modelproperty_not_dynamic)
        sim.setModelProperty(handle,p)
    else
        sim.setObjectInt32Parameter(handle,sim.shapeintparam_static,1) -- we make it static now!
    end
    sim.resetDynamicObject(handle) -- important, otherwise the dynamics engine doesn't notice the change!
end

removePart=function(handle,isModel)
    if isModel then
        sim.removeModel(handle)
    else
        sim.removeObject(handle)
    end
end

getDistributionValue=function(distribution)
    -- Distribution string could be:
    -- {} --> returns nil
    -- {{}} --> returns nil
    -- a,a,b,c --> returns a,b, or c
    -- {{2,a},{1,b},{1,c}} --> returns a,b, or c
    if #distribution>0 then
        if (type(distribution[1])~='table') or (#distribution[1]>0) then
            if (type(distribution[1])=='table') and (#distribution[1]==2) then
                local cnt=0
                for i=1,#distribution,1 do
                   cnt=cnt+distribution[i][1] 
                end
                local p=sim.getFloatParameter(sim.floatparam_rand)*cnt
                cnt=0
                for i=1,#distribution,1 do
                    if cnt+distribution[i][1]>=p then
                        return distribution[i][2]
                    end
                    cnt=cnt+distribution[i][1] 
                end
            else
                local cnt=#distribution
                local p=1+math.floor(sim.getFloatParameter(sim.floatparam_rand)*cnt-0.0001)
                return distribution[p]
            end
        end
    end
end

getLabels=function(partH)
    -- There can be up to 3 labels in this part:
    local possibleLabels=sim.getObjectsInTree(partH,sim.object_shape_type,1)
    local labels={}
    for objInd=1,#possibleLabels,1 do
        local h=possibleLabels[objInd]
        local data=sim.readCustomDataBlock(h,'XYZ_PARTLABEL_INFO')
        if data then
            labels[#labels+1]=h
        end
    end
    return labels
end

adjustSizeData=function(partH,sx,sy,sz)
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

regenerateOrRemoveLabels=function(partH,enabledLabels)
    -- There can be up to 3 labels in this part:
    local possibleLabels=sim.getObjectsInTree(partH,sim.object_shape_type,1)
    local labelData=sim.unpackTable(sim.readCustomDataBlock(partH,simBWF.modelTags.PART))['labelData']
    for ind=1,3,1 do
        for objInd=1,#possibleLabels,1 do
            local h=possibleLabels[objInd]
            if h>=0 then
                local data=sim.readCustomDataBlock(h,'XYZ_PARTLABEL_INFO')
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

makeInvisibleOrNonRespondableToOtherParts=function(handle,invisible,nonRespondableToOtherParts)
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
            sim.setObjectInt32Parameter(objs[i],sim.shapeintparam_respondable_mask,sim.boolOr32(m,65280)-32512)
        else
            sim.setObjectInt32Parameter(objs[i],sim.shapeintparam_respondable_mask,sim.boolOr32(m,65280)-32768)
        end
    end
end

getSensorState=function()
    if sensorHandle>=0 then
        local data=sim.readCustomDataBlock(sensorHandle,'XYZ_BINARYSENSOR_INFO')
        if data then
            data=sim.unpackTable(data)
            return data['detectionState']
        end
        local data=sim.readCustomDataBlock(sensorHandle,'XYZ_STATICPICKWINDOW_INFO')
        if data then
            data=sim.unpackTable(data)
            return data['triggerState']
        end
        local data=sim.readCustomDataBlock(sensorHandle,simBWF.modelTags.OLDSTATICPLACEWINDOW)
        if data then
            data=sim.unpackTable(data)
            return data['triggerState']
        end
    end
    return 0
end

getConveyorDistanceTrigger=function()
    if conveyorHandle>=0 then
        local data=sim.readCustomDataBlock(conveyorHandle,simBWF.modelTags.CONVEYOR)
        if data then
            data=sim.unpackTable(data)
            local d=data['encoderDistance']
            if d then
                if not lastConveyorDistance then
                    lastConveyorDistance=d
                end
                if math.abs(lastConveyorDistance-d)>conveyorTriggerDist then
                    lastConveyorDistance=d
                    return true,d
                end
            end
        end
    end
    return false,d
end

prepareStatisticsDialog=function(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="Part production count: 0" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

updateStatisticsDialog=function(enabled)
    if statUi then
        simUI.setLabelText(statUi,1,"Part production count: "..productionCount,true)
    end
end

wasMultiFeederTriggered=function()
    local data=sim.unpackTable(sim.readCustomDataBlock(model,simBWF.modelTags.PARTFEEDER))
    local val=data['multiFeederTriggerCnt']
    if val and val~=multiFeederTriggerLastState then
        multiFeederTriggerLastState=val
        return true
    end
    return false
end

function getStartStopTriggerType()
    if stopTriggerSensor~=-1 then
        local data=sim.readCustomDataBlock(stopTriggerSensor,'XYZ_BINARYSENSOR_INFO')
        if data then
            data=sim.unpackTable(data)
            local state=data['detectionState']
            if not lastStopTriggerState then
                lastStopTriggerState=state
            end
            if lastStopTriggerState~=state then
                lastStopTriggerState=state
                return -1 -- means stop
            end
        end
    end
    if startTriggerSensor~=-1 then
        local data=sim.readCustomDataBlock(startTriggerSensor,'XYZ_BINARYSENSOR_INFO')
        if data then
            data=sim.unpackTable(data)
            local state=data['detectionState']
            if not lastStartTriggerState then
                lastStartTriggerState=state
            end
            if lastStartTriggerState~=state then
                lastStartTriggerState=state
                return 1 -- means restart
            end
        end
    end
    return 0
end

function manualTrigger_callback()
    manualTrigger=true
end
function getBaseAndParts(h)
    local otherModels={}
    local objs=sim.getObjectsInTree(h,sim.handle_all,1)
    for i=1,#objs,1 do
        local o=objs[i]
        local p=sim.getModelProperty(o)
        local isModel=sim.boolAnd32(p,sim.modelproperty_not_model)==0
        if isModel then
            otherModels[#otherModels+1]=o
        end
    end
    if #otherModels>0 then
        local baseParts={}
        local toParse={h}
        while #toParse>0 do
            local o=toParse[1]
            table.remove(toParse,1)
            baseParts[#baseParts+1]=o
            i=0
            while true do
                local child=sim.getObjectChild(o,i)
                if child<0 then
                    break
                end
                local p=sim.getModelProperty(child)
                local isModel=sim.boolAnd32(p,sim.modelproperty_not_model)==0
                if not isModel then
                    toParse[#toParse+1]=child
                end
                i=i+1
            end
        end
        return baseParts,otherModels
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    producedPartsDummy=sim.getObjectHandle('genericFeeder_ownedParts')
    smallLabel=sim.getObjectHandle('genericFeeder_smallLabel')
    largeLabel=sim.getObjectHandle('genericFeeder_largeLabel')
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTFEEDER)
    data=sim.unpackTable(data)
    prepareStatisticsDialog(sim.boolAnd32(data['bitCoded'],128)>0)
    productionCount=0
    stopTriggerSensor=simBWF.getReferencedObjectHandle(model,3)
    startTriggerSensor=simBWF.getReferencedObjectHandle(model,4)
    sensorHandle=simBWF.getReferencedObjectHandle(model,1)
    conveyorHandle=simBWF.getReferencedObjectHandle(model,2)
    conveyorTriggerDist=data['conveyorDist']
    mode=0 -- 0=frequency, 1=sensor, 2=user, 3=conveyor, 4=multi-feeder
    local tmp=sim.boolAnd32(data['bitCoded'],4+8+16)
    if tmp==4 then mode=1 end
    if tmp==8 then mode=2 end
    if tmp==12 then mode=3 end
    if tmp==16 then mode=4 end
        if tmp==20 then mode=5 end
    sensorLastState=0
    multiFeederTriggerLastState=0
    getStartStopTriggerType()

    local parts=simBWF.getAllPartsFromPartRepositoryV0()
    partsData={}
    if parts then
        for i=1,#parts,1 do
            local h=parts[i][2]
            local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
            data=sim.unpackTable(data)
            data['handle']=h
            partsData[data['name']]=data
        end
    else
        sim.addStatusbarMessage('\nWarning: no part repository found in the scene.\n')
    end
    allProducedParts={}
    timeForIdlePartToDeactivate=simBWF.modifyPartDeactivationTime(data['deactivationTime'])
    counter=0
    wasEnabled=false
    fromStartStopTriggerEnable=true
    fromStartStopTriggerWasEnable=true
    if mode==5 then
    local xml ='<button text="Trigger '..simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_)..'" on-click="manualTrigger_callback" style="* {min-width: 220px; min-height: 50px;}"/>'

    manualTriggerUi=simBWF.createCustomUi(xml,"Manual trigger",nil,false,"",false,false,false,"")
    end
end

function sysCall_actuation()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()

    local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTFEEDER)
    data=sim.unpackTable(data)
    local distributionExtent={data['length'],data['width'],data['height']}
    local dropFrequency=data['frequency']
    local feederAlgo=data['algorithm']
    local enabled=sim.boolAnd32(data['bitCoded'],2)>0
    if enabled then
        if not wasEnabled then
            fromStartStopTriggerEnable=true
            fromStartStopTriggerWasEnable=true
            sensorLastState=getSensorState()
            lastDropTime=nil
            nothing,lastConveyorDistance=getConveyorDistanceTrigger()
        end
    end
    wasEnabled=enabled

    local trigger=getStartStopTriggerType()
    if enabled then
        if trigger>0 then
            fromStartStopTriggerEnable=true
        end
        if trigger<0 then
            fromStartStopTriggerEnable=false
        end
        if fromStartStopTriggerEnable and not fromStartStopTriggerWasEnable then
            sensorLastState=getSensorState()
            lastDropTime=nil
            nothing,lastConveyorDistance=getConveyorDistanceTrigger()
        end
    end
    fromStartStopTriggerWasEnable=fromStartStopTriggerEnable
    
    if enabled and fromStartStopTriggerEnable then
        -- The feeder is enabled
        local partDistribution='{'..data['partDistribution']..'}'
        local f=loadstring("return "..partDistribution)
        partDistribution=f()
        local destinationDistribution='{'..data['destinationDistribution']..'}'
        local f=loadstring("return "..destinationDistribution)
        destinationDistribution=f()
        local shiftDistribution='{'..data['shiftDistribution']..'}'
        local f=loadstring("return "..shiftDistribution)
        shiftDistribution=f()
        local rotationDistribution='{'..data['rotationDistribution']..'}'
        local f=loadstring("return "..rotationDistribution)
        rotationDistribution=f()
        local massDistribution='{'..data['weightDistribution']..'}'
        local f=loadstring("return "..massDistribution)
        massDistribution=f()
        local labelDistribution='{'..data['labelDistribution']..'}'
        local f=loadstring("return "..labelDistribution)
        labelDistribution=f()

        local scalingDistribution=nil
        if data['sizeScaling'] and data['sizeScaling']>0 then
            if data['sizeScaling']==1 then
                scalingDistribution='{'..data['isoSizeScalingDistribution']..'}'
            end
            if data['sizeScaling']==2 then
                scalingDistribution='{'..data['nonIsoSizeScalingDistribution']..'}'
            end
            local f=loadstring("return "..scalingDistribution)
            scalingDistribution=f()
        end

        if true then --feederAlgo then
            local sensorState=getSensorState()
            local partToDrop=nil
            local errStr=nil
            local t=sim.getSimulationTime()
            if mode==0 then
                -- Frequency triggered
                if not lastDropTime then
                    lastDropTime=t-9999
                end
                if t-lastDropTime>(1/dropFrequency) then
                    lastDropTime=t
                    partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
                end
            end
            if mode==1 then
                -- Sensor triggered
                if sensorState~=sensorLastState then
                    partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
                end
            end
            if mode==3 and getConveyorDistanceTrigger() then
                -- Conveyor belt distance triggered
                partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
            end
            if mode==2 then
                -- User triggered
                local algo=assert(loadstring(feederAlgo))
                if algo() then
                    partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
                end
            end
            if mode==4 then
                -- Multi-feeder triggered
                if wasMultiFeederTriggered() then
                    partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
                end
            end
                if mode==5 then
                    -- Manually triggered
                    if manualTrigger then
                        manualTrigger=nil
                        lastDropTime=t
                        partToDrop,errStr=getPartToDrop(distributionExtent,partDistribution,destinationDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
                    end
                end
            if errStr then
                sim.addStatusbarMessage('\n'..errStr..'\n')
            end
            sensorLastState=sensorState
            if partToDrop then
                counter=counter+1
                local itemName=partToDrop[1]
                local itemDestination=partToDrop[2]
                local itemPosition=partToDrop[3]
                local itemOrientation=partToDrop[4]
                local itemMass=partToDrop[5]
                local itemScaling=partToDrop[6]
                local dat=partsData[itemName]
                if dat then
                    productionCount=productionCount+1
                    local h=dat['handle']
                    local hmitems,itemsOnBase=getBaseAndParts(h)
                    if not hmitems then
                        local p=sim.getModelProperty(h)
                        local isModel=sim.boolAnd32(p,sim.modelproperty_not_model)==0
                        local tble
                        if isModel then
                            tble=sim.copyPasteObjects({h},1)
                        else
                            tble=sim.copyPasteObjects({h},0)
                        end
                        h=tble[1]
                        sim.setObjectParent(h,producedPartsDummy,true)
                        local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
                        data=sim.unpackTable(data)
                        local invisible=sim.boolAnd32(data['bitCoded'],1)>0
                        local nonRespondableToOtherParts=sim.boolAnd32(data['bitCoded'],2)>0
                        makeInvisibleOrNonRespondableToOtherParts(h,invisible,nonRespondableToOtherParts)
                        
                        -- Destination:
                        if itemDestination and itemDestination~='<DEFAULT>' then
                            data['destination']=itemDestination
                        end

                        -- Size scaling:
                        if itemScaling then
                            local itemLabels=getLabels(h)
                            for j=1,#itemLabels,1 do
                                sim.setObjectParent(itemLabels[j],-1,true)
                            end
                            if type(itemScaling)~='table' then
                                -- iso-scaling
                                adjustSizeData(h,itemScaling,itemScaling,itemScaling)
                                sim.scaleObjects({h},itemScaling,false)
                            else
                                -- non-iso-scaling
                                adjustSizeData(h,itemScaling[1],itemScaling[2],itemScaling[3])
                                if isModel then
                                    if sim.canScaleModelNonIsometrically(h,itemScaling[1],itemScaling[2],itemScaling[3]) then
                                        sim.scaleModelNonIsometrically(h,itemScaling[1],itemScaling[2],itemScaling[3])
                                    end
                                else
                                    if sim.canScaleObjectNonIsometrically(h,itemScaling[1],itemScaling[2],itemScaling[3]) then
                                        sim.scaleObject(h,itemScaling[1],itemScaling[2],itemScaling[3],0)
                                    end
                                end
                            end
                            for j=1,#itemLabels,1 do
                                sim.setObjectParent(itemLabels[j],h,true)
                            end
                        end
                        
                        -- Mass:
                        if itemMass and itemMass~='<DEFAULT>' then
                            setItemMass(h,itemMass)
                        end

                       -- Labels:
                        local enabledLabels=getDistributionValue(labelDistribution)
                        if invisible then
                            enabledLabels=0
                        end
                        if enabledLabels and enabledLabels>=0 then
                            regenerateOrRemoveLabels(h,enabledLabels)
                        end
                        if not itemPosition then
                            itemPosition={0,0,0} -- default
                        end
                        if not itemOrientation then
                            itemOrientation={0,0,0} -- default
                        end

                        -- Position:
                        sim.setObjectPosition(h,model,itemPosition)

                        -- Orientation:
                        sim.setObjectOrientation(h,model,itemOrientation)

                        data['instanciated']=true
                        sim.writeCustomDataBlock(h,simBWF.modelTags.PART,sim.packTable(data))
                        
                        p=sim.getObjectPosition(h,-1)
                        local partData={h,t,p,isModel,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
                        allProducedParts[#allProducedParts+1]=partData
                    else
                        -- Here we have a base part, with several items on top to pick:
                        -- First the base part:
                        local tble
                        tble=sim.copyPasteObjects(hmitems,0)
                        h=tble[1]
                        local p=sim.getModelProperty(h)
                        local isModel=sim.boolAnd32(p,sim.modelproperty_not_model)==0
                        sim.setObjectParent(h,producedPartsDummy,true)
                        if not itemPosition then
                            itemPosition={0,0,0} -- default
                        end
                        if not itemOrientation then
                            itemOrientation={0,0,0} -- default
                        end
                        sim.setObjectPosition(h,model,itemPosition)
                        sim.setObjectOrientation(h,model,itemOrientation)
                        local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
                        data=sim.unpackTable(data)
                        data['instanciated']=true
                        sim.writeCustomDataBlock(h,simBWF.modelTags.PART,'')
                        sim.writeCustomDataBlock(h,'XYZ_FEEDERPARTDUMMY_INFO',sim.packTable(data))
                        
                        p=sim.getObjectPosition(h,-1)
                        local partData={h,t,p,isModel,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
                        allProducedParts[#allProducedParts+1]=partData
                        -- Now the other items:
                        for cci=1,#itemsOnBase,1 do
                            local ih=itemsOnBase[cci]
                            local tble
                            tble=sim.copyPasteObjects({ih},1) -- is always a model
                            iih=tble[1]
                            sim.setObjectParent(h,producedPartsDummy,true)
                            local invisible=sim.boolAnd32(data['bitCoded'],1)>0
                            local nonRespondableToOtherParts=sim.boolAnd32(data['bitCoded'],2)>0
                            makeInvisibleOrNonRespondableToOtherParts(h,invisible,nonRespondableToOtherParts)
                            if itemDestination and itemDestination~='<DEFAULT>' then
                                data['destination']=itemDestination
                            end
                            -- No scaling in this case!
                            -- No other mass in this case!
                            -- No labels in this case!
                            local mm=sim.getObjectMatrix(ih,hmitems[1])
                            sim.setObjectMatrix(iih,h,mm)
                            data['instanciated']=true
                            sim.writeCustomDataBlock(iih,simBWF.modelTags.PART,sim.packTable(data))
                            p=sim.getObjectPosition(iih,-1)
                            local partData={iih,t,p,true,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
                            allProducedParts[#allProducedParts+1]=partData
                        end
                    end
                end
            end
        end
    end

    i=1
    while i<=#allProducedParts do
        local h=allProducedParts[i][1]
        if sim.isHandleValid(h)>0 then
            local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
            if not data then
                data=sim.readCustomDataBlock(h,'XYZ_FEEDERPARTDUMMY_INFO')
            end
            data=sim.unpackTable(data)
            local p=sim.getObjectPosition(h,-1)
            if allProducedParts[i][5] then
                -- The part is still active
                local deactivate=data['deactivate']
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
                    deactivatePart(h,allProducedParts[i][4])
                    allProducedParts[i][5]=false
                end
            end
            -- Does it want to be destroyed?
            if data['destroy'] or p[3]<-1000 or data['giveUpOwnership'] then
                if not data['giveUpOwnership'] then
                    removePart(h,allProducedParts[i][4])
                end
                table.remove(allProducedParts,i)
            else
                i=i+1
            end
        else
            table.remove(allProducedParts,i)
        end
    end
    updateStatisticsDialog()
end 