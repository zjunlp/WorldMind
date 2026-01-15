function model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,partsData)
    local errString=nil
    local dropName=model.getDistributionValue(partDistribution)
    local thePartD=partsData[dropName]
    local partToDrop=nil
    if thePartD then
--        local destinationName=model.getDistributionValue(destinationDistribution)
        local dropShift=model.getDistributionValue(shiftDistribution)
        if not dropShift then dropShift={0,0,0} end
        dropShift[1]=dropShift[1]*distributionExtent[1]
        dropShift[2]=dropShift[2]*distributionExtent[2]
        dropShift[3]=dropShift[3]*distributionExtent[3]
        local dropRotation=model.getDistributionValue(rotationDistribution)
        local dropMass=model.getDistributionValue(massDistribution)
        local dropScaling=nil
        if scalingDistribution then
            dropScaling=model.getDistributionValue(scalingDistribution)
        end
        partToDrop={name=dropName,position=dropShift,rotation=dropRotation,mass=dropMass,scaling=dropScaling}
    else
        local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
        local msg="WARNING (run-time): Part '"..dropName.."' was not found in the part repository"..nm  
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
    return partToDrop,errString
end

function model.getDistributionValue(distribution)
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

function model.getSensorState()
    if model.sensorHandle>=0 then
        local data=sim.readCustomDataBlock(model.sensorHandle,simBWF.modelTags.BINARYSENSOR)
        if data then
            data=sim.unpackTable(data)
            return data['detectionState']
        end
        local data=sim.readCustomDataBlock(model.sensorHandle,simBWF.modelTags.OLDSTATICPICKWINDOW)
        if data then
            data=sim.unpackTable(data)
            return data['triggerState']
        end
    end
    return 0
end

function model.getConveyorDistanceTrigger()
    if model.conveyorHandle>=0 then
        local data=sim.readCustomDataBlock(model.conveyorHandle,simBWF.modelTags.CONVEYOR)
        if data then
            data=sim.unpackTable(data)
            local d=data['encoderDistance']
            if d then
                if not lastConveyorDistance then
                    lastConveyorDistance=d
                end
                if math.abs(lastConveyorDistance-d)>model.conveyorTriggerDist then
                    lastConveyorDistance=d
                    return true,d
                end
            end
        end
    end
    return false,d
end

function model.prepareStatisticsDialog(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="Part production count: 0" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

function model.updateStatisticsDialog(enabled)
    if statUi then
        simUI.setLabelText(statUi,1,"Part production count: "..model.productionCount,true)
    end
end

function model.wasMultiFeederTriggered()
    local data=model.readInfo()
    local val=data['multiFeederTriggerCnt']
    if val and val~=model.multiFeederTriggerLastState then
        model.multiFeederTriggerLastState=val
        return true
    end
    return false
end

function model.getStartStopTriggerType()
    if model.stopTriggerSensor~=-1 then
        local data=sim.readCustomDataBlock(model.stopTriggerSensor,simBWF.modelTags.BINARYSENSOR)
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
    if model.startTriggerSensor~=-1 then
        local data=sim.readCustomDataBlock(model.startTriggerSensor,simBWF.modelTags.BINARYSENSOR)
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

function model.manualTrigger_callback()
    manualTrigger=true
end

function sysCall_init()
    model.codeVersion=1

    local data=model.readInfo()
    model.maxProductionCnt=data.maxProductionCnt
    if model.maxProductionCnt==0 then
        model.maxProductionCnt=-1 -- means unlimited
    end

    model.online=simBWF.isSystemOnline()
    if model.online then

    else
        model.prepareStatisticsDialog(sim.boolAnd32(data['bitCoded'],128)>0)
        model.productionCount=0
        model.stopTriggerSensor=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)
        model.startTriggerSensor=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)
        model.sensorHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.SENSOR)
        model.conveyorHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
        model.conveyorTriggerDist=data['conveyorDist']
        model.mode=0 -- 0=frequency, 1=sensor, 2=user, 3=conveyor, 4=multi-feeder
        local tmp=sim.boolAnd32(data['bitCoded'],4+8+16)
        if tmp==4 then model.mode=1 end
        if tmp==8 then model.mode=2 end
        if tmp==12 then model.mode=3 end
        if tmp==16 then model.mode=4 end
        if tmp==20 then model.mode=5 end
        model.sensorLastState=0
        model.multiFeederTriggerLastState=0
        model.getStartStopTriggerType()
        local parts=simBWF.getAllPartsFromPartRepository()
        model.partsData={}
        if parts then
            for i=1,#parts,1 do
                local h=parts[i][2]
                local dat=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
                dat=sim.unpackTable(dat)
                dat['handle']=h
                model.partsData[simBWF.getObjectAltName(h)]=dat
            end
        else
            sim.addStatusbarMessage('\nWarning: no part repository found in the scene.\n')
        end
        model.allProducedParts={}
        model.timeForIdlePartToDeactivate=simBWF.modifyPartDeactivationTime(data['deactivationTime'])
        counter=0
        model.wasEnabled=false
        model.fromStartStopTriggerEnable=true
        model.fromStartStopTriggerWasEnable=true
        if model.mode==5 then
            local xml ='<button text="Trigger part production"  on-click="model.manualTrigger_callback" style="* {min-width: 220px; min-height: 50px;}"/>'

            model.manualTriggerUi=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,_MODELVERSION_,_CODEVERSION_),nil,false,"",false,false,false,"")
        end
    end
end

function sysCall_actuation()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()
    local data=model.readInfo()
    if model.online then
    
    
    else
        local distributionExtent=data['size']
        local dropFrequency=data['frequency']
        local feederAlgo=data['algorithm']
        local enabled=sim.boolAnd32(data['bitCoded'],2)>0
        local nothing
        if model.maxProductionCnt~=0 then
            if enabled then
                if not model.wasEnabled then
                    model.fromStartStopTriggerEnable=true
                    model.fromStartStopTriggerWasEnable=true
                    model.sensorLastState=model.getSensorState()
                    lastDropTime=nil
                    nothing,lastConveyorDistance=model.getConveyorDistanceTrigger()
                end
            end
            model.wasEnabled=enabled

            local trigger=model.getStartStopTriggerType()
            if enabled then
                if trigger>0 then
                    model.fromStartStopTriggerEnable=true
                end
                if trigger<0 then
                    model.fromStartStopTriggerEnable=false
                end
                if model.fromStartStopTriggerEnable and not model.fromStartStopTriggerWasEnable then
                    model.sensorLastState=model.getSensorState()
                    lastDropTime=nil
                    nothing,lastConveyorDistance=model.getConveyorDistanceTrigger()
                end
            end
            model.fromStartStopTriggerWasEnable=model.fromStartStopTriggerEnable
            
            if enabled and model.fromStartStopTriggerEnable then
                -- The feeder is enabled
                local partDistribution='{'..data['partDistribution']..'}'
                local f=loadstring("return "..partDistribution)
                partDistribution=f()
--                local destinationDistribution='{'..data['destinationDistribution']..'}'
--                local f=loadstring("return "..destinationDistribution)
--                destinationDistribution=f()
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

                local sensorState=model.getSensorState()
                local partToDrop=nil
                local errStr=nil
                local t=sim.getSimulationTime()
                if model.mode==0 then
                    -- Frequency triggered
                    if not lastDropTime then
                        lastDropTime=t-9999
                    end
                    if t-lastDropTime>(1/dropFrequency) then
                        lastDropTime=t
                        partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                    end
                end
                if model.mode==1 then
                    -- Sensor triggered
                    if sensorState~=model.sensorLastState then
                        partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                    end
                end
                if model.mode==3 and model.getConveyorDistanceTrigger() then
                    -- Conveyor belt distance triggered
                    partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                end
                if model.mode==2 then
                    -- User triggered
                    local algo=assert(loadstring(feederAlgo))
                    if algo() then
                        partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                    end
                end
                if model.mode==4 then
                    -- Multi-feeder triggered
                    if model.wasMultiFeederTriggered() then
                        partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                    end
                end
                if model.mode==5 then
                    -- Manually triggered
                    if manualTrigger then
                        manualTrigger=nil
                        lastDropTime=t
                        partToDrop,errStr=model.getPartToDrop(distributionExtent,partDistribution,shiftDistribution,rotationDistribution,massDistribution,scalingDistribution,model.partsData)
                    end
                end
                if errStr then
                    sim.addStatusbarMessage('\n'..errStr..'\n')
                end
                model.sensorLastState=sensorState
                if partToDrop then
                    if model.maxProductionCnt>0 then
                        model.maxProductionCnt=model.maxProductionCnt-1
                    end
                    counter=counter+1
                    local itemName=partToDrop.name
--                    local itemDestination=partToDrop[2]
                    local itemPosition=partToDrop.position
                    local itemOrientation=partToDrop.orientation
                    local itemMass=partToDrop.mass
                    local itemScaling=partToDrop.scaling
                    local dat=model.partsData[itemName]
                    if dat then
                        model.productionCount=model.productionCount+1
                        local h=dat['handle']
                        
--                        if itemDestination and itemDestination=='<DEFAULT>' then
--                            itemDestination=nil
--                        end
                        if not itemPosition then
                            itemPosition=sim.getObjectPosition(model.handle,-1) -- default
                        else
                            itemPosition=sim.multiplyVector(sim.getObjectMatrix(model.handle,-1),itemPosition)
                        end
                        if not itemOrientation then
                            itemOrientation=sim.getObjectOrientation(model.handle,-1) -- default
                        else
                            local m=sim.buildMatrix({0,0,0},itemOrientation)
                            m=sim.multiplyMatrices(sim.getObjectMatrix(model.handle,-1),m)
                            itemOrientation=sim.getEulerAnglesFromMatrix(m)
                        end
                        if itemMass and itemMass=='<DEFAULT>' then
                            itemMass=nil
                        end
                        local labelsToEnable=model.getDistributionValue(labelDistribution)
                        
                        simBWF.instanciatePart(h,itemPosition,itemOrientation,itemMass,itemScaling,labelsToEnable,true)
                    end
                end
            end
        end
        
        model.updateStatisticsDialog()
    end
end

