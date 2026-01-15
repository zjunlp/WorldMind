function getTriggerType()
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

function overrideMasterMotionIfApplicable(override)
    if masterConveyor>=0 then
        local data=sim.readCustomDataBlock(masterConveyor,simBWF.modelTags.CONVEYOR)
        if data then
            data=sim.unpackTable(data)
            local stopRequests=data['stopRequests']
            if override then
                stopRequests[model]=true
            else
                stopRequests[model]=nil
            end
            data['stopRequests']=stopRequests
            sim.writeCustomDataBlock(masterConveyor,simBWF.modelTags.CONVEYOR,sim.packTable(data))
        end
    end
end

function getMasterDeltaShiftIfApplicable()
    if masterConveyor>=0 then
        local data=sim.readCustomDataBlock(masterConveyor,simBWF.modelTags.CONVEYOR)
        if data then
            data=sim.unpackTable(data)
            local totalShift=data['encoderDistance']
            local retVal=totalShift
            if previousMasterTotalShift then
                retVal=totalShift-previousMasterTotalShift
            end
            previousMasterTotalShift=totalShift
            return retVal
        end
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    data=sim.unpackTable(data)
    stopTriggerSensor=simBWF.getReferencedObjectHandle(model,1)
    startTriggerSensor=simBWF.getReferencedObjectHandle(model,2)
    masterConveyor=simBWF.getReferencedObjectHandle(model,3)
    getTriggerType()
    path=sim.getObjectHandle('genericCurvedConveyorTypeA90_path')
    endPad1=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPad1')
    endPad2=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPad2')
    lastT=sim.getSimulationTime()
    beltVelocity=0
    totShift=0
end 

function sysCall_actuation()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    data=sim.unpackTable(data)
    maxVel=data['velocity']
    accel=data['acceleration']
    enabled=sim.boolAnd32(data['bitCoded'],64)>0
    if not enabled then
        maxVel=0
    end
    local stopRequests=data['stopRequests']
    local trigger=getTriggerType()
    if trigger>0 then
        stopRequests[model]=nil -- restart
    end
    if trigger<0 then
        stopRequests[model]=true -- stop
    end
    if next(stopRequests) then
        maxVel=0
        overrideMasterMotionIfApplicable(true)
    else
        overrideMasterMotionIfApplicable(false)
    end

    t=sim.getSimulationTime()
    dt=t-lastT
    lastT=t

    local masterDeltaShift=getMasterDeltaShiftIfApplicable()
    if masterDeltaShift then
        totShift=totShift+masterDeltaShift
        beltVelocity=masterDeltaShift/dt
    else
        local dv=maxVel-beltVelocity
        if math.abs(dv)>accel*dt then
            beltVelocity=beltVelocity+accel*dt*math.abs(dv)/dv
        else
            beltVelocity=maxVel
        end
        totShift=totShift+dt*beltVelocity
    end
    
    sim.setPathPosition(path,totShift)

    sim.setObjectFloatParameter(endPad1,sim.shapefloatparam_texture_y,totShift)
    sim.setObjectFloatParameter(endPad2,sim.shapefloatparam_texture_y,totShift)

    relativeLinearVelocity={0,beltVelocity,0}
    
    sim.resetDynamicObject(endPad1)
    m=sim.getObjectMatrix(endPad1,-1)
    m[4]=0
    m[8]=0
    m[12]=0
    absoluteLinearVelocity=sim.multiplyVector(m,relativeLinearVelocity)
    sim.setObjectFloatParameter(endPad1,sim.shapefloatparam_init_velocity_x,absoluteLinearVelocity[1])
    sim.setObjectFloatParameter(endPad1,sim.shapefloatparam_init_velocity_y,absoluteLinearVelocity[2])
    sim.setObjectFloatParameter(endPad1,sim.shapefloatparam_init_velocity_z,absoluteLinearVelocity[3])

    sim.resetDynamicObject(endPad2)
    m=sim.getObjectMatrix(endPad2,-1)
    m[4]=0
    m[8]=0
    m[12]=0
    absoluteLinearVelocity=sim.multiplyVector(m,relativeLinearVelocity)
    sim.setObjectFloatParameter(endPad2,sim.shapefloatparam_init_velocity_x,absoluteLinearVelocity[1])
    sim.setObjectFloatParameter(endPad2,sim.shapefloatparam_init_velocity_y,absoluteLinearVelocity[2])
    sim.setObjectFloatParameter(endPad2,sim.shapefloatparam_init_velocity_z,absoluteLinearVelocity[3])

end



