function model.getTriggerType()
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

function model.overrideMasterMotionIfApplicable(override)
    if model.masterConveyor>=0 then
        local data=sim.readCustomDataBlock(model.masterConveyor,simBWF.modelTags.THERMOFORMER) 
        if data then
            data=sim.unpackTable(data)
            local stopRequests=data['stopRequests']
            if override then
                stopRequests[model.handle]=true
            else
                stopRequests[model.handle]=nil
            end
            data['stopRequests']=stopRequests
            sim.writeCustomDataBlock(model.masterConveyor,simBWF.modelTags.THERMOFORMER,sim.packTable(data))
        end
    end
end

function model.getMasterDeltaShiftIfApplicable()
    if model.masterConveyor>=0 then
        local data=sim.readCustomDataBlock(model.masterConveyor,simBWF.modelTags.THERMOFORMER) 
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
    model.codeVersion=1
    
    local data=model.readInfo()
    model.stopTriggerSensor=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)
    model.startTriggerSensor=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)
    model.masterConveyor=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.MASTERCONVEYOR)
    model.outputboxHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUTBOX)
    model.getTriggerType()
    model.length=data['length']
    model.height=data['height']
    model.lastT=sim.getSimulationTime()
    model.beltVelocity=0
    model.totShift=0
    model.online=simBWF.isSystemOnline()
    model.runningStateInSimulation=false
    if model.online then
        local data={}
        data.id=model.handle
        simBWF.query('conveyor_getState',data) -- just to reset from last call
    end
end 

function sysCall_actuation()
    local t=sim.getSimulationTime()
    local dt=t-model.lastT
    model.lastT=t
    local data=sim.readCustomDataBlock(model.handle,simBWF.modelTags.THERMOFORMER)
    data=sim.unpackTable(data)
    if model.online then
        local ds=0
        local data={}
        data.id=model.handle
        
        local res,retData=simBWF.query('conveyor_getState',data)
        if res=='ok' then
            ds=retData.displacement
        else
            if simBWF.isInTestMode() then
                ds=0.001
            end
        end
        model.totShift=model.totShift+ds
    else
        local maxVel=data['velocity']
        local accel=data['acceleration']
        if not model.runningStateInSimulation then
            maxVel=0
        end
        local stopRequests=data['stopRequests']
        local trigger=model.getTriggerType()
        if trigger>0 then
            stopRequests[model.handle]=nil -- restart
        end
        if trigger<0 then
            stopRequests[model.handle]=true -- stop
        end
        if next(stopRequests) then
            maxVel=0
            model.overrideMasterMotionIfApplicable(true)
        else
            model.overrideMasterMotionIfApplicable(false)
        end

        local masterDeltaShift=model.getMasterDeltaShiftIfApplicable()
        if masterDeltaShift then
            model.totShift=model.totShift+masterDeltaShift
            model.beltVelocity=masterDeltaShift/dt
        else
            local dv=maxVel-model.beltVelocity
            if math.abs(dv)>accel*dt then
                model.beltVelocity=model.beltVelocity+accel*dt*math.abs(dv)/dv
            else
                model.beltVelocity=maxVel
            end
            model.totShift=model.totShift+dt*model.beltVelocity
        end


        local data={}
        data.id=model.handle
        data.displacement=model.totShift
        simBWF.query('conveyor_state',data)
        
        model.runningStateInSimulation=true -- running by default
        if model.outputboxHandle>=0 then
            model.runningStateInSimulation=simBWF.callCustomizationScriptFunction("model.ext.getState",model.outputboxHandle)
        end
        --[[
        local res,retData=simBWF.query('conveyor_state',data)
        if retData then
            model.runningStateInSimulation=retData.runningState
            if simBWF.isInTestMode() then
                model.runningStateInSimulation=true -- testing
            end
        end
        --]]
    end
    model.beltVelocity=0
    if model.previousTotShift then
        model.beltVelocity=(model.totShift-model.previousTotShift)/dt
    end
    
    updateConveyorForMotion(dt)
    
    data['encoderDistance']=model.totShift
    model.previousTotShift=model.totShift

    model.writeInfo(data)
end

function model.thermoformer_actuation()
    local t=sim.getSimulationTime()
    local dt=t-model.lastT
    model.lastT=t
    local data=model.readInfo()
--    local data=sim.readCustomDataBlock(model.handle,simBWF.modelTags.THERMOFORMER)
--    data=sim.unpackTable(data)
    local stationDisplacement=nil
    if model.online then
        local ds=0
        local data={}
        data.id=model.handle
        
        local res,retData=simBWF.query('conveyor_getState',data)
        if res=='ok' then
            ds=retData.displacement
            stationDisplacement=retData.stationDisplacement
        else
            if simBWF.isInTestMode() then
                ds=0.001
            end
        end
        model.totShift=model.totShift+ds
    else
        local theShift=0
        if model.rmlObject==nil then
            if model.dwellTime then
                model.dwellTime=model.dwellTime-sim.getSimulationTimeStep()
                if model.dwellTime<=0 then
                    model.dwellTime=nil
                    stationDisplacement=model.totShift+theShift
                end
            else
                local maxVel=data.velocity
                local accel=data.acceleration
                local displ=data.thermo_rowColCnt[1]*data.thermo_rowColStep[1]+data.thermo_stationSpacing
                model.rmlObject=sim.rmlPos(1,0.0001,-1,{0,0,0},{maxVel,accel,0},{1},{displ,0,0})
                model.previousShift=0
            end
        else
            local result,newPosVelAccel=sim.rmlStep(model.rmlObject,sim.getSimulationTimeStep())
            if result~=-1 then
                theShift=newPosVelAccel[1]-model.previousShift
                model.previousShift=newPosVelAccel[1]
            end
            if result==1 or result==-1 then
                sim.rmlRemove(model.rmlObject)
                model.rmlObject=nil
                if data.thermo_dwellTime>0 then
                    model.dwellTime=data.thermo_dwellTime
                else
                    stationDisplacement=model.totShift+theShift
                end
            end
        end
        
        model.totShift=model.totShift+theShift
        
        
    end
    
    if stationDisplacement then
        model.displaceBoxes(stationDisplacement)
        model.newStationReached()
        simBWF.callCustomizationScriptFunction("model.ext.createOriginPallet",model.handle,stationDisplacement)
    end
    model.displaceBoxes(model.totShift)
    
--    model.beltVelocity=0
--    if model.previousTotShift then
--        model.beltVelocity=(model.totShift-model.previousTotShift)/dt
--    end
    
    
    data['encoderDistance']=model.totShift
    model.previousTotShift=model.totShift

    model.writeInfo(data)
end

function model.displaceBoxes(totShift)
    local objs=sim.getObjectsInTree(model.specHandles.boxes,sim.handle_all,1)
    for i=1,#objs,1 do
        local d=sim.readCustomDataBlock(objs[i],'thermoformerOpenBox')
        d=sim.unpackTable(d)
        local p=sim.getObjectPosition(objs[i],model.handle)
        p[1]=d.initX+totShift
        sim.setObjectPosition(objs[i],model.handle,p)
    end
end

function model.newStationReached()
    local data=model.readInfo()
    local objs=sim.getObjectsInTree(model.specHandles.boxes,sim.handle_all,1)
    local bs={data.thermo_extrusionSize[1]+data.thermo_wallThickness*2,data.thermo_extrusionSize[2]+data.thermo_wallThickness*2,data.thermo_extrusionSize[3]+data.thermo_wallThickness}
    for i=1,#objs,1 do
        local d=sim.readCustomDataBlock(objs[i],'thermoformerOpenBox')
        d=sim.unpackTable(d)
        d.stationIndex=d.stationIndex+1
        sim.writeCustomDataBlock(objs[i],'thermoformerOpenBox',sim.packTable(d))
        if d.stationIndex>=data.thermo_stationCnt-1 then
            local h=sim.createPureShape(0,8,bs,0.1)
            sim.setObjectPosition(h,objs[i],{0,0,0})
            sim.setObjectOrientation(h,objs[i],{0,0,0})
            sim.setShapeColor(objs[i],nil,sim.colorcomponent_ambient_diffuse,data.thermo_color)
            sim.removeObject(objs[i])

            local partData=simBWF.readPartInfo(h)
            local itemPosition=sim.getObjectPosition(h,-1)
            local itemOrientation=sim.getObjectOrientation(h,-1)
            simBWF.writePartInfo(h,partData)
            simBWF.instanciatePart(h,itemPosition,itemOrientation,nil,nil,nil,false)
        end
    end
end


