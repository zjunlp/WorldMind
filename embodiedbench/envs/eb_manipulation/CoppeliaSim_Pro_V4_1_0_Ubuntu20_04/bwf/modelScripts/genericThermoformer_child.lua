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

getPartMass=function(part)
    local currentMass=0
    local objects={part}
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
    return currentMass
end

function isPartDetected(partHandle)
    local shapesToTest={}
    if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_model)>0 then
        -- We have a single shape which is not a model. Is the shape detectable?
        if sim.boolAnd32(sim.getObjectSpecialProperty(partHandle),sim.objectspecialproperty_detectable_all)>0 then
            shapesToTest[1]=partHandle -- yes, it is detectable
        end
    else
        -- We have a model. Does the model have the detectable flags overridden?
        if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_detectable)==0 then
            -- No, now take all model shapes that are detectable:
            local t=sim.getObjectsInTree(partHandle,sim.object_shape_type,0)
            for i=1,#t,1 do
                if sim.boolAnd32(sim.getObjectSpecialProperty(t[i]),sim.objectspecialproperty_detectable_all)>0 then
                    shapesToTest[#shapesToTest+1]=t[i]
                end
            end
        end
    end
    for i=1,#shapesToTest,1 do
        if sim.checkProximitySensor(sensor,shapesToTest[i])>0 then
            return true
        end
    end
    return false
end

function getAllParts()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    for i=1,#l,1 do
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
        if isInstanciated then
            if not data['thermoformingPart'] then
                retL[#retL+1]=l[i]
            end
        end
    end
    return retL
end

checkSensor=function()
    local p=getAllParts()
    for i=1,#p,1 do
        if isPartDetected(p[i]) then
            return p[i]
        end
    end
    return -1
end

handlePartAtLocation=function(h)
    local wrappingMass=sim.getShapeMassAndInertia(h)
    sim.setObjectPosition(sensor,h,{0,0,-depth})
    local part=checkSensor()
    local mass=0
    if part>=0 then
        mass=getPartMass(part)
        sim.removeObject(part)
    end
    local s=sim.createPureShape(0,8,{width,length,depth},mass+wrappingMass)
    sim.setObjectPosition(s,h,{0,0,0})
    sim.setObjectOrientation(s,h,{0,0,0})
    sim.setShapeColor(s,'',sim.colorcomponent_ambient_diffuse,color)
    sim.setObjectInt32Parameter(s,sim.objintparam_visibility_layer,1+256)
    sim.setObjectSpecialProperty(s,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
    local p=sim.getObjectProperty(s)
    p=sim.boolOr32(p,sim.objectproperty_dontshowasinsidemodel)
    sim.setObjectProperty(s,p)
    local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
    data=sim.unpackTable(data)
    data['palletPattern']=0 -- none
    sim.writeCustomDataBlock(s,simBWF.modelTags.PART,sim.packTable(data))
    sim.setObjectParent(s,partHolder,true)
    local partData={s,sim.getSimulationTime(),sim.getObjectPosition(s,-1),false,true} -- handle, lastMovingTime, lastPosition, isModel, isActive
    allProducedParts[#allProducedParts+1]=partData
    sim.removeObject(h)
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    data=sim.unpackTable(data)
    stopTriggerSensor=simBWF.getReferencedObjectHandle(model,1)
    startTriggerSensor=simBWF.getReferencedObjectHandle(model,2)
    local columns=data['columns']
    local columnStep=data['columnStep']
    local palletSpacing=data['pullLength']-columns*columnStep
    movementDist=columns*columnStep+palletSpacing
    cuttingStationIndex=data['cuttingStationIndex']
    getTriggerType()
    width=data['extrusionWidth']
    length=data['extrusionLength']
    depth=data['extrusionDepth']
    color=data['color']
    dwellTime=data['dwellTime']
    timeForIdlePartToDeactivate=simBWF.modifyPartDeactivationTime(data['deactivationTime'])
    sampleHolder=sim.getObjectHandle('genericThermoformer_sampleHolder')
    partHolder=sim.getObjectHandle('genericThermoformer_partHolder')
    sensor=sim.getObjectHandle('genericThermoformer_sensor')
    beltVelocity=0
    totShift=0
    movementUnderway=false
    dwellStart=0
    partsToMove={} -- the static parts (open boxes)
    allProducedParts={} -- the dynamic parts (boxes)
end 

function sysCall_actuation()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()
    if movementUnderway then
        local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        local enabled=sim.boolAnd32(data['bitCoded'],64)>0
        local stopRequests=data['stopRequests']
        local trigger=getTriggerType()
        if trigger>0 then
            stopRequests[model]=nil -- restart
        end
        if trigger<0 then
            stopRequests[model]=true -- stop
        end
        if next(stopRequests) then
            enabled=false
        end
        if enabled then
            if not rmlPosObj then
                posVelAccel={posVelAccel[1],0,0}
                rmlPosObj=sim.rmlPos(1,0.0001,-1,posVelAccel,{maxVel,accel,999999},{1},{movementDist,0})
            end
            res,posVelAccel=sim.rmlStep(rmlPosObj,dt)
            if res>0 then
                sim.rmlRemove(rmlPosObj)
                rmlPosObj=nil
                movementUnderway=false
                dwellStart=t
            end
        else
            if rmlPosObj then
                sim.rmlRemove(rmlPosObj)
                rmlPosObj=nil
                rmlVelObj=sim.rmlVel(1,0.0001,-1,posVelAccel,{accel,999999},{1},{0})
            end
            if rmlVelObj then
                res,posVelAccel=sim.rmlStep(rmlVelObj,dt)
                if res>0 then
                    sim.rmlRemove(rmlVelObj)
                    rmlVelObj=nil
                end
            end
        end

        local dShift=posVelAccel[1]-lastRelPos
        lastRelPos=posVelAccel[1]
        totShift=totShift+dShift
        for i=1,#partsToMove,1 do
            local p=sim.getObjectPosition(partsToMove[i],model)
            p[2]=p[2]+dShift
            sim.setObjectPosition(partsToMove[i],model,p)
        end
        local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        data['encoderDistance']=totShift
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,sim.packTable(data))

    else
        local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        enabled=sim.boolAnd32(data['bitCoded'],64)>0
        if enabled then
            if t-dwellStart>dwellTime then
                local i=1
                while i<=#partsToMove do
                    local h=partsToMove[i]
                    local p=sim.getObjectPosition(h,sampleHolder)
                    if p[2]>(cuttingStationIndex-1.5)*movementDist then
                        table.remove(partsToMove,i)
                        local h2=handlePartAtLocation(h)
                        sim.setObjectInt32Parameter(h,sim.shapeintparam_static,0)
                    else
                        i=i+1
                    end
                end


                local samples=sim.getObjectsInTree(sampleHolder,sim.handle_all,1)
                local objects=sim.copyPasteObjects(samples,0)
                for i=1,#objects,1 do
                    sim.setObjectInt32Parameter(objects[i],sim.objintparam_visibility_layer,1+256)
                    sim.setObjectSpecialProperty(objects[i],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
                    local p=sim.getObjectProperty(objects[i])
                    p=sim.boolOr32(p,sim.objectproperty_selectmodelbaseinstead+sim.objectproperty_dontshowasinsidemodel)-sim.objectproperty_selectmodelbaseinstead-sim.objectproperty_dontshowasinsidemodel
                    sim.setObjectProperty(objects[i],p)
                    sim.setObjectInt32Parameter(objects[i],sim.shapeintparam_respondable,1)
                    sim.setObjectParent(objects[i],partHolder,true)
                    partsToMove[#partsToMove+1]=objects[i]
                    local dta=sim.readCustomDataBlock(objects[i],simBWF.modelTags.PART)
                    dta=sim.unpackTable(dta)
                    dta['instanciated']=true
                    sim.writeCustomDataBlock(objects[i],simBWF.modelTags.PART,sim.packTable(dta))
                end
                maxVel=data['velocity']
                accel=data['acceleration']
                posVelAccel={0,0,0}
                lastRelPos=0
                rmlPosObj=sim.rmlPos(1,0.0001,-1,posVelAccel,{maxVel,accel,999999},{1},{movementDist,0})
                movementUnderway=true
            end
        end
    end

    i=1
    while i<=#allProducedParts do
        local h=allProducedParts[i][1]
        if sim.isHandleValid(h)>0 then
            local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
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
                    sim.setObjectInt32Parameter(h,sim.shapeintparam_static,1) -- we make it static now!
                    sim.resetDynamicObject(h) -- important, otherwise the dynamics engine doesn't notice the change!
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
end 
