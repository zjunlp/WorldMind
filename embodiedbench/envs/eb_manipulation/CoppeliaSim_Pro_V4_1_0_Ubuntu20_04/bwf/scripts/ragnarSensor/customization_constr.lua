function model.alignCalibrationBallsWithInputAndReturnRedBall()
    local conveyorHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)
    if conveyorHandle>=0 then
        -- Work with thresholds here, otherwise the scene modifies itself continuously little by little:
        local c=model.readInfo()
        local flipped=sim.boolAnd32(c.bitCoded,2)>0
        local p=sim.getObjectOrientation(model.handle,conveyorHandle)
        if flipped then
            local correct=(math.abs(p[1])>0.1*math.pi/180) or (math.abs(p[2])>0.1*math.pi/180)
            if (math.abs(p[3]-math.pi)>0.1*math.pi/180) and (math.abs(p[3]+math.pi)>0.1*math.pi/180) then
                correct=true
            end
            if correct then
                sim.setObjectOrientation(model.handle,conveyorHandle,{0,0,math.pi})
            end
        else
            local correct=(math.abs(p[1])>0.1*math.pi/180) or (math.abs(p[2])>0.1*math.pi/180) or (math.abs(p[3])>0.1*math.pi/180)
            if correct then
                sim.setObjectOrientation(model.handle,conveyorHandle,{0,0,0})
            end
        end
    else
        local h=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
        
        -- First align the ragnar sensor with its input item:
        if h>=0 then
            h=simBWF.callCustomizationScriptFunction("model.ext.alignCalibrationBallsWithInputAndReturnRedBall",h)

            local p=sim.getObjectOrientation(model.handle,h)
            local correct=(math.abs(p[1])>0.1*math.pi/180) or (math.abs(p[2])>0.1*math.pi/180) or (math.abs(p[3])>0.1*math.pi/180)
            local p=sim.getObjectPosition(model.handle,h)
            correct=correct or (math.abs(p[2])>0.0001) or (math.abs(p[3])>0.0001)

            -- Ball1 should be distant by the calibration ball distance from the connected item's ball1:
            local c=model.readInfo()
            local d=c['calibrationBallDistance']
            if math.abs(p[1]-d)>0.001 then
                p[1]=d
                correct=true
            end
            if correct then
                sim.setObjectOrientation(model.handle,h,{0,0,0})
                sim.setObjectPosition(model.handle,h,{p[1],0,0})
            end
            
            local r,p=sim.getObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions)
            r=sim.boolOr32(r,1+4)-(1+4) -- forbid rotation and translation when simulation is not running
            sim.setObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions,r)
        else
            local r,p=sim.getObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions)
            r=sim.boolOr32(r,1+4) -- allow rotation and translation when simulation is not running
            sim.setObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions,r)
        end
    end
    return model.handle
end

function model.adjustSensor()
    local c=model.readInfo()
    -- The blue ball should be in the Y axis of the sensor's frame, and within +- 0.5 of the origin:
    local p=sim.getObjectPosition(model.handles.blueBall,model.handle)
    local correct=(math.abs(p[1])>0.0005)or(math.abs(p[3])>0.0005)
    if p[2]>-0.1 and p[2]<0 then
        p[2]=0.11
        correct=true
    end
    if p[2]<0.1 and p[2]>=0 then
        p[2]=-0.11
        correct=true
    end
    if p[2]<-1.01 then 
        p[2]=-1.0
        correct=true
    end
    if p[2]>1.01 then 
        p[2]=1.0
        correct=true
    end
    local r,minZ=sim.getObjectFloatParameter(model.handles.sensor,sim.objfloatparam_objbbox_min_z)
    local r,maxZ=sim.getObjectFloatParameter(model.handles.sensor,sim.objfloatparam_objbbox_max_z)
    local s=maxZ-minZ
    local r,minX=sim.getObjectFloatParameter(model.handles.sensor,sim.objfloatparam_objbbox_min_x)
    local r,maxX=sim.getObjectFloatParameter(model.handles.sensor,sim.objfloatparam_objbbox_max_x)
    local sx=maxX-minX
    if math.abs(p[2]-s)>0.001 then
        correct=true
    end
    if math.abs(sx-c.detectionWidth)>0.001 then
        correct=true
    end
    if correct then
        sim.setObjectPosition(model.handles.blueBall,model.handle,{0,p[2],0})
        sim.scaleObject(model.handles.sensor,c.detectionWidth/sx,1,math.abs(p[2])/s)
        if p[2]>=0 then
            sim.setObjectOrientation(model.handles.sensor,model.handle,{-math.pi/2,0,0})
        else
            sim.setObjectOrientation(model.handles.sensor,model.handle,{math.pi/2,0,0})
        end
        simBWF.markUndoPoint()
    end
    
end

function model.avoidCircularInput(inputItem)
    -- We have: ragnarSensor --> item1 --> item2 ... --> itemN
    -- None of the above item's input should be 'inputItem'
    -- If 'inputItem' is -1, then none of the above item's input should be 'model.handle'
    -- A. Check this ragnarSensor:
    if inputItem>0 then
        local h=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
        if h==inputItem then
            simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,-1) -- this input closed the loop. We open it here.
            model.updatePluginRepresentation()
        end
    end
    
    if inputItem==-1 then
        inputItem=model.handle
    end

    -- B. Check connected items:
    local h=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    if h>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.avoidCircularInput",h,inputItem)
    end
end

function model.forbidInput(inputItem)
    local h=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)
    if h==inputItem then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,-1)
        model.updatePluginRepresentation()
    end
end
