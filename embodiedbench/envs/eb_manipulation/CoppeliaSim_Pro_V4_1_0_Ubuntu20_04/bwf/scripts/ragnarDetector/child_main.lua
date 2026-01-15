function model.getFakeDetectedPartsInWindow()
    local m=sim.getObjectMatrix(model.handles.detectorBox,-1)
    local op=sim.getObjectPosition(model.handles.detectorSensor,model.handles.detectorBox)
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    for i=1,#l,1 do
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
        if isInstanciated then
            local p=sim.getObjectPosition(l[i],model.handles.detectorBox)
            if (math.abs(p[1])<model.boxSize[1]*0.5) and (math.abs(p[2])<model.boxSize[2]*0.5) and (math.abs(p[3])<model.boxSize[3]*0.5) then
                sim.setObjectPosition(model.handles.detectorSensor,model.handles.detectorBox,{p[1],p[2],op[3]})
                local r,dist,pt,obj,n=sim.handleProximitySensor(model.handles.detectorSensor)
                if r>0 then
                    -- Only if we detected the same object (there might be overlapping objects)
                    while obj~=-1 do
                        local data2=sim.readCustomDataBlock(obj,simBWF.modelTags.PART)
                        if data2 then
                            break
                        end
                        obj=sim.getObjectParent(obj)
                    end
                    if obj==l[i] then
                        p=sim.multiplyVector(m,{p[1],p[2],op[3]-dist})
--                        retL[#retL+1]={id=l[i],type=data.type,pos=p,name=data.name,destination=data.destination}
                        retL[#retL+1]={id=l[i],type=data.type,pos=p,name=data.name}
                    end
                end
            end
        end
    end
    return retL
end

function model.getObjectSize(h)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    return {sx,sy,sz}
end

function sysCall_init()
    model.codeVersion=1
    
    model.showDetections=false
    model.online=simBWF.isSystemOnline()
    local data=model.readInfo()
    
    model.showDetections=sim.boolAnd32(data.bitCoded,4)~=0
    model.sphereContainer=sim.addDrawingObject(sim.drawing_spherepoints,0.007,0,-1,9999,{1,0,0})

    model.ball1Mi=sim.getObjectMatrix(model.handle,-1)
    sim.invertMatrix(model.ball1Mi)
    model.boxSize=model.getObjectSize(model.handles.detectorBox)
    model.alreadyFakeDetectedAndTransmittedParts={}
end

function sysCall_sensing()
    -- Following for the fake detection:
    local t=sim.getSimulationTime()
    local detected=model.getFakeDetectedPartsInWindow()
    sim.addDrawingObjectItem(model.sphereContainer,nil)
    local dataToTransmit={}
    dataToTransmit.id=model.handle
    dataToTransmit.pos={}
    dataToTransmit.types={}
    dataToTransmit.names={}
--    dataToTransmit.destinations={}
    for i=1,#detected,1 do
        local id=detected[i].id
        if not model.alreadyFakeDetectedAndTransmittedParts[id] then
            local p=sim.multiplyVector(model.ball1Mi,detected[i].pos) -- the point is now relative to the red calibration ball
            dataToTransmit.pos[#dataToTransmit.pos+1]=p
            dataToTransmit.types[#dataToTransmit.types+1]=detected[i].type
            dataToTransmit.names[#dataToTransmit.names+1]=detected[i].name
--            dataToTransmit.destinations[#dataToTransmit.destinations+1]=detected[i].destination
            if model.showDetections then
                sim.addDrawingObjectItem(model.sphereContainer,detected[i].pos)
            end
        end
        model.alreadyFakeDetectedAndTransmittedParts[id]=t
    end
    if #dataToTransmit.types>0 then
        simBWF.query('ragnarDetector_detections',dataToTransmit)
    end
    for key,value in pairs(model.alreadyFakeDetectedAndTransmittedParts) do
        if t-value>2 then
            model.alreadyFakeDetectedAndTransmittedParts[key]=nil
        end
    end
end


function sysCall_cleanup()
end

