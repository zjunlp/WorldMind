displayParts=function(trackingWindowParts)
    sim.addDrawingObjectItem(sphere1Container,nil)
    sim.addDrawingObjectItem(lineContainerR,nil)
    sim.addDrawingObjectItem(lineContainerG,nil)
    sim.addDrawingObjectItem(lineContainerB,nil)
    local al=0.06
    for key,value in pairs(trackingWindowParts) do
        local p=value['pickPos']
        local a=value['axes']
        sim.addDrawingObjectItem(sphere1Container,{p[1],p[2],p[3],0,0,1})
        sim.addDrawingObjectItem(lineContainerR,{p[1],p[2],p[3]+0.001,p[1]+a[1][1]*al,p[2]+a[1][2]*al,p[3]+a[1][3]*al+0.001})
        sim.addDrawingObjectItem(lineContainerG,{p[1],p[2],p[3]+0.001,p[1]+a[2][1]*al,p[2]+a[2][2]*al,p[3]+a[2][3]*al+0.001})
        sim.addDrawingObjectItem(lineContainerB,{p[1],p[2],p[3]+0.001,p[1]+a[3][1]*al,p[2]+a[3][2]*al,p[3]+a[3][3]*al+0.001})
    end
end

getSensorState=function()
    if sensorHandle>=0 then
        local data=sim.readCustomDataBlock(sensorHandle,'XYZ_BINARYSENSOR_INFO')
        if data then
            data=sim.unpackTable(data)
            return data['detectionState']
        end
    end
    return 0
end

getPartMass=function(partHandle)
    local m=0
    if partHandle>=0 then
        local modProp=sim.getModelProperty(partHandle)
        if sim.boolAnd32(modProp,sim.modelproperty_not_model)==0 then
            local objects={partHandle}
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
                        m=m+sim.getShapeMassAndInertia(handle)
                    end
                end
            end
        else
            m=m+sim.getShapeMassAndInertia(partHandle)
        end
    end
    return m
end

getAxesWithOrderingAccordingToSize=function(partHandle)
    local modProp=sim.getModelProperty(partHandle)
    local sx=0
    local sy=0
    local sz=0
    if sim.boolAnd32(modProp,sim.modelproperty_not_model)==0 then
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_x )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_x )
        sx=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_y )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_y )
        sy=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_min_z )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_modelbbox_max_z )
        sz=mmax-mmin
    else
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_x )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_x )
        sx=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_y )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_y )
        sy=mmax-mmin
        local r,mmin=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_min_z )
        local r,mmax=sim.getObjectFloatParameter(partHandle,sim.objfloatparam_objbbox_max_z )
        sz=mmax-mmin
    end
    local m=sim.getObjectMatrix(partHandle,-1)
    local axes={{sx,{m[1],m[5],m[9]}},{sy,{m[2],m[6],m[10]}},{sz,{m[3],m[7],m[11]}}}
    if axes[1][1]>axes[2][1] then
        local tmp=axes[1]
        axes[1]=axes[2]
        axes[2]=tmp
    end
    if axes[2][1]>axes[3][1] then
        local tmp=axes[2]
        axes[2]=axes[3]
        axes[3]=tmp
    end
    if axes[1][1]>axes[2][1] then
        local tmp=axes[1]
        axes[1]=axes[2]
        axes[2]=tmp
    end
    return {axes[1][2],axes[2][2],axes[3][2]}
end

getAllPartsInWindow=function()
    local retVal={}
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    for i=1,#l,1 do
        local partHandle=l[i]
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(partHandle)
        if isInstanciated then
            local relPos=sim.getObjectPosition(partHandle,model)
            if (math.abs(relPos[1])<width*0.5) and (math.abs(relPos[2])<length*0.5) and (relPos[3]>0) and (relPos[3]<height) then
                local partData={}
                partData['partName']=data['name']
                partData['destinationName']=data['destination']
                partData['pickPos']=sim.getObjectPosition(partHandle,-1)
                partData['velocityVect']={0,0,0}
                partData['axes']=getAxesWithOrderingAccordingToSize(partHandle)
                local m=sim.getObjectMatrix(model,-1)
                partData['normalVect']={m[3],m[7],m[11]} -- the normal is the one from the model, not the object
                partData['hasLabel']=false
                partData['transform']={sim.getObjectPosition(partHandle,-1),sim.getObjectQuaternion(partHandle,-1)}
                partData['mass']=getPartMass(partHandle)
                local h=sim.createDummy(0.005)
                sim.setObjectParent(h,model,true)
                sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1024)
                sim.setObjectPosition(h,-1,partData['transform'][1])
                sim.setObjectQuaternion(h,-1,partData['transform'][2])
                partData['dummyHandle']=h
                retVal[partHandle]=partData
            end
        end
    end
    return retVal
end

removeTrackedPart=function(partHandle)
    local h=trackedParts[partHandle]['dummyHandle']
    local objs=sim.getObjectsInTree(h) -- if the part is decorated, it could have several dummy children
    for i=1,#objs,1 do
        sim.removeObject(objs[i])
    end
    trackedParts[partHandle]=nil
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO')
    data=sim.unpackTable(data)
    sensorHandle=simBWF.getReferencedObjectHandle(model,simBWF.STATICPICKWINDOW_SENSOR_REF)
    width=data['width']
    length=data['length']
    height=data['height']
    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(data['bitCoded'],4)>0)
    sphere1Container=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{0,1,0})
    lineContainerR=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{1,0,0})
    lineContainerG=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,1,0})
    lineContainerB=sim.addDrawingObject(sim.drawing_lines,2,0,-1,9999,{0,0,1})

    trackedParts={}
    previousSensorState=0
    previousTimeWhenOwnTriggered=0
    mode=0 -- 0=fill mode (waiting for sensor trigger), 1=empty mode (waiting for trackedParts to be empty)
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    local data=sim.readCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO')
    data=sim.unpackTable(data)
    local sensorState=getSensorState()

    if mode==0 then
        if sensorState~=previousSensorState then
            trackedParts=getAllPartsInWindow()
            local cnt=0
            for key,value in pairs(trackedParts) do
                cnt=cnt+1
            end
            if cnt>0 then
                mode=1
            end
        else
            if t-previousTimeWhenOwnTriggered>0.251 then
                data['triggerState']=data['triggerState']+1
                previousTimeWhenOwnTriggered=t
            end
        end
    end
    local toRemove=data['itemsToRemoveFromTracking']
    for i=1,#toRemove,1 do
        removeTrackedPart(toRemove[i])
    end
    data['itemsToRemoveFromTracking']={}
    if mode>0 then
        local cnt=0
        for key,value in pairs(trackedParts) do
            cnt=cnt+1
        end
        if cnt==0 then
            mode=0
        end
    end

    if showPoints then
        displayParts(trackedParts)
    end

    data['trackedItemsInWindow']=trackedParts
    sim.writeCustomDataBlock(model,'XYZ_STATICPICKWINDOW_INFO',sim.packTable(data))

    previousSensorState=sensorState
end