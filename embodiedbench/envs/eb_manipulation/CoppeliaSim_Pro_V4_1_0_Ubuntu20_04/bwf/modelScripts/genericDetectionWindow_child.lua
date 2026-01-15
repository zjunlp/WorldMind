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

function checkIfLabelIsVisible(labelShape,hs)
    local successIndex=0 -- no detection
    local bits={1,2,4}
    local pos={height,width/2,-width/2}
    local orient={{-math.pi,0,0},{0,-math.pi/2,-math.pi/2},{0,math.pi/2,math.pi/2}}
    local modelM=sim.getObjectMatrix(model,-1)
    for sensL=1,3,1 do
        if sim.boolAnd32(detectorLocations,bits[sensL])>0 then
            -- that detector is enabled
            local labelM=sim.getObjectMatrix(labelShape,model)
            sim.setObjectOrientation(sensor2,model,orient[sensL])
            local sensorM=sim.getObjectMatrix(sensor2,model)
            local dotP=-labelM[3]*sensorM[3]-labelM[7]*sensorM[7]-labelM[11]*sensorM[11]
            if dotP>1 then dotP=1 end
            if dotP<-1 then dotP=-1 end
            local angle=math.acos(dotP)
            if math.abs(angle)<maxLabelAngle then
                -- ok, the angle looks fine
                -- Now we check 9 points on the label:
                local checkFailed=false
                local dispPts={}
                for x=-1,1,1 do
                    for y=-1,1,1 do
                        local labelPt={labelM[4]+labelM[1]*x*hs[1]+labelM[2]*y*hs[2],labelM[8]+labelM[5]*x*hs[1]+labelM[6]*y*hs[2],labelM[12]+labelM[9]*x*hs[1]+labelM[10]*y*hs[2]}
                        dispPts[#dispPts+1]=sim.multiplyVector(modelM,labelPt)
                        if sensL==1 then
                            sim.setObjectPosition(sensor2,model,{labelPt[1],labelPt[2],pos[sensL]})
                        else
                            sim.setObjectPosition(sensor2,model,{pos[sensL],labelPt[2],labelPt[3]})
                        end
                        if x==0 and y==0 then
                            sim.setObjectPosition(sensor3,sensor2,{0,0,0})
                            sim.setObjectOrientation(sensor3,sensor2,{0,0,0})
                        end
                        local r,dist,pt,obj,n=sim.handleProximitySensor(sensor2)
                        if r<=0 or obj~=labelShape then
                            checkFailed=true
                            break
                        end
                    end
                    if checkFailed then
                        break
                    end
                end
                if not checkFailed then
                    if showPoints then
                        for i=1,#dispPts,1 do
                            sim.addDrawingObjectItem(labelSpheres,{dispPts[i][1],dispPts[i][2],dispPts[i][3],0,0,1})
                        end
                    end
                    if colorLabels then
                        sim.setShapeColor(labelShape,nil,sim.colorcomponent_ambient_diffuse,{1,0.5,0})
                    end
                    successIndex=sensL -- success
                    break
                end
            end
        end
    end
    if successIndex>0 and showLabels then
        if successIndex==1 then
            sim.setObjectFloatParameter(sensor3,sim.visionfloatparam_far_clipping,height)
        else
            sim.setObjectFloatParameter(sensor3,sim.visionfloatparam_far_clipping,width)
        end
        sim.setObjectFloatParameter(sensor3,sim.visionfloatparam_ortho_size,math.max(hs[1],hs[2])*2.5)
        sim.handleVisionSensor(sensor3)
    end
    sim.setObjectOrientation(sensor2,model,orient[1])
    return successIndex>0
end

function checkIfPartHasVisibleLabels(obj)
    local objs=sim.getObjectsInTree(obj,sim.object_shape_type,1)
    for i=1,#objs,1 do
        local data=sim.readCustomDataBlock(objs[i],'XYZ_PARTLABEL_INFO')
        if data then
            -- This is a label
            local label=objs[i]
            local r,mmin=sim.getObjectFloatParameter(label,sim.objfloatparam_objbbox_min_x )
            local r,mmax=sim.getObjectFloatParameter(label,sim.objfloatparam_objbbox_max_x )
            local sx=mmax-mmin
            local r,mmin=sim.getObjectFloatParameter(label,sim.objfloatparam_objbbox_min_y )
            local r,mmax=sim.getObjectFloatParameter(label,sim.objfloatparam_objbbox_max_y )
            local sy=mmax-mmin
            if checkIfLabelIsVisible(label,{sx*0.45,sy*0.45}) then
                return true
            end
        end
    end
    return false
end

function getAllVisiblePartsInWindow()
    if showPoints then
        sim.addDrawingObjectItem(labelSpheres,nil) -- clear the label detection spheres
    end
    local m=sim.getObjectMatrix(model,-1)
    local op=sim.getObjectPosition(sensor1,model)
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    local retL2={}
    for i=1,#l,1 do
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
        if isInstanciated then
            local p=sim.getObjectPosition(l[i],model)
            if (math.abs(p[1])<width*0.5) and (math.abs(p[2])<length*0.5) and (p[3]>0) and (p[3]<height) then
                retL2[#retL2+1]=l[i]
                sim.setObjectPosition(sensor1,model,{p[1],p[2],op[3]})
                local r,dist,pt,obj,n=sim.handleProximitySensor(sensor1)
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
                        local hasLabel=false
                        if detectorLocations>0 then
                            hasLabel=checkIfPartHasVisibleLabels(obj)
                        end
                        if n[3]<0 then
                            n[1]=-n[1]
                            n[2]=-n[2]
                            n[3]=-n[3]
                        end
                        -- We fix this for now (donut problem):
                        n[1]=0
                        n[2]=0
                        n[3]=1

                        p=sim.multiplyVector(m,{p[1],p[2],op[3]-dist})
                        retL[l[i]]={data['name'],p,{0,0,0},0,getPartMass(l[i]),data['destination'],n,hasLabel} -- name, pickPos, dxVector,vel,mass,destination,detectedSurfaceNormalVector,labelPresent
                    end
                end
            end
        end
    end
    return retL,retL2
end

displayParts=function(parts)
    sim.addDrawingObjectItem(sphereContainer,nil)
    sim.addDrawingObjectItem(lineContainer,nil)
    sim.addDrawingObjectItem(line2Container,nil)
    for key,value in pairs(parts) do
        sim.addDrawingObjectItem(sphereContainer,{value[2][1],value[2][2],value[2][3],0,0,1})
        sim.addDrawingObjectItem(lineContainer,{value[2][1],value[2][2],value[2][3]+0.001,value[2][1]+value[3][1],value[2][2]+value[3][2],value[2][3]+value[3][3]+0.001})
        sim.addDrawingObjectItem(line2Container,{value[2][1],value[2][2],value[2][3]+0.001,value[2][1]+value[7][1]*0.1,value[2][2]+value[7][2]*0.1,value[2][3]+value[7][3]*0.1+0.001})
    end
end

displayConsoleIfNeeded=function(info)
    if console then
        sim.auxiliaryConsolePrint(console,nil)
        for key,value in pairs(info) do
            local str=sim.getObjectName(key)..':\n'
            str=str..'    handle: '..key..', partName: '..value['partName']..', destinationName: '..value['destinationName']..'\n'
            str=str..'    pick position: ('
            str=str..simBWF.format("%.0f",value['pickPos'][1]*1000)..','
            str=str..simBWF.format("%.0f",value['pickPos'][2]*1000)..','
            str=str..simBWF.format("%.0f",value['pickPos'][3]*1000)..')\n'
            str=str..'    velocity vector: ('
            str=str..simBWF.format("%.0f",value['velocityVect'][1]*1000)..','
            str=str..simBWF.format("%.0f",value['velocityVect'][2]*1000)..','
            str=str..simBWF.format("%.0f",value['velocityVect'][3]*1000)..')\n'
            str=str..'    normal vector: ('
            str=str..simBWF.format("%.0f",value['normalVect'][1]*1000)..','
            str=str..simBWF.format("%.0f",value['normalVect'][2]*1000)..','
            str=str..simBWF.format("%.0f",value['normalVect'][3]*1000)..')\n'
            str=str..'    mass: '..simBWF.format("%.2f",value['mass'])..'\n'
            str=str..'    label detected: '..tostring(value['hasLabel'])..'\n----------------------------------------------------------------\n'
            sim.auxiliaryConsolePrint(console,str)
        end
    end
end

prepareStatisticsDialog=function(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                <label id="2" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                <label id="3" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                <label id="4" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

updateStatisticsDialog=function(totalP,detectedPNoLabel,detectedPWithLabel)
    if statUi then
        if totalP>0 then
            local d=detectedPNoLabel+detectedPWithLabel
            simUI.setLabelText(statUi,1,simBWF.format("Detected parts with label: %.1f [%%]",100*detectedPWithLabel/totalP),true)
            simUI.setLabelText(statUi,2,simBWF.format("Detected parts without label: %.1f [%%]",100*detectedPNoLabel/totalP),true)
            simUI.setLabelText(statUi,3,simBWF.format("Not detected parts: %.1f [%%]",100*(totalP-d)/totalP),true)
            simUI.setLabelText(statUi,4,simBWF.format("Total parts: %.0f ",totalP),true)
        else
            simUI.setLabelText(statUi,1,"Detected parts with label: 0.0 [%]",true)
            simUI.setLabelText(statUi,2,"Detected parts without label: 0.0 [%]",true)
            simUI.setLabelText(statUi,3,"Not detected parts: 0.0 [%]",true)
            simUI.setLabelText(statUi,4,"Total parts: 0",true)
        end
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    sensor1=sim.getObjectHandle('genericDetectionWindow_sensor1')
    sensor2=sim.getObjectHandle('genericDetectionWindow_sensor2')
    sensor3=sim.getObjectHandle('genericDetectionWindow_sensor3')
    local data=sim.readCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO')
    data=sim.unpackTable(data)
    width=data['width']
    length=data['length']
    height=data['height']
    maxLabelAngle=data['maxLabelAngle']
    if sim.boolAnd32(data['bitCoded'],2)>0 then
        console=sim.auxiliaryConsoleOpen('Parts in detection window',1000,4,nil,{600,300},nil,{1,0.9,0.9})
    end
    
    showPoints=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(data['bitCoded'],4)>0)
    showLabels=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(data['bitCoded'],8)>0)
    detectorLocations=0
    if (sim.boolAnd32(data['bitCoded'],16)>0) then detectorLocations=detectorLocations+1 end
    if (sim.boolAnd32(data['bitCoded'],32)>0) then detectorLocations=detectorLocations+2 end
    if (sim.boolAnd32(data['bitCoded'],64)>0) then detectorLocations=detectorLocations+4 end
    if detectorLocations==0 then showLabels=false end
    colorLabels=(sim.boolAnd32(data['bitCoded'],128)>0)
    labelSpheres=sim.addDrawingObject(sim.drawing_spherepoints,0.005,0,-1,9999,{0,0,0})
    sphereContainer=sim.addDrawingObject(sim.drawing_spherepoints,0.015,0,-1,9999,{1,0,0})
    lineContainer=sim.addDrawingObject(sim.drawing_lines,3,0,-1,9999,{1,0.25,0})
    line2Container=sim.addDrawingObject(sim.drawing_lines,3,0,-1,9999,{0,0.5,1})
    if showLabels then
        floatingView=sim.floatingViewAdd(0.1,0.9,0.2,0.2,0)
        sim.adjustView(floatingView,sensor3,64,'Label Detection')
    end
    prepareStatisticsDialog(sim.boolAnd32(data['bitCoded'],256)>0)
    previousParts={}
    previousTime=0
    allPartsInWindowForAWhile={}
    totalPartCnt=0
    totalPartWithLabelCnt=0
    totalPartWithoutLabelCnt=0
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    local dt=t-previousTime
    local detectedParts,allParts=getAllVisiblePartsInWindow()
    for i=1,#allParts,1 do
        if not allPartsInWindowForAWhile[allParts[i]] then
            allPartsInWindowForAWhile[allParts[i]]={t,false,false}
            totalPartCnt=totalPartCnt+1
        end
    end

    local inf={}
    for key,value in pairs(detectedParts) do
        local dat=previousParts[key]
        if dat then
            local dv={value[2][1]-dat[2][1],value[2][2]-dat[2][2],value[2][3]-dat[2][3]}
            dv[1]=dv[1]/dt
            dv[2]=dv[2]/dt
            dv[3]=dv[3]/dt
            local l=math.sqrt(dv[1]*dv[1]+dv[2]*dv[2]+dv[3]*dv[3])
            value[3]=dv
            value[4]=l
        end
        local b={}
        b['partName']=value[1]
        b['destinationName']=value[6]
        b['pickPos']=value[2]
        b['velocityVect']=value[3]
        b['mass']=value[5]
        b['axes']=getAxesWithOrderingAccordingToSize(key)
        b['normalVect']=value[7]
        b['hasLabel']=value[8]
        b['transform']={sim.getObjectPosition(key,-1),sim.getObjectQuaternion(key,-1)}
        inf[key]=b
        if allPartsInWindowForAWhile[key][2]==false then
            allPartsInWindowForAWhile[key][2]=true
            if value[8] then
                totalPartWithLabelCnt=totalPartWithLabelCnt+1
            else
                totalPartWithoutLabelCnt=totalPartWithoutLabelCnt+1
            end
        end
    end
    local data=sim.readCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO')
    data=sim.unpackTable(data)
    data['detectedItems']=inf
    sim.writeCustomDataBlock(model,'XYZ_DETECTIONWINDOW_INFO',sim.packTable(data))
    if showPoints then
        displayParts(detectedParts)
    end
    displayConsoleIfNeeded(inf)
    previousParts=detectedParts
    previousTime=t
    updateStatisticsDialog(totalPartCnt,totalPartWithoutLabelCnt,totalPartWithLabelCnt)
    local toRemove={}
    for key,value in pairs(allPartsInWindowForAWhile) do
        if t-value[1]>60 then
            toRemove[#toRemove+1]=key
        end
    end
    for i=1,#toRemove,1 do
        allPartsInWindowForAWhile[toRemove[i]]=nil
    end
end
