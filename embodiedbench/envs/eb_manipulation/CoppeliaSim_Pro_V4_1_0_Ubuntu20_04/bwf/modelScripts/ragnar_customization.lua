function removeFromPluginRepresentation()
--[[
    local data={}
    data.id=model
    simBWF.query('object_delete',data)
    --]]
end

function updatePluginRepresentation()
--[[
    local c=readInfo()
    local data={}
    data.id=model
    data.name=sim.getObjectName(model)
    data.pos=sim.getObjectPosition(model,-1)
    data.ypr=sim.getObjectOrientation(model,-1)
    data.primaryArmLength=c.primaryArmLengthInMM/1000
    data.secondaryArmLength=c.secondaryArmLengthInMM/1000
    simBWF.query('ragnar_update',data)
    --]]
end

function ext_getItemData_pricing()
    local c=readInfo()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.brVersion=0
    obj.type='ragnar'
    obj.ragnarType='default'
    local tmp={'standard','high-power'}
    obj.motors=tmp[c.motorType+1]
    local tmp={'std','wd'}
    obj.exterior=tmp[c.exteriorType+1]
    local tmp={'experimental','industrial'}
    obj.frame=tmp[c.frameType+1]
    obj.primary_arms=c.primaryArmLengthInMM
    obj.secondary_arms=c.secondaryArmLengthInMM
    local dep={}
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICWINDOW1_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local id=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF)
    if id>=0 then dep[#dep+1]=sim.getObjectName(id) end
    local ob=sim.getObjectsInTree(model)
    for i=1,#ob,1 do
        local data=sim.readCustomDataBlock(ob[i],simBWF.modelTags.RAGNARGRIPPER)
        if data then
            dep[#dep+1]=sim.getObjectName(ob[i])
            break
        end
    end
    if #dep>0 then
        obj.dependencies=dep
    end
    return obj
end

setFkMode=function()
    -- disable the platform positional constraints:
    sim.setIkElementProperties(ikGroup,ikModeTipDummy,0)
    -- Set the driving joints into passive mode (not taken into account during IK resolution):
    sim.setJointMode(fkDrivingJoints[1],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[2],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[3],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[4],sim.jointmode_passive,0)
end

function getZPosition()
    return sim.getObjectPosition(model,-1)[3]
end

function openFrame(open)
    local a=0
    if open then
        a=-math.pi
    end
    for i=1,3,1 do
        sim.setJointPosition(frameOpenClose[i],a)
    end
end

function setFrameVisible(visible)
    local p=0
    if not visible then
        p=sim.modelproperty_not_collidable+sim.modelproperty_not_detectable+sim.modelproperty_not_dynamic+
          sim.modelproperty_not_measurable+sim.modelproperty_not_renderable+sim.modelproperty_not_respondable+
          sim.modelproperty_not_visible+sim.modelproperty_not_showasinsidemodel
    end
    sim.setModelProperty(frameModel,p)
end

function setLowBeamsVisible(visible)
    for i=1,2,1 do
        if not visible then
            sim.setObjectSpecialProperty(frameBeams[i],0)
            sim.setObjectInt32Parameter(frameBeams[i],sim.objintparam_visibility_layer,0)
        else
            sim.setObjectSpecialProperty(frameBeams[i],sim.objectspecialproperty_collidable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_measurable+sim.objectspecialproperty_renderable)
            sim.setObjectInt32Parameter(frameBeams[i],sim.objintparam_visibility_layer,1)
        end
    end
end

function isFrameOpen()
    return math.abs(sim.getJointPosition(frameOpenClose[1]))<0.1
end

function getAvailableStaticPartWindows()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_STATICPICKWINDOW_INFO')
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableStaticTargetWindows()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_STATICPLACEWINDOW_INFO')
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableTrackingWindows()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.TRACKINGWINDOW)
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableDropLocations(returnMap)
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data1=sim.readCustomDataBlock(l[i],simBWF.modelTags.OLDLOCATION)
        local data2=sim.readCustomDataBlock(l[i],'XYZ_BUCKET_INFO')
        if data1 or data2 then
            if returnMap then
                retL[sim.getObjectName(l[i])]=l[i]
            else
                retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='ragnar'
    end
    if not info['primaryArmLengthInMM'] then
        info['primaryArmLengthInMM']=300
    end
    if not info['secondaryArmLengthInMM'] then
        info['secondaryArmLengthInMM']=550
    end
    if not info['maxVel'] then
        info['maxVel']=1
    end
    if not info['maxAccel'] then
        info['maxAccel']=1
    end
    if not info['dwellTime'] then
        info['dwellTime']=0.1
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- 1=visualize trajectory, 2=frame open, 4=reserved. set to 0,8= frame low beam visible, 16=reserved. set to 0, 64=enabled, 128=show statistics, 256=show ws, 512=show ws also during simulation, 1024=attach part to target via a force sensor, 2048=pick part without target in sight, 4096=ragnar in FK mode and idle, 8192=showGraph, 16384=reflectConfig
    end
    if not info['trackingTimeShift'] then
        info['trackingTimeShift']=0
    end
    if not info['algorithm'] then
        info['algorithm']=''
    end
    if not info['pickOffset'] then
        info['pickOffset']={0,0,0}
    end
    if not info['placeOffset'] then
        info['placeOffset']={0,0,0}
    end
    if not info['pickRounding'] then
        info['pickRounding']=0.05
    end
    if not info['placeRounding'] then
        info['placeRounding']=0.05
    end
    if not info['pickNulling'] then
        info['pickNulling']=0.005
    end
    if not info['placeNulling'] then
        info['placeNulling']=0.005
    end
    if not info['pickApproachHeight'] then
        info['pickApproachHeight']=0.1
    end
    if not info['placeApproachHeight'] then
        info['placeApproachHeight']=0.1
    end

    if not info['connectionIp'] then
        info['connectionIp']="127.0.0.1"
    end
    if not info['connectionPort'] then
        info['connectionPort']=19800
    end
    if not info['connectionTimeout'] then
        info['connectionTimeout']=1
    end
    if not info['connectionBufferSize'] then
        info['connectionBufferSize']=1000
    end
    if not info['motorType'] then
        info['motorType']=0 -- 0=standard, 1=high-power
    end
    if not info['exteriorType'] then
        info['exteriorType']=0 -- 0=standard, 1=wash-down
    end
    if not info['frameType'] then
        info['frameType']=0 -- 0=experimental, 1=industrial
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    getDefaultInfoForNonExistingFields(data)
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,simBWF.modelTags.RAGNAR,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.RAGNAR,'')
    end
end

function getLinkBLength(a,f)
    local tol=0.001 -- very small tolerance value to make sure the nominal robot has sizes a=300, b=550
    return 0.05*math.ceil((a*f-tol)/0.05)
end

function showHideWorkspace(show)
    local r,minZ=sim.getObjectFloatParameter(workspace,sim.objfloatparam_objbbox_min_z)
    local r,maxZ=sim.getObjectFloatParameter(workspace,sim.objfloatparam_objbbox_max_z)
    local s=maxZ-minZ
    local inf=readInfo()
    local primaryArmLengthInMM=inf['primaryArmLengthInMM']
    local a=primaryArmLengthInMM/1000+0.0005
    
    
    local d=3.569384*a -- 3.569384=1.0726/0.3005
    sim.scaleObject(workspace,d/s,d/s,d/s)

    local p={-0.00485*a/0.3005,-0.00176*a/0.3005,-0.48947*a/0.3005}
    sim.setObjectPosition(workspace,sim.handle_parent,p)

    if show then
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,0)
    end
end

function isWorkspaceVisible()
    local c=readInfo()
    return sim.boolAnd32(c['bitCoded'],256)>0
end

function adjustRobot()
    local inf=readInfo()
    local primaryArmLengthInMM=inf['primaryArmLengthInMM']
    local secondaryArmLengthInMM=inf['secondaryArmLengthInMM']

--    local a=0.2+((primaryArmLengthInMM-200)/50)*0.05+0.0005
    local a=primaryArmLengthInMM/1000+0.0005
    local b=secondaryArmLengthInMM/1000
 

    local c=0.025
    local x=math.sqrt(a*a-c*c)
    local upAdjust=x-math.sqrt(0.3005*0.3005-c*c) -- Initial lengths are 300.5 and 550.0 (not 300/550!)
    local downAdjust=b-0.55
    local dx=a*28/30
    local ddx=dx-0.28

---[[
    for i=1,4,1 do
        sim.setJointPosition(upperArmAdjust[i],upAdjust)
    end

    for i=1,8,1 do
        sim.setJointPosition(lowerArmAdjust[i],downAdjust)
    end


    for i=1,4,1 do
        sim.setJointPosition(upperArmLAdjust[i],upAdjust*0.5)
    end

    for i=1,8,1 do
        sim.setJointPosition(lowerArmLAdjust[i],downAdjust*0.5)
    end

    for i=1,2,1 do
        sim.setJointPosition(frontAndRearCoverAdjust[i],ddx)
    end

    for i=1,3,1 do
        local h=middleCoverParts[i]
        local r,minY=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
        local r,maxY=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
        local s=maxY-minY
        local d=0.28+0.0122+ddx*2
        sim.scaleObject(h,1,d/s,1)
    end

    for i=1,2,1 do
        local h=middleCoverParts[3+i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local d=0.3391
        if a<0.18 then
            d=0.1187
        elseif a<0.23 then
            d=0.2204
        end
        sim.scaleObject(h,d/s,d/s,d/s)
    end


    for i=1,4,1 do
        local h=upperLinks[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local d=0.242+upAdjust
        sim.scaleObject(h,1,1,d/s)
    end

    for i=1,8,1 do
        local h=lowerLinks[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local r,minX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
        local r,maxX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
        local sx=maxX-minX
        local d=0.5+downAdjust
        local diam=0.01
        if d>=0.5 then
            diam=0.014
        end
        sim.scaleObject(h,diam/sx,diam/sx,d/s)
    end

    local p=sim.getObjectPosition(ikTarget,model)

    relZPos=-a*2

    sim.setObjectPosition(ikTarget,model,{p[1],p[2],relZPos})

    sim.handleIkGroup(ikGroup)

    -- The frame:
    local nomS={0.9674,0.9674,0.9674,0.411,0.98509,0.98509,0.7094,0.7094}
    for i=1,4,1 do
        local h=frameBeams[i]
        local r,minY=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
        local r,maxY=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
        local s=maxY-minY
        local d=nomS[i]+ddx*2
        sim.scaleObject(h,1,d/s,1)
    end
    sim.setJointPosition(frameJoints[1],ddx)
    sim.setJointPosition(frameJoints[2],ddx)
--]]
end

function adjustHeight(z)
    local dz=z-1.36
    local nomS={0.9674,0.9674,0.9674,0.411,0.98509,0.98509,0.7094,0.7094}
    for i=5,8,1 do
        local h=frameBeams[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local d=nomS[i]+dz
        sim.scaleObject(h,1,1,d/s)
    end
    local c=readInfo()


    sim.setJointPosition(frameJoints[3],-dz)
    sim.setJointPosition(frameJoints[4],-dz)
    sim.setJointPosition(frameJoints[5],-dz)
    sim.setJointPosition(frameJoints[6],-dz)
    local p=sim.getObjectPosition(model,-1)
    sim.setObjectPosition(model,-1,{p[1],p[2],z})

    for i=7,10,1 do
        sim.setJointPosition(frameJoints[i],-dz*0.5)
    end
end

function getJointPositions(handles)
    local retTable={}
    for i=1,#handles,1 do
        retTable[i]=sim.getJointPosition(handles[i])
    end
    return retTable
end

function setJointPositions(handles,positions)
    for i=1,#handles,1 do
        sim.setJointPosition(handles[i],positions[i])
    end
end

function updateLinkLengthDisplay()
    if ui then
        local c=readInfo()
        simUI.setLabelText(ui,1,'Primary arm length: '..simBWF.format("%.0f",c['primaryArmLengthInMM'])..' mm')
        simUI.setLabelText(ui,91,'Secondary arm length: '..simBWF.format("%.0f",c['secondaryArmLengthInMM'])..' mm')
    end
end

function updateMovementParamDisplay()
    if ui then
        local c=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,10,simBWF.format("%.0f",c['maxVel']*1000),true)
        simUI.setEditValue(ui,11,simBWF.format("%.0f",c['maxAccel']*1000),true)
        simUI.setEditValue(ui,12,simBWF.format("%.3f",c['dwellTime']),true)
        simUI.setEditValue(ui,13,simBWF.format("%.3f",c['trackingTimeShift']),true)
        local off=c['pickOffset']
        simUI.setEditValue(ui,1001,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
        off=c['placeOffset']
        simUI.setEditValue(ui,1002,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
        simUI.setEditValue(ui,1003,simBWF.format("%.0f",c['pickRounding']*1000),true)
        simUI.setEditValue(ui,1004,simBWF.format("%.0f",c['placeRounding']*1000),true)
        simUI.setEditValue(ui,1005,simBWF.format("%.0f",c['pickNulling']*1000),true)
        simUI.setEditValue(ui,1006,simBWF.format("%.0f",c['placeNulling']*1000),true)
        simUI.setEditValue(ui,1007,simBWF.format("%.0f",c['pickApproachHeight']*1000),true)
        simUI.setEditValue(ui,1008,simBWF.format("%.0f",c['placeApproachHeight']*1000),true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function setArmLength(primaryArmLengthInMM,secondaryArmLengthInMM)
    local allowedB={} -- in multiples of 50
    allowedB[200]={400,450}
    allowedB[250]={450,600}
    allowedB[300]={550,700}
    allowedB[350]={650,800}
    allowedB[400]={750,900}
    allowedB[450]={850,1000}
    allowedB[500]={900,1150}
    allowedB[550]={1000,1250}
    
    local allowedA={} -- in multiples of 50
    allowedA[400]={200,200}
    allowedA[450]={200,250}
    allowedA[500]={250,250}
    allowedA[550]={250,300}
    allowedA[600]={250,300}
    allowedA[650]={300,350}
    allowedA[700]={300,350}
    allowedA[750]={350,400}
    allowedA[800]={350,400}
    allowedA[850]={400,450}
    allowedA[900]={400,500}
    allowedA[950]={450,500}
    allowedA[1000]={450,550}
    allowedA[1050]={500,550}
    allowedA[1100]={500,550}
    allowedA[1150]={500,550}
    allowedA[1200]={550,550}
    allowedA[1250]={550,550}
    
    local c=readInfo()
    if primaryArmLengthInMM then
        -- We changed the primary arm length
        c['primaryArmLengthInMM']=primaryArmLengthInMM
        local allowed=allowedB[primaryArmLengthInMM]
        secondaryArmLengthInMM=c['secondaryArmLengthInMM']
        if secondaryArmLengthInMM<allowed[1] then
            secondaryArmLengthInMM=allowed[1]
        end
        if secondaryArmLengthInMM>allowed[2] then
            secondaryArmLengthInMM=allowed[2]
        end
        c['secondaryArmLengthInMM']=secondaryArmLengthInMM
    else
        -- We changed the secondary arm length
        c['secondaryArmLengthInMM']=secondaryArmLengthInMM
        local allowed=allowedA[secondaryArmLengthInMM]
        primaryArmLengthInMM=c['primaryArmLengthInMM']
        if primaryArmLengthInMM<allowed[1] then
            primaryArmLengthInMM=allowed[1]
        end
        if primaryArmLengthInMM>allowed[2] then
            primaryArmLengthInMM=allowed[2]
        end
        c['primaryArmLengthInMM']=primaryArmLengthInMM
    end
    writeInfo(c)
    simBWF.markUndoPoint()
    adjustRobot()
    showHideWorkspace(isWorkspaceVisible())
    updateLinkLengthDisplay()
end

function sizeAChange_callback(ui,id,newVal)
    setArmLength(200+newVal*50,nil)
    local c=readInfo()
    simUI.setSliderValue(ui,92,(c['secondaryArmLengthInMM']-400)/50,true)
end

function sizeBChange_callback(ui,id,newVal)
    setArmLength(nil,400+newVal*50)
    local c=readInfo()
    simUI.setSliderValue(ui,2,(c['primaryArmLengthInMM']-200)/50,true)
end

function ZChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    local z=getZPosition()
    if newValue then
        newValue=newValue/1000
        if newValue<1.0 then newValue=1.0 end
        if newValue>3 then newValue=3 end
        if newValue~=z then
            z=newValue
            adjustHeight(newValue)
            simBWF.markUndoPoint()
        end
    end
    simUI.setEditValue(ui,77,simBWF.format("%.0f",z*1000),true)
end

function velocityChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
--        if newValue>5000 then newValue=5000 end
        newValue=newValue/1000
        if newValue~=c['maxVel'] then
            c['maxVel']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
        adjustMaxVelocityMaxAcceleration()
    end
    updateMovementParamDisplay()
end

function accelerationChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
--        if newValue>35000 then newValue=35000 end
        newValue=newValue/1000
        if newValue~=c['maxAccel'] then
            c['maxAccel']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
            adjustMaxVelocityMaxAcceleration()
        end
    end
    updateMovementParamDisplay()
end

function dwellTimeChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0.01 then newValue=0.01 end
        if newValue>1 then newValue=1 end
        if newValue~=c['dwellTime'] then
            c['dwellTime']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function trackingTimeShiftChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<-1 then newValue=-1 end
        if newValue>1 then newValue=1 end
        if newValue~=c['trackingTimeShift'] then
            c['trackingTimeShift']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function pickOffsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['pickOffset']={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    updateMovementParamDisplay()
end

function placeOffsetChange_callback(ui,id,newVal)
    local c=readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['placeOffset']={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    updateMovementParamDisplay()
end

function pickRoundingChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=c['pickRounding'] then
            c['pickRounding']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function placeRoundingChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>200 then newValue=200 end
        newValue=newValue/1000
        if newValue~=c['placeRounding'] then
            c['placeRounding']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end


function pickNullingChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>50 then newValue=50 end
        newValue=newValue/1000
        if newValue~=c['pickNulling'] then
            c['pickNulling']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function placeNullingChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>50 then newValue=50 end
        newValue=newValue/1000
        if newValue~=c['placeNulling'] then
            c['placeNulling']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function pickApproachHeightChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<10 then newValue=10 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=c['pickApproachHeight'] then
            c['pickApproachHeight']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function placeApproachHeightChange_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<10 then newValue=10 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=c['placeApproachHeight'] then
            c['placeApproachHeight']=newValue
            writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    updateMovementParamDisplay()
end

function visualizeWorkspaceClick_callback(uiHandle,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],256)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-256
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    showHideWorkspace(newVal>0)
end

function visualizeWorkspaceSimClick_callback(uiHandle,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],512)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-512
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function visualizeTrajectoryClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function openFrameClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    openFrame(newVal~=0)
end

function adjustMaxVelocityMaxAcceleration()
    local c=readInfo()
    local mv,ma
    if c['motorType']==0 then
        -- Default motor
        mv=MAX_VEL_DEFAULT_MOTOR
        ma=MAX_ACCEL_DEFAULT_MOTOR
    end
    if c['motorType']==1 then
        -- High power motor
        mv=MAX_VEL_HIGHPOWER_MOTOR
        ma=MAX_ACCEL_HIGHPOWER_MOTOR
    end
    if c['maxVel']>mv then
        c['maxVel']=mv
        simBWF.markUndoPoint()
    end
    if c['maxAccel']>ma then
        c['maxAccel']=ma
        simBWF.markUndoPoint()
    end
    writeInfo(c)
end

function motorTypeChange_callback(uiHandle,id,newIndex)
    local newType=motorType_comboboxItems[newIndex+1][2]
    local c=readInfo()
    c['motorType']=newType
    writeInfo(c)
    simBWF.markUndoPoint()
    updateMotorTypeCombobox()
    adjustMaxVelocityMaxAcceleration()
    updateMovementParamDisplay()
--    updatePluginRepresentation()
end

function updateMotorTypeCombobox()
    local c=readInfo()
    local loc={{'standard',0},{'high-power',1}}
    motorType_comboboxItems=simBWF.populateCombobox(ui,95,loc,{},loc[c['motorType']+1][1],false,{})
end

function exteriorTypeChange_callback(uiHandle,id,newIndex)
    local newType=exteriorType_comboboxItems[newIndex+1][2]
    local c=readInfo()
    c['exteriorType']=newType
    writeInfo(c)
    
    local col={0.75,0.75,1}
    if newType==0 then
        col={1,1,1}
    end
    local s=sim.getObjectsInTree(model,sim.object_shape_type)
    for i=1,#s,1 do
        sim.setShapeColor(s[i],'RAGNAR_GEAR',sim.colorcomponent_ambient_diffuse,col)
    end
    
    simBWF.markUndoPoint()
    updateExteriorTypeCombobox()
    updatePluginRepresentation()
end

function updateExteriorTypeCombobox()
    local c=readInfo()
    local loc={{'standard',0},{'wash-down',1}}
    exteriorType_comboboxItems=simBWF.populateCombobox(ui,96,loc,{},loc[c['exteriorType']+1][1],false,{})
end

function frameTypeChange_callback(uiHandle,id,newIndex)
    local newType=frameType_comboboxItems[newIndex+1][2]
    local c=readInfo()
    c['frameType']=newType
    writeInfo(c)
    
    setFrameVisible(newType~=0)
    
    simBWF.markUndoPoint()
    updateFrameTypeCombobox()
    updatePluginRepresentation()
end

function updateFrameTypeCombobox()
    local c=readInfo()
    local loc={{'experimental',0},{'industrial',1}}
    frameType_comboboxItems=simBWF.populateCombobox(ui,97,loc,{},loc[c['frameType']+1][1],false,{})
end


function visibleFrameLowBeamsClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setLowBeamsVisible(newVal~=0)
end

function enabledClicked_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function showStatisticsClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    if sim.getSimulationState()~=sim.simulation_stopped then
        sim.callScriptFunction("ext_enableDisableStats_fromCustomizationScript@"..sim.getObjectName(model),sim.scripttype_childscript,newVal~=0)
    end
end

function ragnarIsIdle_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4096)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4096
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function attachPartClicked_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1024)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1024
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function pickWithoutTargetClicked_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2048)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2048
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function ip_callback(uiHandle,id,newValue)
    local c=readInfo()
    if c['connectionIp']~=newValue then
        c['connectionIp']=newValue
        simBWF.markUndoPoint()
        writeInfo(c)
    end
    simUI.setEditValue(ui,1200,c['connectionIp'],true)
end

function port_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>65525 then newValue=65525 end
        if c['connectionPort']~=newValue then
            c['connectionPort']=newValue
            simBWF.markUndoPoint()
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,1201,simBWF.format("%i",c['connectionPort']),true)
end

function timeout_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0.01 then newValue=0.01 end
        if newValue>10 then newValue=10 end
        if c['connectionTimeout']~=newValue then
            c['connectionTimeout']=newValue
            simBWF.markUndoPoint()
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,1202,simBWF.format("%.2f",c['connectionTimeout']),true)
end

function bufferSize_callback(uiHandle,id,newValue)
    local c=readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>10000 then newValue=10000 end
        if c['connectionBufferSize']~=newValue then
            c['connectionBufferSize']=newValue
            simBWF.markUndoPoint()
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,1203,simBWF.format("%i",c['connectionBufferSize']),true)
end

function connect_callback()
    connect()
end

function pause_callback()
    paused=true
    updateEnabledDisabledItems()
    enableMouseInteractionsOnPlot(true)
    if plotUi then
        simUI.setTitle(plotUi,sim.getObjectName(model)..' (paused)',true)
    end
end

function resume_callback()
    paused=false
    updateEnabledDisabledItems()
    enableMouseInteractionsOnPlot(false)
    if plotUi then
        simUI.setTitle(plotUi,sim.getObjectName(model)..' (online)',true)
    end
end

function disconnect_callback()
    disconnect()
end

enableMouseInteractionsOnPlot=function(enable)
    if plotUi then
        simUI.setMouseOptions(plotUi,1,enable,enable,enable,enable)
    end
end

function closePlot()
    if plotUi then
        local x,y=simUI.getPosition(plotUi)
        previousPlotDlgPos={x,y}
        local x,y=simUI.getSize(plotUi)
        previousPlotDlgSize={x,y}
        plotTabIndex=simUI.getCurrentTab(plotUi,77)
        simUI.destroy(plotUi)
        plotUi=nil
    end
end

function connect()
    connected=true
    updateEnabledDisabledItems()
    if sim.fastIdleLoop then
        sim.fastIdleLoop(true)
    else
        sim.setInt32Parameter(sim.intparam_idle_fps,0)
    end
    local c=readInfo()

    if not plotUi and sim.boolAnd32(c['bitCoded'],8192)>0 then
        local xml=[[<tabs id="77">
                <tab title="Axes angles">
                <plot id="1" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
                <tab title="Axes errors">
                <plot id="2" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
                <tab title="Axes velocity">
                <plot id="3" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
                <tab title="Platform velocity">
                <plot id="4" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
            </tabs>]]
        if not previousPlotDlgPos then
            previousPlotDlgPos="bottomRight"
        end
        plotUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' (online)',previousPlotDlgPos,true,"closePlot",false,true,false,nil,previousPlotDlgSize)
        simUI.setPlotLabels(plotUi,1,"Time (seconds)","degrees")
        if not plotTabIndex then
            plotTabIndex=0
        end
        simUI.setCurrentTab(plotUi,77,plotTabIndex,true)

        local curveStyle=simUI.curve_style.line
        local scatterShape={scatter_shape=simUI.curve_scatter_shape.none,scatter_size=5,line_size=1}
        simUI.addCurve(plotUi,1,simUI.curve_type.time,'axis1',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,1,simUI.curve_type.time,'axis2',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,1,simUI.curve_type.time,'axis3',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(plotUi,1,simUI.curve_type.time,'axis4',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(plotUi,1,true)
        simUI.addCurve(plotUi,2,simUI.curve_type.time,'axis1',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,2,simUI.curve_type.time,'axis2',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,2,simUI.curve_type.time,'axis3',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(plotUi,2,simUI.curve_type.time,'axis4',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(plotUi,2,true)
        simUI.addCurve(plotUi,3,simUI.curve_type.time,'axis1',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,3,simUI.curve_type.time,'axis2',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,3,simUI.curve_type.time,'axis3',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(plotUi,3,simUI.curve_type.time,'axis4',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(plotUi,3,true)
        simUI.addCurve(plotUi,4,simUI.curve_type.time,'X',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,4,simUI.curve_type.time,'Y',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(plotUi,4,simUI.curve_type.time,'Z',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(plotUi,4,simUI.curve_type.time,'Rot',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(plotUi,4,true)
    end
    memorizedMotorAngles={}
    memorizedMotorAngles[1]=sim.getJointPosition(fkDrivingJoints[1])
    memorizedMotorAngles[2]=sim.getJointPosition(fkDrivingJoints[2])
    memorizedMotorAngles[3]=sim.getJointPosition(fkDrivingJoints[3])
    memorizedMotorAngles[4]=sim.getJointPosition(fkDrivingJoints[4])
    setFkMode()

    blabla=0
    enableMouseInteractionsOnPlot(false)
    local data={}
    data.id=model
    data.ip=c.connectionIp
    data.port=c.connectionPort
    data.timeout=c.connectionTimeout
    data.bufferSize=c.connectionBufferSize
    simBWF.query('ragnar_connectReal',data)
end

function disconnect()
    if memorizedMotorAngles then
        moveToJointPositions(memorizedMotorAngles)
    end

    local data={}
    data.id=model
    simBWF.query('ragnar_disconnectReal',data)
    if plotUi then
        simUI.setTitle(plotUi,sim.getObjectName(model),true)
    end
    if sim.fastIdleLoop then
        sim.fastIdleLoop(false)
    else
        sim.setInt32Parameter(sim.intparam_idle_fps,8)
    end
    connected=false
    paused=false
    updateEnabledDisabledItems()
    enableMouseInteractionsOnPlot(true)
end

function showGraphClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8192)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8192
        closePlot()
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function reflectConfigClick_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16384)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16384
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function updatePlotAndRagnarFromRealRagnarIfNeeded()
    if connected and not paused then
        local c=readInfo()
        local data={}
        data.id=model
        data.stateCount=c.connectionBufferSize
        local result,retData=simBWF.query('ragnar_getRealStates',data)
        if plotUi then
            if result=='ok' then
                for i=1,4,1 do
                    local label='axis'..i
                    simUI.clearCurve(plotUi,1,label)
                    if #retData.timeStamps>0 then
                        local t={}
                        local x={}
                        for j=1,1000,1 do
                            t[j]=blabla+0.01*j
                            x[j]=math.sin(t[j]*(1+0.1*i))
                        end
                        simUI.addCurveTimePoints(plotUi,1,label,retData.timeStamps,retData.motorAngles[i])
                    end
                end
                simUI.rescaleAxesAll(plotUi,1,false,false)
                simUI.replot(plotUi,1)
            end
            --[[
            else
                -- To fake a signal
                if not blabla then
                    blabla=0
                end
                blabla=blabla+0.01
                for i=1,4,1 do
                    local label='axis'..i
                    simUI.clearCurve(plotUi,1,label)
                    local t={}
                    local x={}
                    for j=1,1000,1 do
                        t[j]=blabla+0.01*j
                        x[j]=math.sin(t[j]*(1+0.1*i))
                    end
                    simUI.addCurveTimePoints(plotUi,1,label,t,x)
                end
                simUI.rescaleAxesAll(plotUi,1,false,false)
                simUI.replot(plotUi,1)
            end
            --]]
        end
        if sim.boolAnd32(c['bitCoded'],16384)>0 then
            local desired={0,0,0,0}
            if result=='ok' then
                if #retData.timeStamps>0 then
                    desired[1]=retData.motorAngles[1][#retData.motorAngles[1]]*math.pi/180
                    desired[2]=retData.motorAngles[2][#retData.motorAngles[2]]*math.pi/180
                    desired[3]=retData.motorAngles[3][#retData.motorAngles[3]]*math.pi/180
                    desired[4]=retData.motorAngles[4][#retData.motorAngles[4]]*math.pi/180
                end
            end
            moveToJointPositions(desired)
        end
    end
end

function moveToJointPositions(desired)
    -- avoid too large steps, otherwise FK/IK doesn't work well
    local dx={}
    local current={}
    local md=0
    for i=1,4,1 do
        current[i]=sim.getJointPosition(fkDrivingJoints[i])
        dx[i]=desired[i]-current[i]
        if math.abs(dx[i])>md then
            md=math.abs(dx[i])
        end
    end
    local steps=math.ceil(0.01+md/(5*math.pi/180))
    for i=1,steps,1 do
        for j=1,4,1 do
            sim.setJointPosition(fkDrivingJoints[j],current[j]+i*dx[j]/steps)
        end
        sim.handleIkGroup(ikGroup)
    end
end

function algorithmClick_callback()
    local s="800 600"
    local p="100 100"
    if algoDlgSize then
        s=algoDlgSize[1]..' '..algoDlgSize[2]
    end
    if algoDlgPos then
        p=algoDlgPos[1]..' '..algoDlgPos[2]
    end
    local xml = [[
        <editor title="Pick and Place Algorithm" editable="true" searchable="true"
            tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190"
            selectionColor="128 128 255" size="]]..s..[[" position="]]..p..[["
            useVrepKeywords="true" isLua="true">
            <keywords2 color="255 100 100" >
                <item word="ragnar_getAllTrackedParts" autocomplete="true" calltip="table allTrackedParts=ragnar_getAllTrackedParts()" />
                <item word="ragnar_getDropLocationInfo" autocomplete="true" calltip="table locationInfo=ragnar_getDropLocationInfo(string destinationName)" />
                <item word="ragnar_moveToPickLocation" autocomplete="true" calltip="ragnar_moveToPickLocation(map part,bool attachPart,number stackingShift)" />
                <item word="ragnar_attachPart" autocomplete="true" calltip="ragnar_attachPart(map part)" />
                <item word="ragnar_detachPart" autocomplete="true" calltip="ragnar_detachPart()" />
                <item word="ragnar_stopTrackingPart" autocomplete="true" calltip="ragnar_stopTrackingPart(map part)" />
                <item word="ragnar_moveToDropLocation" autocomplete="true" calltip="ragnar_moveToDropLocation(map locationInfo,bool detachPart)" />
                <item word="ragnar_getAttachToTarget" autocomplete="true" calltip="bool attach=ragnar_getAttachToTarget()" />
                <item word="ragnar_getPickWithoutTarget" autocomplete="true" calltip="bool pickWithoutTarget=ragnar_getPickWithoutTarget()" />
                <item word="ragnar_getStacking" autocomplete="true" calltip="number stacking=ragnar_getStacking()" />
                
                <item word="ragnar_startPickTime" autocomplete="true" calltip="ragnar_startPickTime(bool isAuxiliaryWindow)" />
                <item word="ragnar_endPickTime" autocomplete="true" calltip="ragnar_endPickTime()" />
                <item word="ragnar_startPlaceTime" autocomplete="true" calltip="ragnar_startPlaceTime()" />
                <item word="ragnar_endPlaceTime" autocomplete="true" calltip="ragnar_endPlaceTime(bool isOtherLocation)" />
                <item word="ragnar_startCycleTime" autocomplete="true" calltip="ragnar_startCycleTime()" />
                <item word="ragnar_endCycleTime" autocomplete="true" calltip="ragnar_endCycleTime(bool didSomething)" />
                <item word="updateMotionParameters" autocomplete="true" calltip="updateMotionParameters()" />

                <item word="ragnar_getTrackingLocationInfo" autocomplete="true" calltip="ragnar_getTrackingLocationInfo(map locationInfo,number processingStage)" />
                <item word="ragnar_moveToTrackingLocation" autocomplete="true" calltip="ragnar_moveToTrackingLocation(map trackingLocationInfo,bool detachPart,bool attachPartToLocation)" />
                <item word="ragnar_incrementTrackedLocationProcessingStage" autocomplete="true" calltip="ragnar_incrementTrackedLocationProcessingStage(map trackingLocationInfo)" />
            </keywords2>

        </editor>
    ]]

    local c=readInfo()
    local initialText=c['algorithm']
    local modifiedText
    modifiedText,algoDlgSize,algoDlgPos=sim.openTextEditor(initialText,xml)
    c['algorithm']=modifiedText
    writeInfo(c)
    simBWF.markUndoPoint()
end

function updateEnabledDisabledItems()
    if ui then
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,2,simStopped,true)
        simUI.setEnabled(ui,92,simStopped,true)
        simUI.setEnabled(ui,300,simStopped,true)
        simUI.setEnabled(ui,301,simStopped,true)
        simUI.setEnabled(ui,95,simStopped,true)
        simUI.setEnabled(ui,96,simStopped,true)
        simUI.setEnabled(ui,97,simStopped,true)
        simUI.setEnabled(ui,303,simStopped,true)
  --      simUI.setEnabled(ui,304,simStopped,true)
        simUI.setEnabled(ui,306,simStopped,true)
        simUI.setEnabled(ui,3,simStopped,true)
        simUI.setEnabled(ui,305,simStopped,true)
        simUI.setEnabled(ui,20,simStopped,true)
        simUI.setEnabled(ui,39,simStopped,true)
        simUI.setEnabled(ui,21,simStopped,true)
        simUI.setEnabled(ui,22,simStopped,true)
        simUI.setEnabled(ui,77,simStopped,true)
        simUI.setEnabled(ui,501,simStopped,true)
        simUI.setEnabled(ui,502,simStopped,true)
        simUI.setEnabled(ui,503,simStopped,true)
        simUI.setEnabled(ui,504,simStopped,true)
        simUI.setEnabled(ui,2001,simStopped,true)

        local connectionAllowed=simStopped
        simUI.setEnabled(ui,1200,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1201,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1202,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1203,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1204,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1208,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1209,connectionAllowed and not connected,true)
        simUI.setEnabled(ui,1205,connectionAllowed and connected and not paused,true)
        simUI.setEnabled(ui,1206,connectionAllowed and connected and paused,true)
        simUI.setEnabled(ui,1207,connectionAllowed and connected,true)

    end
end


function staticPartWindowChange_callback(ui,id,newIndex)
    local newLoc=comboStaticPartWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICWINDOW1_REF,newLoc)
    simBWF.markUndoPoint()
    updateStaticWindowComboboxes()
end

function staticTargetWindowChange_callback(ui,id,newIndex)
    local newLoc=comboStaticTargetWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICTARGETWINDOW1_REF,newLoc)
    simBWF.markUndoPoint()
    updateStaticTargetWindowComboboxes()
end

function partTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=comboPartTrackingWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF,newLoc)
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF,-1)
    end
    simBWF.markUndoPoint()
    updateTrackingWindowComboboxes()
end

function auxPartTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=comboAuxPartTrackingWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF,newLoc)
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF,-1)
    end
    simBWF.markUndoPoint()
    updateTrackingWindowComboboxes()
end

function locationTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=comboLocationTrackingWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF,newLoc)
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF,-1)
    end
    simBWF.markUndoPoint()
    updateTrackingWindowComboboxes()
end

function auxLocationTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=comboAuxLocationTrackingWindow[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF,newLoc)
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF,-1)
    end
    if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)==newLoc then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF,-1)
    end
    simBWF.markUndoPoint()
    updateTrackingWindowComboboxes()
end

function updateDropLocationComboboxes()
    local loc=getAvailableDropLocations(false)
    comboDropLocations={}
    local exceptItems={}
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF))]=true
--    exceptItems[simBWF.NONE_TEXT]=nil
    comboDropLocations[1]=simBWF.populateCombobox(ui,501,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF))]=true
--    exceptItems[simBWF.NONE_TEXT]=nil
    comboDropLocations[2]=simBWF.populateCombobox(ui,502,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF))]=true
--    exceptItems[simBWF.NONE_TEXT]=nil
    comboDropLocations[3]=simBWF.populateCombobox(ui,503,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF))]=true
--    exceptItems[simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF))]=true
--    exceptItems[simBWF.NONE_TEXT]=nil
    comboDropLocations[4]=simBWF.populateCombobox(ui,504,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF)),true,{{simBWF.NONE_TEXT,-1}})
end

function dropLocationChange_callback(ui,id,newIndex)
    local newLoc=comboDropLocations[id-500][newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF+id-500-1,newLoc)
    for i=1,4,1 do
        if i~=id-500 then
            if simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF+i-1)==newLoc then
                simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF+i-1,-1)
            end
        end
    end
    simBWF.markUndoPoint()
    updateDropLocationComboboxes()
end

function updateTrackingWindowComboboxes()
    local loc=getAvailableTrackingWindows()
    local exceptItems={}
    comboPartTrackingWindow=simBWF.populateCombobox(ui,20,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
    comboAuxPartTrackingWindow=simBWF.populateCombobox(ui,39,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
    comboLocationTrackingWindow=simBWF.populateCombobox(ui,21,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)),true,{{simBWF.NONE_TEXT,-1}})

    exceptItems={}
    comboAuxLocationTrackingWindow=simBWF.populateCombobox(ui,22,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateStaticWindowComboboxes()
    local loc=getAvailableStaticPartWindows()
    local exceptItems={}
    comboStaticPartWindow=simBWF.populateCombobox(ui,505,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICWINDOW1_REF)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateStaticTargetWindowComboboxes()
    local loc=getAvailableStaticTargetWindows()
    local exceptItems={}
    comboStaticTargetWindow=simBWF.populateCombobox(ui,506,loc,exceptItems,simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICTARGETWINDOW1_REF)),true,{{simBWF.NONE_TEXT,-1}})
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="78">
    <tab title="General" layout="form">
                <label text="Enabled"/>
                <checkbox text="" on-change="enabledClicked_callback" id="1000"/>

                <label text="Maximum speed (mm/s)"/>
                <edit on-editing-finished="velocityChange_callback" id="10"/>

                <label text="Maximum acceleration (mm/s^2)"/>
                <edit on-editing-finished="accelerationChange_callback" id="11"/>

                <label text="Dwell time (s)"/>
                <edit on-editing-finished="dwellTimeChange_callback" id="12"/>

                <label text="Tracking time shift (s)"/>
                <edit on-editing-finished="trackingTimeShiftChange_callback" id="13"/>
    </tab>
    <tab title="Pick/Place">
            <group layout="form" flat="true">
                <label text="Pick approach height (mm)"/>
                <edit on-editing-finished="pickApproachHeightChange_callback" id="1007"/>

                <label text="Pick offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="pickOffsetChange_callback" id="1001"/>

                <label text="Pick rounding (mm)"/>
                <edit on-editing-finished="pickRoundingChange_callback" id="1003"/>

                <label text="Pick nulling accuracy (mm)"/>
                <edit on-editing-finished="pickNullingChange_callback" id="1005"/>
            </group>
            <group layout="form" flat="true">
                <label text="Place approach height (mm)"/>
                <edit on-editing-finished="placeApproachHeightChange_callback" id="1008"/>

                <label text="Place offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="placeOffsetChange_callback" id="1002"/>

                <label text="Place rounding (mm)"/>
                <edit on-editing-finished="placeRoundingChange_callback" id="1004"/>

                <label text="Place nulling accuracy (mm)"/>
                <edit on-editing-finished="placeNullingChange_callback" id="1006"/>
            </group>


            <group layout="form" flat="true">
                <label text="Pick and place algorithm"/>
                <button text="Edit"  on-click="algorithmClick_callback" id="403" />

                <label text="Pick also without target in sight"/>
                <checkbox text="" on-change="pickWithoutTargetClicked_callback" id="2001"/>

                <label text="Attach part to target"/>
                <checkbox text="" on-change="attachPartClicked_callback" id="2000"/>
            </group>
    </tab>
    <tab title="Configuration" layout="form">
                <label text="Static part window"/>
                <combobox id="505" on-change="staticPartWindowChange_callback">
                </combobox>

                <label text="Primary part tracking window"/>
                <combobox id="20" on-change="partTrackingWindowChange_callback">
                </combobox>

                <label text="Auxiliary part tracking window"/>
                <combobox id="39" on-change="auxPartTrackingWindowChange_callback">
                </combobox>

                <label text=""/>
                <label text=""/>

                <label text="Static target window"/>
                <combobox id="506" on-change="staticTargetWindowChange_callback">
                </combobox>
                
                <label text="Primary target tracking window"/>
                <combobox id="21" on-change="locationTrackingWindowChange_callback">
                </combobox>

                <label text="Auxiliary target tracking window"/>
                <combobox id="22" on-change="auxLocationTrackingWindowChange_callback">
                </combobox>

                <label text="Drop location 1"/>
                <combobox id="501" on-change="dropLocationChange_callback">
                </combobox>

                <label text="Drop location 2"/>
                <combobox id="502" on-change="dropLocationChange_callback">
                </combobox>

                <label text="Drop location 3"/>
                <combobox id="503" on-change="dropLocationChange_callback">
                </combobox>

                <label text="Drop location 4"/>
                <combobox id="504" on-change="dropLocationChange_callback">
                </combobox>
                
    </tab>
    <tab title="Robot">
            <group layout="form" flat="true">
                <label text="Primary arm length" id="1"/>
                <hslider tick-position="above" tick-interval="1" minimum="0" maximum="7" on-change="sizeAChange_callback" id="2"/>

                <label text="Secondary arm length" id="91"/>
                <hslider tick-position="above" tick-interval="1" minimum="0" maximum="17" on-change="sizeBChange_callback" id="92"/>

                <label text="Z position (mm)"/>
                <edit on-editing-finished="ZChange_callback" id="77"/>
                
                <label text="Motor type"/>
                <combobox id="95" on-change="motorTypeChange_callback"></combobox>

                <label text="Exterior type"/>
                <combobox id="96" on-change="exteriorTypeChange_callback"></combobox>

                <label text="Frame type"/>
                <combobox id="97" on-change="frameTypeChange_callback"></combobox>
                
                <label text="Frame is open"/>
                <checkbox text="" checked="false" on-change="openFrameClick_callback" id="301"/>

                <label text="Frame low beams are visible"/>
                <checkbox text="" checked="false" on-change="visibleFrameLowBeamsClick_callback" id="303"/>
            </group>
            <label text="" style="* {margin-left: 350px;}"/>
    </tab>
    <tab title="More" layout="form">
                <label text="Visualize workspace"/>
                <checkbox text="" checked="false" on-change="visualizeWorkspaceClick_callback" id="3"/>

                <label text="Visualize workspace also during simulation"/>
                <checkbox text="" checked="false" on-change="visualizeWorkspaceSimClick_callback" id="305"/>

                <label text="Visualize trajectory"/>
                <checkbox text="" checked="false" on-change="visualizeTrajectoryClick_callback" id="300"/>

                <label text="Show statistics"/>
                <checkbox text="" checked="false" on-change="showStatisticsClick_callback" id="304"/>

                <label text="Ragnar is slave (special)"/>
                <checkbox text="" checked="false" on-change="ragnarIsIdle_callback" id="306"/>
    </tab>
    <tab title="Online" layout="grid">
        <group flat="true" layout="form">
                <label text="IP address"/>
                <edit on-editing-finished="ip_callback" id="1200"/>

                <label text="Port"/>
                <edit on-editing-finished="port_callback" id="1201"/>

                <label text="Timeout (s)"/>
                <edit on-editing-finished="timeout_callback" id="1202"/>

                <label text="Buffer size (states)"/>
                <edit on-editing-finished="bufferSize_callback" id="1203"/>

                <label text="Show graph upon connection"/>
                <checkbox text="" checked="false" on-change="showGraphClick_callback" id="1208"/>

                <label text="Reflect Ragnar configuration upon connection"/>
                <checkbox text="" checked="false" on-change="reflectConfigClick_callback" id="1209"/>
        </group>
        <br/>
        <group flat="true">
                <button text="Connect" on-click="connect_callback" id="1204" />
                <button text="Pause" on-click="pause_callback" id="1205" />
                <button text="Resume" on-click="resume_callback" id="1206" />
                <button text="Disconnect" on-click="disconnect_callback" id="1207" />
        </group>
    </tab>
    </tabs>
        ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
        local c=readInfo()
        simUI.setSliderValue(ui,2,(c['primaryArmLengthInMM']-200)/50,true)
        simUI.setSliderValue(ui,92,(c['secondaryArmLengthInMM']-400)/50,true)
        simUI.setCheckboxValue(ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],256)~=0),true)
        simUI.setCheckboxValue(ui,305,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],512)~=0),true)
        simUI.setCheckboxValue(ui,300,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui,301,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(ui,303,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(ui,1000,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],64)~=0),true)
        simUI.setCheckboxValue(ui,304,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],128)~=0),true)
        simUI.setCheckboxValue(ui,306,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],4096)~=0),true)
        simUI.setCheckboxValue(ui,2000,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1024)~=0),true)
        simUI.setCheckboxValue(ui,2001,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],2048)~=0),true)
        simUI.setEditValue(ui,77,simBWF.format("%.0f",getZPosition()*1000),true)

        simUI.setEditValue(ui,1200,c['connectionIp'],true)
        simUI.setEditValue(ui,1201,simBWF.format("%i",c['connectionPort']),true)
        simUI.setEditValue(ui,1202,simBWF.format("%.2f",c['connectionTimeout']),true)
        simUI.setEditValue(ui,1203,simBWF.format("%i",c['connectionBufferSize']),true)
        simUI.setCheckboxValue(ui,1208,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],8192)~=0),true)
        simUI.setCheckboxValue(ui,1209,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],16384)~=0),true)

        updateStaticWindowComboboxes()
        updateStaticTargetWindowComboboxes()
        updateTrackingWindowComboboxes()
        updateDropLocationComboboxes()
        updateMotorTypeCombobox()
        updateExteriorTypeCombobox()
        updateFrameTypeCombobox()
        updateLinkLengthDisplay()
        updateMovementParamDisplay()
        updateEnabledDisabledItems()
        simUI.setCurrentTab(ui,78,dlgMainTabIndex,true)
    end
end

function showDlg()
    if not ui then
        createDlg()
    end
end

function removeDlg()
    if ui then
        if version>30301 or ( version==30301 and revision>=4 ) then
            local x,y=simUI.getPosition(ui)
            previousDlgPos={x,y}
            dlgMainTabIndex=simUI.getCurrentTab(ui,78)
        end
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    MAX_VEL_DEFAULT_MOTOR=5
    MAX_ACCEL_DEFAULT_MOTOR=35

    MAX_VEL_HIGHPOWER_MOTOR=2.5
    MAX_ACCEL_HIGHPOWER_MOTOR=25
    
    version=sim.getInt32Parameter(sim.intparam_program_version)
    revision=sim.getInt32Parameter(sim.intparam_program_revision)

    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    -- Following for backward compatibility:
    if _info['partTrackingWindow'] then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['partTrackingWindow']))
        _info['partTrackingWindow']=nil
    end
    if _info['auxPartTrackingWindow'] then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['auxPartTrackingWindow']))
        _info['auxPartTrackingWindow']=nil
    end
    if _info['targetTrackingWindow'] then
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['targetTrackingWindow']))
        _info['targetTrackingWindow']=nil
    end
    if _info['dropLocations'] then
        while #_info['dropLocations']>4 do
            table.remove(_info['dropLocations'])
        end
        while #_info['dropLocations']<4 do
            table.insert(_info['dropLocations'],simBWF.NONE_TEXT)
        end
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['dropLocations'][1]))
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION2_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['dropLocations'][2]))
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION3_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['dropLocations'][3]))
        simBWF.setReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION4_REF,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['dropLocations'][4]))
        _info['dropLocations']=nil
    end
    if _info['sizeA'] then
        local p1=200+math.floor(0.5+(_info['sizeA']-0.2005)/0.05)*50
        local p2=50*math.ceil((_info['sizeA']*_info['paramF']-0.001)/0.05)
        _info['primaryArmLengthInMM']=p1
        _info['secondaryArmLengthInMM']=p2
        _info['sizeA']=nil
        _info['paramF']=nil
    end
    if sim.boolAnd32(_info['bitCoded'],4)>0 then
        _info['bitCoded']=sim.boolOr32(_info['bitCoded'],4)-4
        _info['frameType']=1 -- industrial
    end
    if sim.boolAnd32(_info['bitCoded'],16)>0 then
        _info['bitCoded']=sim.boolOr32(_info['bitCoded'],16)-16
        _info['exteriorType']=1 -- wash-down
    end
    ----------------------------------------
    writeInfo(_info)
    adjustMaxVelocityMaxAcceleration()
    connected=false
    paused=false

    ikGroup=sim.getIkGroupHandle('Ragnar')
    ikTarget=sim.getObjectHandle('Ragnar_InvKinTarget')
    ikModeTipDummy=sim.getObjectHandle('Ragnar_InvKinTip')
    fkDrivingJoints={-1,-1,-1,-1}
    fkDrivingJoints[1]=sim.getObjectHandle('Ragnar_A1DrivingJoint1')
    fkDrivingJoints[2]=sim.getObjectHandle('Ragnar_A1DrivingJoint2')
    fkDrivingJoints[3]=sim.getObjectHandle('Ragnar_A1DrivingJoint3')
    fkDrivingJoints[4]=sim.getObjectHandle('Ragnar_A1DrivingJoint4')


    dlgMainTabIndex=0

    upperLinks={}
    lowerLinks={}

    upperArmAdjust={}
    lowerArmAdjust={}

    upperArmLAdjust={}
    lowerArmLAdjust={}

    frontAndRearCoverAdjust={sim.getObjectHandle('Ragnar_frontAdjust'),sim.getObjectHandle('Ragnar_rearAdjust')}
    middleCoverParts={}

    drivingJoints={}

    for i=1,4,1 do
        drivingJoints[#upperLinks+1]=sim.getObjectHandle('Ragnar_A1DrivingJoint'..i)

        upperLinks[#upperLinks+1]=sim.getObjectHandle('Ragnar_upperArmLink'..i-1)
        lowerLinks[#lowerLinks+1]=sim.getObjectHandle('Ragnar_lowerArmLinkA'..i-1)
        lowerLinks[#lowerLinks+1]=sim.getObjectHandle('Ragnar_lowerArmLinkB'..i-1)

        upperArmAdjust[#upperArmAdjust+1]=sim.getObjectHandle('Ragnar_upperArmAdjust'..i-1)
        lowerArmAdjust[#lowerArmAdjust+1]=sim.getObjectHandle('Ragnar_lowerArmAdjustA'..i-1)
        lowerArmAdjust[#lowerArmAdjust+1]=sim.getObjectHandle('Ragnar_lowerArmAdjustB'..i-1)

        upperArmLAdjust[#upperArmLAdjust+1]=sim.getObjectHandle('Ragnar_upperArmLAdjust'..i-1)
        lowerArmLAdjust[#lowerArmLAdjust+1]=sim.getObjectHandle('Ragnar_lowerArmLAdjustA'..i-1)
        lowerArmLAdjust[#lowerArmLAdjust+1]=sim.getObjectHandle('Ragnar_lowerArmLAdjustB'..i-1)
    end

    for i=1,5,1 do
        middleCoverParts[i]=sim.getObjectHandle('Ragnar_middleCover'..i)
    end

    frameModel=sim.getObjectHandle('Ragnar_frame')
    frameBeams={}
    for i=1,8,1 do
        frameBeams[i]=sim.getObjectHandle('Ragnar_frame_beam'..i)
    end
    frameJoints={}
    frameJoints[1]=sim.getObjectHandle('Ragnar_frame_widthJ1')
    frameJoints[2]=sim.getObjectHandle('Ragnar_frame_widthJ2')
    frameJoints[3]=sim.getObjectHandle('Ragnar_frame_heightJ1')
    frameJoints[4]=sim.getObjectHandle('Ragnar_frame_heightJ2')
    frameJoints[5]=sim.getObjectHandle('Ragnar_frame_heightJ3')
    frameJoints[6]=sim.getObjectHandle('Ragnar_frame_heightJ4')
    frameJoints[7]=sim.getObjectHandle('Ragnar_frame_lengthJ1')
    frameJoints[8]=sim.getObjectHandle('Ragnar_frame_lengthJ2')
    frameJoints[9]=sim.getObjectHandle('Ragnar_frame_lengthJ3')
    frameJoints[10]=sim.getObjectHandle('Ragnar_frame_lengthJ4')

    frameOpenClose={}
    for i=1,3,1 do
        frameOpenClose[i]=sim.getObjectHandle('Ragnar_frame_openCloseJ'..i)
    end

    workspace=sim.getObjectHandle('Ragnar_workspace')

	
    updatePluginRepresentation()
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
end

showOrHideUiIfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg()
    else
        removeDlg()
    end
end

function sysCall_nonSimulation()
    showOrHideUiIfNeeded()
    updatePlotAndRagnarFromRealRagnarIfNeeded()
end

function sysCall_sensing()
    if simJustStarted then
        updateEnabledDisabledItems()
    end
    simJustStarted=nil
    showOrHideUiIfNeeded()
end

function sysCall_suspend()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    updateEnabledDisabledItems()
    local c=readInfo()
    if sim.boolAnd32(c['bitCoded'],256)==256 then
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,0)
    end
end

function sysCall_beforeSimulation()
    disconnect()
    closePlot()
    simJustStarted=true
    local c=readInfo()
    local showWs=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],256+512)==256+512)
    if showWs then
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(workspace,sim.objintparam_visibility_layer,0)
    end
end

function sysCall_beforeInstanceSwitch()
    disconnect()
    closePlot()
    removeDlg()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    disconnect()
    closePlot()
    removeDlg()
    removeFromPluginRepresentation()
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end