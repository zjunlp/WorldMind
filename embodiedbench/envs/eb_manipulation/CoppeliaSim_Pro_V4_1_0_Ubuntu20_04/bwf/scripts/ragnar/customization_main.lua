function model.setFrameDoorState(s)
    simBWF.callCustomizationScriptFunction('model.ext.adjustFrame',model.handles.frameModel,nil,nil,nil,s)
end

function model.setFrameState(s)
    simBWF.callCustomizationScriptFunction('model.ext.adjustFrame',model.handles.frameModel,s,nil,nil,nil)
end

function model.hideHousing(hide)
    for i=1,#model.handles.housingItems,1 do
        local l=1
        if hide then
            l=0
        end
        sim.setObjectInt32Parameter(model.handles.housingItems[i],sim.objintparam_visibility_layer,l)
    end
end

function model.getAvailableTrackingWindows(pick)
    local theType=0
    if not pick then
        theType=1
    end
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.TRACKINGWINDOW)
        if data then
            data=sim.unpackTable(data)
            if data['type']==theType then -- 0 is for pick, 1 is for place
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.getAvailableFrames(pick)
    local theType=0
    if not pick then
        theType=1
    end
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.LOCATIONFRAME)
        if data then
            data=sim.unpackTable(data)
            if data['type']==theType then -- 0 is for pick, 1 is for place
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function model.getAvailableConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableInputBoxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.INPUTBOX)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getAvailableOutputBoxes()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.OUTPUTBOX)
        if data then
            retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
        end
    end
    return retL
end

function model.getModelInputOutputConnectionIndex(modelHandle,input)
    -- returns the connection index (1-6) if yes, otherwise -1:
    if modelHandle~=-1 then
        for i=1,8,1 do
            if input then
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)==modelHandle then
                    return i
                end
            else
                if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)==modelHandle then
                    return i
                end
            end
        end
    end
    return -1
end

function model.disconnectInputOrOutputBoxConnection(modelHandle,input)
    local refreshDlg=false
    if modelHandle~=-1 then
        for i=1,8,1 do
            if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)==modelHandle then
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1,-1)
                refreshDlg=true
                break
            end
            if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)==modelHandle then
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1,-1)
                refreshDlg=true
                break
            end
        end
    end
    if refreshDlg then
        model.dlg.refresh()
    end
end

function model.getPlatform()
    return model.platform
end

function model.getPlatformUniqueId()
    local retVal=''
    if model.platform>=0 then
        retVal=sim.getObjectStringParameter(model.platform,sim.objstringparam_unique_id)
    end
    return retVal
end

function model.getGripper()
    return model.gripper
end

function model.getGripperUniqueId()
    local retVal=''
    if model.gripper>=0 then
        retVal=sim.getObjectStringParameter(model.gripper,sim.objstringparam_unique_id)
    end
    return retVal
end

function model.updateWs()
    local gripper=model.getGripper()
    if gripper>=0 then
        local grData=sim.unpackTable(sim.readCustomDataBlock(gripper,simBWF.modelTags.RAGNARGRIPPER))
    
        local inf=model.readInfo()
        local primaryArmL=inf['primaryArmLengthInMM']/1000
        local secondaryArmL=inf['secondaryArmLengthInMM']/1000
        local alpha=sim.getJointPosition(model.handles.alphaOffsetJ1)
        local beta=sim.getJointPosition(model.handles.betaOffsetJ1)
        local ax=sim.getJointPosition(model.handles.xOffsetJ1)
        local ay=sim.getJointPosition(model.handles.yOffsetJ1)
        local r=grData.kinematricsParams[1] --    0.15
        local gamma1=grData.kinematricsParams[2] -- 0.5236 -- i.e. 30 deg
        local gamma2=grData.kinematricsParams[3] -- 2.0944 -- i.e. 120 deg

        local data={}
        data.details=1
        
        -- See the ragnar.pdf document in the repository for the meaning of following arm parameters.
        -- For now, the plugin expects arm in following order: 3,4,1,2
        -- Also, the plugin uses the negative of beta.
        local arm1Param={-ax, -ay, -alpha, -beta, primaryArmL, secondaryArmL, r, -gamma2}
        local arm2Param={ax, -ay, alpha, beta, primaryArmL, secondaryArmL, r, -gamma1}
        local arm3Param={ax, ay, -alpha, beta, primaryArmL, secondaryArmL, r, gamma1}
        local arm4Param={-ax, ay, alpha, -beta, primaryArmL, secondaryArmL, r, gamma2}
        local makeCorrections=true
        if makeCorrections then
            data.armParams={arm3Param,arm4Param,arm1Param,arm2Param}
            for i=1,4,1 do
                data.armParams[i][4]=data.armParams[i][4]*-1
            end
        else
            data.armParams={arm1Param,arm2Param,arm3Param,arm4Param}
        end
        
        local res,retDat=simBWF.query("ragnar_ws",data)
        sim.removePointsFromPointCloud(model.handles.ragnarWs,0,nil,0)
        sim.insertPointsIntoPointCloud(model.handles.ragnarWs,1,retDat.points)
    end
end

function model.adjustRobot()
---[[
    local inf=model.readInfo()
    local primaryArmLengthInMM=inf['primaryArmLengthInMM']
    local secondaryArmLengthInMM=inf['secondaryArmLengthInMM']

--    local a=0.2+((primaryArmLengthInMM-200)/50)*0.05+0.0005
    local a=primaryArmLengthInMM/1000
    local b=secondaryArmLengthInMM/1000
 

    local c=0.025
    local x=math.sqrt(a*a-c*c)
    local primaryAdjust=x-math.sqrt(0.3*0.3-c*c) -- Initial lengths are 300 and 550
    local secondaryAdjust=b-0.55
    local dx=a*28/30
    local ddx=dx-0.28

    for i=1,4,1 do
        sim.setJointPosition(model.handles.primaryArmsEndAdjust[i],primaryAdjust)
    end

    for i=1,8,1 do
        sim.setJointPosition(model.handles.secondaryArmsEndAdjust[i],secondaryAdjust)
    end


    for i=1,4,1 do
        sim.setJointPosition(model.handles.primaryArmsLAdjust[i],primaryAdjust*0.5)
    end

    for i=1,8,1 do
        sim.setJointPosition(model.handles.secondaryArmsLAdjust[i],secondaryAdjust*0.5)
    end

    for i=1,2,1 do
        sim.setJointPosition(model.handles.leftAndRightSideAdjust[i],dx)
    end

    -- Scale the central elements in the X-direction:
    for i=1,#model.handles.centralCover,1 do
        local h=model.handles.centralCover[i]
        local r,minX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
        local r,maxX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
        local s=maxX-minX
        local desiredXSize=((a*28/30)-0.233+0.025+0.03)*2
        if desiredXSize<0.049 then
            desiredXSize=0.05
        end
        sim.scaleObject(h,desiredXSize/s,1,1)
    end

    
    -- Scale the "Ragnar Robot" meshes:
    for i=1,2,1 do
        local h=model.handles.nameElement[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local d=0.3391
        if a<0.399 then
            d=0.22
        end
        if a<0.29 then
            d=0.1
        end
        if a<0.24 then
            d=0.03
        end
        --[[
        local p=sim.getObjectPosition(h,-1)
        if d/s>1.1 then
            p[1]=p[1]+
        end
        if d/s<0.9 then
        
        end
        --]]
        sim.scaleObject(h,d/s,d/s,d/s)
    end

    

    for i=1,4,1 do
        local h=model.handles.primaryArms[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local d=0.242+primaryAdjust
        sim.scaleObject(h,1,1,d/s)
    end

    for i=1,8,1 do
        local h=model.handles.secondaryArms[i]
        local r,minZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
        local r,maxZ=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
        local s=maxZ-minZ
        local r,minX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
        local r,maxX=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
        local sx=maxX-minX
        local d=0.5+secondaryAdjust
        local diam=0.01
        if d>=0.5 then
            diam=0.014
        end
        sim.scaleObject(h,diam/sx,diam/sx,d/s)
    end

    model.executeIk(true)

    -- The frame (width and height):
    local z=inf['frameHeightInMM']/1000
    simBWF.callCustomizationScriptFunction('model.ext.adjustFrame',model.handles.frameModel,nil,2*ddx+0.9525,z+0.349,nil)

    local p=sim.getObjectPosition(model.handles.ragnarRef,sim.handle_parent)
    sim.setObjectPosition(model.handles.ragnarRef,sim.handle_parent,{p[1],p[2],z})
    local p=sim.getObjectPosition(model.handles.frameModel,sim.handle_parent)
    sim.setObjectPosition(model.handles.frameModel,sim.handle_parent,{0,0,0})
    
    model.workspaceUpdateRequest=sim.getSystemTimeInMs(-1)
--]]
end

function model.setArmLength(primaryArmLengthInMM,secondaryArmLengthInMM)
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
    
    local c=model.readInfo()
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
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.adjustRobot()
    model.dlg.refresh()
end

function model.setObjectSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function model.adjustWsBox()
    local c=model.readInfo()
    local s={c.wsBox[2][1]-c.wsBox[1][1],c.wsBox[2][2]-c.wsBox[1][2],c.wsBox[2][3]-c.wsBox[1][3]}
    local p={(c.wsBox[2][1]+c.wsBox[1][1])/2,(c.wsBox[2][2]+c.wsBox[1][2])/2,(c.wsBox[2][3]+c.wsBox[1][3])/2}
    model.setObjectSize(model.handles.ragnarWsBox,s[1],s[2],s[3])
    sim.setObjectPosition(model.handles.ragnarWsBox,model.handles.ragnarRef,p)
end

function model.adjustMaxVelocityMaxAcceleration()
    local c=model.readInfo()
    local mv=C.MOTORTYPES[c.motorType].maxVel
    local ma=C.MOTORTYPES[c.motorType].maxAccel
    if c['maxVel']>mv then
        c['maxVel']=mv
        simBWF.markUndoPoint()
    end
    if c['maxAccel']>ma then
        c['maxAccel']=ma
        simBWF.markUndoPoint()
    end
    model.writeInfo(c)
end

function model.attachOrDetachReferencedItem(previousItem,newItem)
    -- We keep the tracking windows and location frames as orphans, otherwise we might run into trouble with the calibration balls, etc.
    --[[
    if previousItem>=0 then
        sim.setObjectParent(previousItem,-1,true) -- detach previous item
    end
    if newItem>=0 then
        sim.setObjectParent(newItem,model.handle,true) -- attach current item
    end
    --]]
end

function model.executeIk(platformInNominalConfig)
    if model.platform>=0 then
        if platformInNominalConfig then
            local inf=model.readInfo()
            local primaryArmLengthInMM=inf['primaryArmLengthInMM']
            local secondaryArmLengthInMM=inf['secondaryArmLengthInMM']
            sim.setObjectPosition(model.platform,sim.handle_parent,{0,0,-(secondaryArmLengthInMM-primaryArmLengthInMM)/1000-0.2})
            sim.setObjectOrientation(model.platform,sim.handle_parent,{0,0,0})
        end

        for i=1,4,1 do
            -- We handle each branch individually:
            local ld=sim.getLinkDummy(model.handles.ikTips[i])
            if ld>=0 then
                -- We make sure we don't perform too large jumps:
                local p=sim.getObjectPosition(model.handles.ikTips[i],ld)
                local l=math.sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3])
                local steps=math.ceil(0.00001+l/0.05)
                local start=sim.getObjectPosition(model.handles.ikTips[i],-1)
                local goal=sim.getObjectPosition(ld,-1)
                for j=1,steps,1 do
                    local t=j/steps
                    local pos={start[1]*(1-t)+goal[1]*t,start[2]*(1-t)+goal[2]*t,start[3]*(1-t)+goal[3]*t}
                    sim.setObjectPosition(ld,-1,pos)
                    sim.handleIkGroup(model.handles.ikGroups[i])
                end
            end
        end
    end
end

function model.checkAndHandlePlatformAttachment()
    -- Here we need to check and react to following situations:
    
    -- 1. The user deleted the gripper or platform via the del or cut operation
    -- 2. The user detached the gripper or platform via the assembly/desassembly toolbar button
    -- 3. The user attached the gripper or platform via the assembly/desassembly toolbar button
    --    Platform attachment happens in 2 steps normally:
    --    a) the user attaches the platform to the robot via the toolbar button
    --    b) this routine detects that, and attaches the platform to the attachement point of the robot, handles IK, etc.
    
    local previousPlatform=model.platform
    local previousGripper=model.gripper
    local previousPlatformUniqueId=model.platformUniqueId
    local previousGripperUniqueId=model.gripperUniqueId
    local action=false
    
    -- Check if we want to attach a new platform to Ragnar (independently of whether there is already a platform attached):
    local objs=sim.getObjectsInTree(model.handle,sim.handle_all,1+2)-- first children of Ragnar model base only
    local newPlatform=-1
    for i=1,#objs,1 do
        if sim.readCustomDataBlock(objs[i],simBWF.modelTags.RAGNARGRIPPERPLATFORM) then
            -- Yes!
            newPlatform=objs[i]
            break
        end
    end

    -- Now check if there is already a platform attached, remove it if we wanna attach a new platform:
    local currentPlatform=sim.getObjectChild(model.handles.ragnarGripperPlatformAttachment,0)
    if currentPlatform>=0 then
        -- Yes!
        if newPlatform>=0 then
            model.removeAndDeletePlatform() -- we remove the old platform, including a possible gripper attached to it. This also sets the arms into nominal configuration     
            currentPlatform=-1
            action=true
        end
    end
    
    if currentPlatform==-1 then
        -- Ok, we have no platform attached
        -- Check if a previous platform was detached but is still linked via IK (can happen if the user changes the parent of the platform):
        for i=1,#model.handles.ikTips,1 do
            local ld=sim.getLinkDummy(model.handles.ikTips[i])
            if ld>=0 then
                sim.removeObject(ld) -- Yes, we delete the linked dummy on that old platform
            end
        end
        
        -- Did we have a platform attached previously?
        if previousPlatform>=0 then
            -- yes! Set the arm joints into nominal position
            model.removeAndDeletePlatform()
            action=true
        end
        
        -- Did we want to attach a new platform?
        if newPlatform>=0 then
            -- Now correctly attach the new platform and create target dummies
            model.attachPlatformToEmptySpot(newPlatform)
            action=true
        end
    end
    
    -- If we didn't have any action here, update the gripper handle and ID
    if not action then
        if model.platform>=0 then
            model.gripper=simBWF.callCustomizationScriptFunction('model.ext.getGripper',model.platform)
            model.gripperUniqueId=model.getGripperUniqueId()
        else
            model.gripper=-1
            model.gripperUniqueId=''
        end
    end
    
    -- If something has changed, update the job data and serialized job-related models. Refresh the dialog:
    if previousPlatformUniqueId~=model.platformUniqueId or previousGripperUniqueId~=model.gripperUniqueId then
        model.updateJobDataFromCurrentSituation(model.currentJob)
        model.dlg.updateEnabledDisabledItems()
    end
end

function model.removeAndDeletePlatform()
    local platf=sim.getObjectChild(model.handles.ragnarGripperPlatformAttachment,0)
    if platf>=0 then
        -- remove the model:
        sim.removeModel(platf)
    end
     -- Set the arm joints into nominal position:
    for i=1,#model.handles.motorJoints,1 do
        local objs=sim.getObjectsInTree(model.handles.motorJoints[i],sim.object_joint_type,0)
        for j=1,#objs,1 do
            if sim.getJointMode(objs[j])==sim.jointmode_ik then
                sim.setJointPosition(objs[j],0)
            end
        end
    end
    
    model.platform=-1
    model.gripper=-1
    model.platformUniqueId=''
    model.gripperUniqueId=''
    model.dlg.updateEnabledDisabledItems()
end

function model.attachPlatformToEmptySpot(newPlatform)
    sim.setObjectParent(newPlatform,model.handles.ragnarGripperPlatformAttachment,true)
    local objs=sim.getObjectsInTree(newPlatform,sim.object_dummy_type,1)
    for i=1,#objs,1 do
        local data=sim.readCustomDataBlock(objs[i],simBWF.modelTags.RAGNARGRIPPERPLATFORMIKPT)
        if data then
            data=sim.unpackTable(data)
            local dum=sim.copyPasteObjects({objs[i]},0)[1]
            sim.setObjectParent(dum,objs[i],true)
            sim.setLinkDummy(model.handles.ikTips[data.index],dum)
            sim.setObjectInt32Parameter(dum,sim.dummyintparam_link_type,sim.dummy_linktype_ik_tip_target)
        end
    end
    model.platform=newPlatform
    model.gripper=simBWF.callCustomizationScriptFunction('model.ext.getGripper',model.platform)
    model.platformUniqueId=model.getPlatformUniqueId()
    model.gripperUniqueId=model.getGripperUniqueId()
    model.executeIk(true)
    model.dlg.updateEnabledDisabledItems()
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()
    
    model.adjustMaxVelocityMaxAcceleration()
    model.gripper=-1
    model.platform=sim.getObjectChild(model.handles.ragnarGripperPlatformAttachment,0)
    if model.platform>=0 then
        -- Dont' call simBWF.callCustomizationScriptFunction('model.ext.getGripper',model.platform)
        -- since that can trigger many other calls.
        local objs=sim.getObjectsInTree(model.platform,sim.handle_all,1)
        for i=1,#objs,1 do
            if sim.readCustomDataBlock(objs[i],simBWF.modelTags.RAGNARGRIPPER) then
                model.gripper=objs[i]
                break
            end
        end
    end
    model.platformUniqueId=model.getPlatformUniqueId()
    model.gripperUniqueId=model.getGripperUniqueId()
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.checkAndHandlePlatformAttachment()
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.checkAndHandlePlatformAttachment()
    model.updatePluginRepresentation()
    if model.workspaceUpdateRequest and sim.getSystemTimeInMs(model.workspaceUpdateRequest)>2000 then
        model.workspaceUpdateRequest=nil
        model.updateWs()
    end
--    model.applyCalibrationData() -- can potentially change the position/orientation of the robot
end

function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.dlg.showOrHideDlgIfNeeded()
    model.ext.outputPluginRuntimeMessages()
end

function sysCall_suspended()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_afterSimulation()
    if model.ragnarNormalM then
        sim.setObjectMatrix(model.handle,-1,model.ragnarNormalM)
        model.ragnarNormalM=nil
    end
    model.dlg.updateEnabledDisabledItems()
    local c=model.readInfo()
    if sim.boolAnd32(c['bitCoded'],256)==256 then
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,0)
    end
    if sim.boolAnd32(c['bitCoded'],1)==1 then
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,0)
    end
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    local c=model.readInfo()
    local showWs=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],256+512)==256+512)
    if showWs then
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,0)
    end
    local showWsBox=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1+4)==1+4)
    if showWsBox then
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,0)
    end
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        model.ragnarNormalM=sim.getObjectMatrix(model.handle,-1)
        model.applyCalibrationData() -- can potentially change the position/orientation of the robot
        model.setAttachedLocationFramesIntoCalibrationPose()
    end
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
