function ragnar_startPickTime()
    -- Call this just before starting a pick motion
    _timeMeasurement2Start=sim.getSimulationTime()
end

function ragnar_endPickTime(winType)
    -- Call this just after the end of a pick motion
    local dt=sim.getSimulationTime()-_timeMeasurement2Start
    _totalPickTime=_totalPickTime+dt
    _totalMovementTime=_totalMovementTime+dt
    local windowH=-1
    local avgPickTm
    if winType==2 then
        _auxTrackingWindowPickTime=(9*_auxTrackingWindowPickTime+dt)/10 -- kind of moving average
        windowH=auxPartTrackingWindowHandle
        avgPickTm=_auxTrackingWindowPickTime
    end
    if winType==1 then
        _trackingWindowPickTime=(9*_trackingWindowPickTime+dt)/10 -- kind of moving average
        windowH=partTrackingWindowHandle
        avgPickTm=_trackingWindowPickTime
    end
    if winType==0 then
        _trackingWindowPickTime=(9*_trackingWindowPickTime+dt)/10 -- kind of moving average
--        windowH=partTrackingWindowHandle
        avgPickTm=_trackingWindowPickTime
    end
    if windowH>=0 then
        local data=sim.readCustomDataBlock(windowH,simBWF.modelTags.TRACKINGWINDOW)
        data=sim.unpackTable(data)
        data['associatedRobotTrackingCorrectionTime']=avgPickTm
        sim.writeCustomDataBlock(windowH,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
    end
end

function ragnar_startPlaceTime()
    -- Call this just before starting a place motion
    _timeMeasurement2Start=sim.getSimulationTime()
end

function ragnar_endPlaceTime(isOtherLocation)
    -- Call this just after the end of a place motion
    local dt=sim.getSimulationTime()-_timeMeasurement2Start
    _totalPlaceTime=_totalPlaceTime+dt
    _totalMovementTime=_totalMovementTime+dt
    if isOtherLocation then
        _otherLocationPlaceTime=(9*_otherLocationPlaceTime+dt)/10 -- kind of moving average
    else
        _targetTrackingWindowPlaceTime=(9*_targetTrackingWindowPlaceTime+dt)/10 -- kind of moving average
        local correctionT=0
        if ragnar_getPickWithoutTarget() or stacking>1 then
            correctionT=_targetTrackingWindowPlaceTime
        else
            local cnt=0
            if _auxTrackingWindowPickTime>0 then
                correctionT=correctionT+_auxTrackingWindowPickTime
                cnt=cnt+1
            end
            if _trackingWindowPickTime>0 then
                correctionT=correctionT+_trackingWindowPickTime
                cnt=cnt+1
            end
            correctionT=correctionT/cnt
            correctionT=correctionT+_targetTrackingWindowPlaceTime
        end

        if locationTrackingWindowHandle>=0 then
            local data=sim.readCustomDataBlock(locationTrackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW)
            data=sim.unpackTable(data)
            data['associatedRobotTrackingCorrectionTime']=correctionT
            sim.writeCustomDataBlock(locationTrackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
        end
    end
end

function ragnar_startCycleTime()
    -- Call this just before starting (or trying to start) a pick and place cycle. A pick a place cycle is a 1+ pick and 1 place motion
    _timeMeasurement1Start=sim.getSimulationTime()
    _totalMovementTimeSaved=_totalMovementTime
end

function ragnar_endCycleTime(didSomething)
    -- Call this just after the end of a succeeded or failed pick and place motion
    local dt=sim.getSimulationTime()-_timeMeasurement1Start
    if didSomething then
        local movementTime=_totalMovementTime-_totalMovementTimeSaved
        local lossTime=dt-movementTime
        _cycleTime=(9*_cycleTime+dt)/10 -- kind of moving average
        _lossTime=(9*_lossTime+lossTime)/10 -- kind of moving average
        _totalCycleTime=_totalCycleTime+dt
    end
    if statUi then
        local t=sim.getSimulationTime()
        simUI.setLabelText(statUi,1,simBWF.format("Average cycle time: %.2f [s]",_cycleTime),true)
        simUI.setLabelText(statUi,3,simBWF.format("Average loss time: %.2f [s]",_lossTime),true)
        simUI.setLabelText(statUi,2,simBWF.format("Idle time: %.1f [%%]",100*(t-_totalCycleTime)/t),true)
    end
end

function ext_enableDisableStats_fromCustomizationScript(enableIt)
    if statUi then
        simUI.destroy(statUi)
        statUi=nil
    end
    prepareStatisticsDialog(enableIt)
end

function getToolHandleAndStacking()
    local toolAttachment=sim.getObjectHandle('ragnar_toolAttachment')
    local h=sim.getObjectChild(toolAttachment,0)
    if h>=0 then
        local data=sim.readCustomDataBlock(h,simBWF.modelTags.RAGNARGRIPPER)
        if data then
            data=sim.unpackTable(data)
            local s=data['stacking']
            local ss=data['stackingShift']
            if s<=1 then
                ss=0
            end
            return h,s,ss
        end
    end
    return toolAttachment,1,0
end

function getListOfSelectedLocationsOrBuckets(ragnarSettings)
    local retL={}
    for i=1,4,1 do
        local h=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_DROPLOCATION1_REF+i-1)
        if h>=0 then
            local data=sim.readCustomDataBlock(h,'XYZ_BUCKET_INFO')
            if data then
                data=sim.unpackTable(data)
                local dimension={data['width'],data['length'],data['height']}
                retL[#retL+1]={h,1,dimension} -- 0 is location, 1 is bucket
            else
                data=sim.readCustomDataBlock(h,simBWF.modelTags.OLDLOCATION)
                if data then
                    data=sim.unpackTable(data)
                    retL[#retL+1]={h,0,data['name']} -- 0 is location, 1 is bucket
                end
            end
        end
    end
    return retL
end

function ragnar_getDropLocationInfo(locationName)
    local effLoc={}
    local retV={}
    
    
    local ij=1
    local allLocationNames={}
    for token in (locationName..","):gmatch("([^,]*),") do
        token=token:gsub("%s+","") -- remove spaces
        allLocationNames[ij]=token
        ij=ij+1
    end
    
    
    for loc=1,#allSelectedLocationsOrBuckets,1 do
        if allSelectedLocationsOrBuckets[loc][2]==1 then
            -- This is a bucket
            -- Is that bucket the right bucket for the part, and is it operational? i.e. does it currently accept items?
            local data=sim.unpackTable(sim.readCustomDataBlock(allSelectedLocationsOrBuckets[loc][1],'XYZ_BUCKET_INFO'))
            
            local nameIsSame=false
            for i=1,#allLocationNames,1 do
                if allLocationNames[i]==data['locationWhenEmpty'] then
                    nameIsSame=true
                    break
                end
            end
            
            if nameIsSame and (data['status']=='needToFill') then
                -- yes
                local info={}
                local p=sim.getObjectPosition(allSelectedLocationsOrBuckets[loc][1],model)
                info['pos']=p
                info['isBucket']=true
                retV[#retV+1]=info
            end
            -- Here we need to make sure that we will ignore a possible location that was
            -- specified, and that coincides with the bucket:
            for ll=1,#allSelectedLocationsOrBuckets,1 do
            
            
                local nameIsSame=false
                for i=1,#allLocationNames,1 do
                    if allLocationNames[i]==allSelectedLocationsOrBuckets[ll][3] then
                        nameIsSame=true
                        break
                    end
                end
            
            
                if (allSelectedLocationsOrBuckets[ll][2]==0) and nameIsSame then
                    p=sim.getObjectPosition(allSelectedLocationsOrBuckets[ll][1],allSelectedLocationsOrBuckets[loc][1])
                    if math.sqrt(p[1]*p[1]+p[2]*p[2])<0.02 then
                        effLoc[allSelectedLocationsOrBuckets[ll][1]]=true
                    end
                end
            end
        end
    end
    for loc=1,#allSelectedLocationsOrBuckets,1 do
        if allSelectedLocationsOrBuckets[loc][2]==0 then
            -- This is a location
            -- The right location for the current part, and not yet used via a bucket?
            
            local nameIsSame=false
            for i=1,#allLocationNames,1 do
                if allLocationNames[i]==allSelectedLocationsOrBuckets[loc][3] then
                    nameIsSame=true
                    break
                end
            end
            
            if nameIsSame and (not effLoc[allSelectedLocationsOrBuckets[loc][1]]) then
                -- yes
                local info={}
                local p=sim.getObjectPosition(allSelectedLocationsOrBuckets[loc][1],model)
                info['pos']=p
                info['isBucket']=false
                retV[#retV+1]=info
            end
        end
    end
    return retV
end

function ragnar_getTrackingLocationInfo(locationName,processingStage)
    local ret=getAllTargetsInStaticWindow(staticTargetWindowHandle,locationName,processingStage)
    lastPlaceWindowIndex=2
    if #ret==0 then
        ret=getAllTargetsInTrackingWindow(locationTrackingWindowHandle,locationName,processingStage)
        lastPlaceWindowIndex=0
        if #ret==0 then
            ret=getAllTargetsInTrackingWindow(auxLocationTrackingWindowHandle,locationName,processingStage)
            lastPlaceWindowIndex=1
        end
    end
    return ret
end

function ragnar_incrementTrackedLocationProcessingStage(trackingLocation)
    local wind=-1
    if lastPlaceWindowIndex==0 then
        wind=locationTrackingWindowHandle
    end
    if lastPlaceWindowIndex==1 then
        wind=auxLocationTrackingWindowHandle
    end
    if lastPlaceWindowIndex==2 then
        wind=staticTargetWindowHandle
    end
    if wind>=0 then
        if lastPlaceWindowIndex==2 then
            local data=sim.readCustomDataBlock(wind,simBWF.modelTags.OLDSTATICPLACEWINDOW)
            data=sim.unpackTable(data)
            local tbl=data['targetPositionsToMarkAsProcessed']
            tbl[#tbl+1]=trackingLocation['dummyHandle']
            data['targetPositionsToMarkAsProcessed']=tbl
            sim.writeCustomDataBlock(wind,simBWF.modelTags.OLDSTATICPLACEWINDOW,sim.packTable(data))
        else
            local data=sim.readCustomDataBlock(wind,simBWF.modelTags.TRACKINGWINDOW)
            data=sim.unpackTable(data)
            local tbl=data['targetPositionsToMarkAsProcessed']
            tbl[#tbl+1]=trackingLocation['dummyHandle']
            data['targetPositionsToMarkAsProcessed']=tbl
            sim.writeCustomDataBlock(wind,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
        end
    end
end

getAllTargetsInTrackingWindow=function(trackingWindowHandle,locationName,processingStage)
    local ret={}
    if trackingWindowHandle>=0 then
        local data=sim.readCustomDataBlock(trackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW)
        data=sim.unpackTable(data)
        local trackedTargets=data['trackedTargetsInWindow']
        if trackedTargets then
            local m=sim.getObjectMatrix(model,-1)
            local windowTransfRot=sim.getObjectMatrix(trackingWindowHandle,-1)
            windowTransfRot[4]=0
            windowTransfRot[8]=0
            windowTransfRot[12]=0
            sim.invertMatrix(m)
            sim.invertMatrix(windowTransfRot)
            m[4]=0
            m[8]=0
            m[12]=0
            local cumulVelVectY=0
            local cnt=0
            
            local ij=1
            local allLocationNames={}
            for token in (locationName..","):gmatch("([^,]*),") do
                token=token:gsub("%s+","") -- remove spaces
                allLocationNames[ij]=token
                ij=ij+1
            end
            
            for key,value in pairs(trackedTargets) do
                local nameIsSame=false
                for i=1,#allLocationNames,1 do
                   if allLocationNames[i]==value['partName'] then
                        nameIsSame=true
                        break
                    end
                end
                if value['processingStage']==processingStage and nameIsSame then
                    local dat={}
                    dat['dummyHandle']=key
                    dat['partHandle']=value['partHandle']
                    local p=sim.getObjectPosition(key,model)
                    local pW=sim.getObjectPosition(key,trackingWindowHandle)
                    dat['pos']=p
                    local v=sim.multiplyVector(m,value['velocityVect'])
                    local w=sim.multiplyVector(windowTransfRot,value['velocityVect'])
                    dat['velocityVect']=v
                    dat['sort']=pW[2]
                    dat['ser']=value['ser']
                    ret[#ret+1]=dat
                    cumulVelVectY=cumulVelVectY+w[2]
                    cnt=cnt+1
                end
            end
            if #ret>1 then
                local ascending=true
                if math.abs(cumulVelVectY/cnt)<0.001 then
                    -- very slow. Try to use a previous direction
                    local done=false
                    if windowsAndDirections then
                        local val=windowsAndDirections[trackingWindowHandle]
                        if val then
                            ascending=(val<0)
                            done=true
                        end
                    end
                    if not done then
                        ascending=(cumulVelVectY/cnt<0)
                    end
                else
                    -- fast enough.
                    if not windowsAndDirections then
                        windowsAndDirections={}
                    end
                    windowsAndDirections[trackingWindowHandle]=cumulVelVectY/cnt
                    ascending=(cumulVelVectY/cnt<0)
                end
                if ascending then
                    table.sort(ret,function(a,b) if math.abs(a['sort']-b['sort'])>0.002 then return a['sort']<b['sort'] else return a['ser']>b['ser'] end end)
                else
                    table.sort(ret,function(a,b) if math.abs(a['sort']-b['sort'])>0.002 then return a['sort']>b['sort'] else return a['ser']>b['ser'] end end)
                end
            end
        end
    end
    return ret
end

getAllTargetsInStaticWindow=function(windowHandle,locationName,processingStage)
    local ret={}
    if windowHandle>=0 then
        local data=sim.readCustomDataBlock(windowHandle,'XYZ_STATICPLACEWINDOW_INFO')
        data=sim.unpackTable(data)
        local trackedTargets=data['trackedTargetsInWindow']
        local m=sim.getObjectMatrix(model,-1)
        local windowTransfRot=sim.getObjectMatrix(windowHandle,-1)
        windowTransfRot[4]=0
        windowTransfRot[8]=0
        windowTransfRot[12]=0
        sim.invertMatrix(m)
        sim.invertMatrix(windowTransfRot)
        m[4]=0
        m[8]=0
        m[12]=0
        local cumulVelVectY=0
        local cnt=0
        
        local ij=1
        local allLocationNames={}
        for token in (locationName..","):gmatch("([^,]*),") do
            token=token:gsub("%s+","") -- remove spaces
            allLocationNames[ij]=token
            ij=ij+1
        end
        
        for key,value in pairs(trackedTargets) do
            local nameIsSame=false
            for i=1,#allLocationNames,1 do
               if allLocationNames[i]==value['partName'] then
                    nameIsSame=true
                    break
                end
            end
            if value['processingStage']==processingStage and nameIsSame then
                local dat={}
                dat['dummyHandle']=key
                dat['partHandle']=value['partHandle']
                local p=sim.getObjectPosition(key,model)
                local pW=sim.getObjectPosition(key,windowHandle)
                dat['pos']=p
                local v=sim.multiplyVector(m,value['velocityVect'])
                local w=sim.multiplyVector(windowTransfRot,value['velocityVect'])
                dat['velocityVect']=v
                dat['sort']=pW[2]
                dat['ser']=value['ser']
                ret[#ret+1]=dat
                cumulVelVectY=cumulVelVectY+w[2]
                cnt=cnt+1
            end
        end
        if #ret>1 then
            local ascending=true
            if math.abs(cumulVelVectY/cnt)<0.001 then
                -- very slow. Try to use a previous direction
                local done=false
                if windowsAndDirections then
                    local val=windowsAndDirections[windowHandle]
                    if val then
                        ascending=(val<0)
                        done=true
                    end
                end
                if not done then
                    ascending=(cumulVelVectY/cnt<0)
                end
            else
                -- fast enough.
                if not windowsAndDirections then
                    windowsAndDirections={}
                end
                windowsAndDirections[windowHandle]=cumulVelVectY/cnt
                ascending=(cumulVelVectY/cnt<0)
            end
            if ascending then
                table.sort(ret,function(a,b) if math.abs(a['sort']-b['sort'])>0.002 then return a['sort']<b['sort'] else return a['ser']>b['ser'] end end)
            else
                table.sort(ret,function(a,b) if math.abs(a['sort']-b['sort'])>0.002 then return a['sort']>b['sort'] else return a['ser']>b['ser'] end end)
            end
        end
    end
    return ret
end


getAllPartsInTrackingWindow=function(trackingWindowHandle)
    local ret={}
    if trackingWindowHandle>=0 then
        local data=sim.readCustomDataBlock(trackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW)
        data=sim.unpackTable(data)
        local trackedParts=data['trackedItemsInWindow']
        -- Make all data relative to the robot's ref frame (is right now absolute):
        local transf=sim.getObjectMatrix(model,-1)
        local transfRot=sim.getObjectMatrix(model,-1)
        local windowTransf=sim.getObjectMatrix(trackingWindowHandle,-1)
        local windowTransfRot=sim.getObjectMatrix(trackingWindowHandle,-1)
        transfRot[4]=0
        transfRot[8]=0
        transfRot[12]=0
        windowTransfRot[4]=0
        windowTransfRot[8]=0
        windowTransfRot[12]=0
        sim.invertMatrix(transf)
        sim.invertMatrix(transfRot)
        sim.invertMatrix(windowTransf)
        sim.invertMatrix(windowTransfRot)
        local cumulVelVectY=0
        local cnt=0
        for key,value in pairs(trackedParts) do
            local ppos=sim.multiplyVector(transf,value['pickPos'])
            local pposW=sim.multiplyVector(windowTransf,value['pickPos'])
            value['pickPos']=ppos
            local v=sim.multiplyVector(transfRot,value['velocityVect'])
            local w=sim.multiplyVector(windowTransfRot,value['velocityVect'])
            value['velocityVect']=v
            value['normalVect']=sim.multiplyVector(transfRot,value['normalVect'])
            value['partHandle']=key -- we add that data
            value['sort']=pposW[2]
            ret[#ret+1]=value
            cumulVelVectY=cumulVelVectY+w[2]
            cnt=cnt+1
        end
        if #ret>1 then
            local ascending=true
            if math.abs(cumulVelVectY/cnt)<0.001 then
                -- very slow. Try to use a previous direction
                local done=false
                if windowsAndDirections then
                    local val=windowsAndDirections[trackingWindowHandle]
                    if val then
                        ascending=(val<0)
                        done=true
                    end
                end
                if not done then
                    ascending=(cumulVelVectY/cnt<0)
                end
            else
                -- fast enough.
                if not windowsAndDirections then
                    windowsAndDirections={}
                end
                windowsAndDirections[trackingWindowHandle]=cumulVelVectY/cnt
                ascending=(cumulVelVectY/cnt<0)
            end
            if ascending then
                table.sort(ret,function(a,b) return a['sort']<b['sort'] end)
            else
                table.sort(ret,function(a,b) return a['sort']>b['sort'] end)
            end
        end
    end
    return ret
end

getAllPartsInStaticWindow=function(windowHandle)
    local ret={}
    if windowHandle>=0 then
        local data=sim.readCustomDataBlock(windowHandle,'XYZ_STATICPICKWINDOW_INFO')
        data=sim.unpackTable(data)
        local trackedParts=data['trackedItemsInWindow']
        -- Make all data relative to the robot's ref frame (is right now absolute):
        local transf=sim.getObjectMatrix(model,-1)
        local transfRot=sim.getObjectMatrix(model,-1)
        local windowTransf=sim.getObjectMatrix(windowHandle,-1)
        local windowTransfRot=sim.getObjectMatrix(windowHandle,-1)
        transfRot[4]=0
        transfRot[8]=0
        transfRot[12]=0
        windowTransfRot[4]=0
        windowTransfRot[8]=0
        windowTransfRot[12]=0
        sim.invertMatrix(transf)
        sim.invertMatrix(transfRot)
        sim.invertMatrix(windowTransf)
        sim.invertMatrix(windowTransfRot)
        local cumulVelVectY=0
        local cnt=0
        for key,value in pairs(trackedParts) do
            local ppos=sim.multiplyVector(transf,value['pickPos'])
            value['pickPos']=ppos
            value['normalVect']=sim.multiplyVector(transfRot,value['normalVect'])
            value['partHandle']=key -- we add that data
            local relPos=sim.getObjectPosition(key,ikModeTipDummy)
            value['sort']=relPos[1]*relPos[1]+relPos[2]*relPos[2]+relPos[3]*relPos[3]
            ret[#ret+1]=value
            cnt=cnt+1
        end
        if #ret>1 then
            table.sort(ret,function(a,b) return a['sort']<b['sort'] end)
        end
    end
    return ret
end


removePartFromTrackingWindow=function(trackingWindowHandle,partHandle)
    if trackingWindowHandle>=0 then
        local data=sim.readCustomDataBlock(trackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW)
        data=sim.unpackTable(data)
        local tbl=data['itemsToRemoveFromTracking']
        tbl[#tbl+1]=partHandle
        data['itemsToRemoveFromTracking']=tbl

        -- Remove it also from current simulation step data, otherwise we might pick the same again:
        local trackedParts=data['trackedItemsInWindow']
        trackedParts[partHandle]=nil
        data['trackedItemsInWindow']=trackedParts

        sim.writeCustomDataBlock(trackingWindowHandle,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
    end
end

removePartFromStaticWindow=function(windowHandle,partHandle)
    if windowHandle>=0 then
        local data=sim.readCustomDataBlock(windowHandle,'XYZ_STATICPICKWINDOW_INFO')
        data=sim.unpackTable(data)
        local tbl=data['itemsToRemoveFromTracking']
        tbl[#tbl+1]=partHandle
        data['itemsToRemoveFromTracking']=tbl

        -- Remove it also from current simulation step data, otherwise we might pick the same again:
        local trackedParts=data['trackedItemsInWindow']
        trackedParts[partHandle]=nil
        data['trackedItemsInWindow']=trackedParts

        sim.writeCustomDataBlock(windowHandle,'XYZ_STATICPICKWINDOW_INFO',sim.packTable(data))
    end
end

freezeTrackingWindow=function(windowHandle)
    if windowHandle>=0 then
        local data=sim.readCustomDataBlock(windowHandle,simBWF.modelTags.TRACKINGWINDOW)
        data=sim.unpackTable(data)
        data['freezeStaticWindow']=true
        sim.writeCustomDataBlock(windowHandle,simBWF.modelTags.TRACKINGWINDOW,sim.packTable(data))
    end
end

ragnar_stopTrackingPart=function(part)
    local partHandle=part['partHandle']
    lastPartWinType=part['auxWin']
    if part['auxWin']==2 then
        removePartFromTrackingWindow(auxPartTrackingWindowHandle,partHandle)
    end
    if part['auxWin']==1 then
        removePartFromTrackingWindow(partTrackingWindowHandle,partHandle)
    end
    if part['auxWin']==0 then
        removePartFromStaticWindow(staticPartWindowHandle,partHandle)
    end
end

shiftStackingParts=function(theStackingShift)
    if #attachedParts>0 and theStackingShift>0 then
        local p=sim.getObjectPosition(attachedParts[1],model)
        p[3]=p[3]+theStackingShift
        sim.setObjectPosition(attachedParts[1],model,p)
    end
end


ragnar_attachPart=function(part)
    local partHandle=part['partHandle']
    if sim.isHandleValid(partHandle)>0 then
        local p=sim.getModelProperty(partHandle)
        if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
            -- We have a model
            p=sim.boolOr32(p,sim.modelproperty_not_dynamic)
            sim.setModelProperty(partHandle,p)
        else
            -- We have a shape
            sim.setObjectInt32Parameter(partHandle,sim.shapeintparam_static,1)
            sim.resetDynamicObject(partHandle)
        end
        if #attachedParts==0 then
            previousPartParent=sim.getObjectParent(partHandle)
            sim.setObjectParent(partHandle,ikModeTipDummy,true)
            attachedParts[1]=partHandle
        else
            if ragnar_getAttachToTarget() then
                local sens=sim.createForceSensor(0,{0,1,1,0,0},{0.001,1,1,0,0})
                sim.setObjectInt32Parameter(sens,sim.objintparam_visibility_layer,0) -- hidden
                sim.setObjectPosition(sens,attachedParts[1],{0,0,0})
                sim.setObjectParent(attachedParts[1],sens,true)
                sim.setObjectParent(sens,partHandle,true)
            else
                sim.setObjectParent(attachedParts[1],partHandle,true)
            end
            previousPartParent=sim.getObjectParent(partHandle)
            sim.setObjectParent(partHandle,ikModeTipDummy,true)
            table.insert(attachedParts,1,partHandle)
        end
    end
end

ragnar_detachPart=function()
    if #attachedParts>0 then
    
        if sim.isHandleValid(attachedParts[1])>0 then
            local p=sim.getModelProperty(attachedParts[1])
            if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
                -- We have a model
                p=sim.boolOr32(p,sim.modelproperty_not_dynamic)-sim.modelproperty_not_dynamic
                sim.setModelProperty(attachedParts[1],p)
            else
                -- We have a shape
                sim.setObjectInt32Parameter(attachedParts[1],sim.shapeintparam_static,0)
                sim.resetDynamicObject(attachedParts[1])
            end
            sim.setObjectParent(attachedParts[1],previousPartParent,true)
            for i=2,#attachedParts,1 do
                -- the child parts (in the stacking):
                local p=sim.getModelProperty(attachedParts[i])
                if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
                    -- We have a model
                    p=sim.boolOr32(p,sim.modelproperty_not_dynamic)-sim.modelproperty_not_dynamic
                    sim.setModelProperty(attachedParts[i],p)
                else
                    -- We have a shape
                    sim.setObjectInt32Parameter(attachedParts[i],sim.shapeintparam_static,0)
                    sim.resetDynamicObject(attachedParts[i])
                end
                
                if ragnar_getAttachToTarget() then
                    -- Give up ownership of the child parts in the stacking:
                    local data=sim.readCustomDataBlock(attachedParts[i],simBWF.modelTags.PART)
                    data=sim.unpackTable(data)
                    data['giveUpOwnership']=true
                    sim.writeCustomDataBlock(attachedParts[i],simBWF.modelTags.PART,sim.packTable(data))
                else
                    sim.setObjectParent(attachedParts[i],previousPartParent,true)
                end
            end
            if #attachedParts>1 and (sim.boolAnd32(attachedParts[1],sim.modelproperty_not_model)>0) and ragnar_getAttachToTarget() then
                -- We need to turn the main stack parent into model:
                local p=sim.getModelProperty(attachedParts[1])
                p=sim.boolOr32(p,sim.modelproperty_not_model)-sim.modelproperty_not_model
                sim.setModelProperty(attachedParts[1],p)
            end
            attachedParts={}
        else
            attachedParts={}
        end

    end
end

attachPart1ToPart2=function(part1,part2)
    if (sim.isHandleValid(part1)>0) and (sim.isHandleValid(part2)>0) then
        local f=sim.createForceSensor(0,{0,1,1,0,0},{0.001,0,0,0,0})
        sim.setObjectInt32Parameter(f,sim.objintparam_visibility_layer,256)
        sim.setObjectPosition(f,part1,{0,0,0})
        sim.setObjectParent(part1,f,true)
        sim.setObjectParent(f,part2,true)
        
        local data=sim.unpackTable(sim.readCustomDataBlock(part1,simBWF.modelTags.PART))
        data['giveUpOwnership']=true
        sim.writeCustomDataBlock(part1,simBWF.modelTags.PART,sim.packTable(data))
        
        local objs=sim.getObjectsInTree(part1,sim.object_shape_type)
        for i=1,#objs,1 do
            local r,p=sim.getObjectInt32Parameter(part2,sim.shapeintparam_respondable_mask)
            p=sim.boolAnd32(p,65535-255)
            sim.setObjectInt32Parameter(objs[i],sim.shapeintparam_respondable_mask,p)
            sim.resetDynamicObject(objs[i])
        end
    end
end

handleKinematics=function()
    local res=sim.handleIkGroup(mainIkTask)
    if res==sim.ikresult_fail then
        if kinematicsFailedDialogHandle==-1 then
            kinematicsFailedDialogHandle=sim.displayDialog("IK failure report","IK solver failed.",sim.dlgstyle_message,false,"",nil,{1,0.8,0,0,0,0})
        end
    else
        if kinematicsFailedDialogHandle~=-1 then
            sim.endDialog(kinematicsFailedDialogHandle)
            kinematicsFailedDialogHandle=-1
        end
    end
end

setFkMode=function()
    -- disable the platform positional constraints:
    sim.setIkElementProperties(mainIkTask,ikModeTipDummy,0)
    -- Set the driving joints into passive mode (not taken into account during IK resolution):
    sim.setJointMode(fkDrivingJoints[1],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[2],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[3],sim.jointmode_passive,0)
    sim.setJointMode(fkDrivingJoints[4],sim.jointmode_passive,0)
    -- In FK mode, we want Ik to be handled automatically
    sim.setExplicitHandling(mainIkTask,0)
    sim.switchThread()
end

setIkMode=function()
    sim.switchThread()
    -- In IK mode, we want Ik to be handled in this script:
    sim.setExplicitHandling(mainIkTask,1)
    sim.switchThread()
    -- Make sure the target dummy has the same pose as the tip dummy:
    sim.setObjectPosition(ikModeTargetDummy,ikModeTipDummy,{0,0,0})
    sim.setObjectOrientation(ikModeTargetDummy,ikModeTipDummy,{0,0,0})
    -- enable the platform positional constraints:
    sim.setIkElementProperties(mainIkTask,ikModeTipDummy,sim.ik_x_constraint+sim.ik_y_constraint+sim.ik_z_constraint+sim.ik_alpha_beta_constraint+sim.ik_gamma_constraint)
    -- Set the base joints into ik mode (taken into account during IK resolution):
    sim.setJointMode(fkDrivingJoints[1],sim.jointmode_ik,0)
    sim.setJointMode(fkDrivingJoints[2],sim.jointmode_ik,0)
    sim.setJointMode(fkDrivingJoints[3],sim.jointmode_ik,0)
    sim.setJointMode(fkDrivingJoints[4],sim.jointmode_ik,0)
end

--[[
RobMove = function(blend,nulling)
    
    mDone = 0
    while (mDone < 3) do
        local dt=sim.getSimulationTimeStep()

        local rmlObj=sim.rmlPos(4,0.0001,-1,curPVA,MaxVAJ,selVec,tarPV)
        res,nextPVA=sim.rmlStep(rmlObj,dt)
        sim.rmlRemove(rmlObj)
--        res,nextPVA,syncTime =simRMLPosition(4,dt,-1,curPVA,MaxVAJ,selVec,tarPV)

        newPos = {nextPVA[1],nextPVA[2],nextPVA[3]}
        sim.setObjectPosition(ikModeTargetDummy,model,newPos)
        sim.setObjectOrientation(ikModeTargetDummy,model,{0,0,nextPVA[4]})
        handleKinematics()
        --sim.setObjectOrientation(ikModeTargetDummy2,model,{0,0,-nextPVA[4]})
        --newPos[4] = -newPos[4]
        --sim.setObjectPosition(ikModeTargetDummy2,model,newPos)
        curPVA = nextPVA;
    --    txt = simBWF.format(" newPos ( %.2f,%.2f,%.2f) %.2f",newPos[1],newPos[2],newPos[3],dist2go)
        --sim.addStatusbarMessage(txt)
        dist2go = math.sqrt((nextPVA[1]-tarPV[1])*(nextPVA[1]-tarPV[1]) + (tarPV[2]-nextPVA[2])*(tarPV[2]-nextPVA[2]) + (tarPV[3]-nextPVA[3])*(tarPV[3]-nextPVA[3]))--*1000.0
        if (mDone == 0) and(dist2go < blend) then
            mDone =1    
            dist2go = 100.0--denom*proMov
            tarPV[3] = tarPV[3]-appHight
        end    
        if (mDone == 1) and(dist2go < nulling) then
            mDone =2    
            dist2go = 100.0--denom*proMov
            tarPV[3] = tarPV[3]+appHight
        end
        if (mDone == 2) and(dist2go < blend) then
            mDone =3
        end
        sim.switchThread() -- Important, in order to have the thread synchronized with the simulation loop!
    end
end
--]]
createDummyToFollowWithOffset=function(parentDummy,posOffset)
    local m=sim.getObjectMatrix(toolHandle,-1)
    local dummyHandleToFollow=sim.createDummy(0.001)
    sim.setObjectInt32Parameter(dummyHandleToFollow,sim.objintparam_visibility_layer,1024)
    local v={posOffset[1]*m[1]+posOffset[2]*m[2]+posOffset[3]*m[3],posOffset[1]*m[5]+posOffset[2]*m[6]+posOffset[3]*m[7],posOffset[1]*m[9]+posOffset[2]*m[10]+posOffset[3]*m[11]}
    local p=sim.getObjectPosition(parentDummy,-1)
    sim.setObjectPosition(dummyHandleToFollow,-1,{p[1]+v[1],p[2]+v[2],p[3]+v[3]})
    sim.setObjectOrientation(dummyHandleToFollow,parentDummy,{0,0,0})
    sim.setObjectParent(dummyHandleToFollow,parentDummy,true)
    return dummyHandleToFollow
end

RobPick = function(partData,attachPart,theStackingShift,approachHeight,blend,nulling,dwTime)
    local version=sim.getInt32Parameter(sim.intparam_program_version)
    local _dummyHandleToFollow=partData['dummyHandle']
    local dummyHandleToFollow=createDummyToFollowWithOffset(_dummyHandleToFollow,pickOffset)

    if attachPart then
        shiftStackingParts(theStackingShift)    
    end
    
    mDone = 0
    app = approachHeight
    while (mDone < 4) do
        local dt=sim.getSimulationTimeStep()
        partPos = sim.getObjectPosition(dummyHandleToFollow,model)
        partOr =  sim.getObjectOrientation(dummyHandleToFollow,model)
        partVel,partorV = sim.getObjectVelocity(dummyHandleToFollow)
        if version<=30302 then -- this to fix a problem in versions prior to V3.4.0
            if version>0 then
                version=0
            else
                version=30400
            end
            partVel={0,0,0}
            partorV={0,0,0}
        end
--partOr[3]/angularGain
        partPV = {partPos[1]+partVel[1]*dt,partPos[2]+partVel[2]*dt,partPos[3]+partVel[3]*dt+app,0.0,
             partVel[1]*dt,partVel[2]*dt,partVel[3]*dt,0.0}
        
        local rmlObj=sim.rmlPos(4,0.0001,-1,curPVA,MaxVAJ,selVec,partPV)
        res,nextPVA=sim.rmlStep(rmlObj,dt)
        sim.rmlRemove(rmlObj)

        newPos = {nextPVA[1],nextPVA[2],nextPVA[3]}
        sim.setObjectPosition(ikModeTargetDummy,model,newPos)
--ikModeTipDummy
        sim.setObjectOrientation(ikModeTargetDummy,model,{0,0,nextPVA[4]})
        handleKinematics()
        curPVA = nextPVA;
    --    txt = simBWF.format(" newPos ( %.2f,%.2f,%.2f) %.2f",newPos[1],newPos[2],newPos[3],dist2go)
        --sim.addStatusbarMessage(txt)
        dist2go = math.sqrt((nextPVA[1]-partPV[1])*(nextPVA[1]-partPV[1]) + (partPV[2]-nextPVA[2])*(partPV[2]-nextPVA[2]) + (partPV[3]-nextPVA[3])*(partPV[3]-nextPVA[3]))--*1000.0
        if (mDone == 0) and(dist2go < blend) then
            mDone =1    
            dist2go = 100.0--denom*proMov
            app = .0
        end    
        if (mDone == 1) and(dist2go < nulling) then
            mDone =2    
            dist2go = 100.0--denom*proMov
            t2=sim.getSimulationTime()+dwTime
            if attachPart then
                ragnar_attachPart(partData)
            end
        end
        if (mDone == 2) and(t2 < sim.getSimulationTime()) then
            mDone =3
            dist2go = 100.0--denom*proMov
            app = approachHeight
        end
        if (mDone == 3) and(dist2go < blend) then
            mDone =4    
        end
        sim.switchThread() -- Important, in order to have the thread synchronized with the simulation loop!
    end
    sim.removeObject(dummyHandleToFollow)
end

RobPlace = function(TrackPart,detachPart,approachHeight,blend,nulling,dwTime,attachToTrackingLocation)
    local version=sim.getInt32Parameter(sim.intparam_program_version)
    local dummyHandleToFollow=createDummyToFollowWithOffset(TrackPart,placeOffset)
    mDone = 0
    app = approachHeight
    inFront = 0.0
    placeHeight = 0.01
    while (mDone < 4) do
        local dt=sim.getSimulationTimeStep()
        if( mDone < 3 ) then -- only update unitl picked 
            partPos = sim.getObjectPosition(dummyHandleToFollow,model)
           -- local RobPos = sim.getObjectPosition(model,-1)
            partPos[1] = partPos[1]--dropPos[1]-RobPos[1] --pickupHeight+
            partPos[2] = partPos[2]--dropPos[2]-RobPos[2] --pickupHeight+
            partPos[3] = partPos[3]---0.65+placeHeight --pickupHeight+
            partOr = sim.getObjectOrientation(dummyHandleToFollow,model)
            partVel,partorV = sim.getObjectVelocity(dummyHandleToFollow)
            if version<=30302 then -- this to fix a problem in versions prior to V3.4.0
                if version>0 then
                    version=0
                else
                    version=30400
                end
                partVel={0,0,0}
                partorV={0,0,0}
            end
        end
        partPV = {partPos[1]+partVel[1]*dt,partPos[2]+partVel[2]*dt + inFront,partPos[3]+partVel[3]*dt+app,0.0,
             partVel[1]*dt,partVel[2]*dt,partVel[3]*dt,0.0} --partOr[3]/angularGain
      --  if (partVel[1] > linearVelocity/4/dt) or (partVel[2] > linearVelocity/4/dt) or (partVel[3] > linearVelocity/4/dt) then
        --    PickOK = false
          --  sim.addStatusbarMessage("too fast part")
            --return
        --end


        local rmlObj=sim.rmlPos(4,0.0001,-1,curPVA,MaxVAJ,selVec,partPV)
        res,nextPVA=sim.rmlStep(rmlObj,dt)
        sim.rmlRemove(rmlObj)




        newPos = {nextPVA[1],nextPVA[2],nextPVA[3]}
        sim.setObjectPosition(ikModeTargetDummy,model,newPos)
        --sim.setObjectOrientation(ikModeTargetDummy,-1,{0,0,nextPVA[4]})
        handleKinematics()
        --sim.setObjectOrientation(ikModeTargetDummy2,model,{0,0,-nextPVA[4]})
        --newPos[4] = -newPos[4]
        --sim.setObjectPosition(ikModeTargetDummy2,model,newPos)
        curPVA = nextPVA;
    --    txt = simBWF.format(" newPos ( %.2f,%.2f,%.2f) %.2f",newPos[1],newPos[2],newPos[3],dist2go)
        --sim.addStatusbarMessage(txt)
        dist2go = math.sqrt((nextPVA[1]-partPV[1])*(nextPVA[1]-partPV[1]) + (partPV[2]-nextPVA[2])*(partPV[2]-nextPVA[2]) + (partPV[3]-nextPVA[3])*(partPV[3]-nextPVA[3]))--*1000.0
        if (mDone == 0) and(dist2go < blend) then
            mDone =1    
            dist2go = 100.0--denom*proMov
            app = .0
        end    
        if (mDone == 1) and(dist2go < nulling) then
            mDone =2    
            dist2go = 100.0--denom*proMov
            t2=sim.getSimulationTime()+dwTime
            if detachPart then
                local attachedPartSaved=attachedParts[1]
                ragnar_detachPart()
                if attachToTrackingLocation then
                    attachPart1ToPart2(attachedPartSaved,attachToTrackingLocation)
                end
            end
--            detachPart(partHandleToPick)
           -- sim.setScriptSimulationParameter(sim.getScriptAssociatedWithObject(suctionPad),'active','false')
        end
        if (mDone == 2) and(t2 < sim.getSimulationTime()) then
            mDone =3
            dist2go = 100.0--denom*proMov
            app = approachHeight
        end
        if (mDone == 3) and(dist2go < blend) then
            mDone =4    
        end
        sim.switchThread() -- Important, in order to have the thread synchronized with the simulation loop!
    end
    sim.removeObject(dummyHandleToFollow)
end

ragnar_moveToPickLocation=function(partData,attachPart,theStackingShift)
    local partHandle=partData['partHandle']
    lastPartWinType=partData['auxWin']
    if partData['auxWin']==2 then
        freezeTrackingWindow(auxPartTrackingWindowHandle)
    end
    if partData['auxWin']==1 then
        freezeTrackingWindow(partTrackingWindowHandle)
    end
    --[[
    if partData['auxWin']==0 then
        freezeStaticWindow(staticPartWindowHandle)
    end
    --]]
    cycleStartTime=sim.getSimulationTime()
    RobPick(partData,attachPart,theStackingShift,pickApproachHeight,pickRounding,pickNulling,dwellTime)
end

ragnar_moveToDropLocation=function(dropLocationInfo,detachPart)
    local dropPos=dropLocationInfo['pos']

    xth = 0*math.pi/180 --0.0--
 --   tarPV = {dropPos[1],dropPos[2],dropPos[3],xth,0.0,0.0,0.0,0.0}
 --   RobMove(0.05,0.005)
    local dropDum=sim.createDummy(0.001)
    sim.setObjectInt32Parameter(dropDum,sim.objintparam_visibility_layer,0)
    sim.setObjectPosition(dropDum,model,dropPos)
    local p=sim.getObjectPosition(dropDum,-1)
 --   p[3]=0.9 -- hard-code the drop height for now
    sim.setObjectPosition(dropDum,-1,p)
    RobPlace(dropDum,detachPart,placeApproachHeight,placeRounding,placeNulling,dwellTime)
    sim.removeObject(dropDum)
end

ragnar_moveToTrackingLocation=function(trackingLocationInfo,detachPart,attachToTrackingLocation)
    local partHandle=-1
    if attachToTrackingLocation then
        partHandle=trackingLocationInfo['partHandle']
    end
    RobPlace(trackingLocationInfo['dummyHandle'],detachPart,placeApproachHeight,placeRounding,placeNulling,dwellTime,partHandle)
end

ragnar_getAllTrackedParts=function()
    local r=getAllPartsInTrackingWindow(partTrackingWindowHandle)
    for i=1,#r,1 do
        r[i]['auxWin']=1
    end
    local r2=getAllPartsInTrackingWindow(auxPartTrackingWindowHandle)
    for i=1,#r2,1 do
        r2[i]['auxWin']=2
        r[#r+1]=r2[i]
    end
    local r3=getAllPartsInStaticWindow(staticPartWindowHandle)
    for i=1,#r3,1 do
        r3[i]['auxWin']=0
        r[#r+1]=r3[i]
    end
    return r
end

ragnar_getAttachToTarget=function()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    data=sim.unpackTable(data)
    return(sim.boolAnd32(data['bitCoded'],1024)>0)
end

ragnar_getStacking=function()
    return stacking,stackingShift
end

ragnar_getPickWithoutTarget=function()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    data=sim.unpackTable(data)
    return(sim.boolAnd32(data['bitCoded'],2048)>0)
end

ragnar_getEnabled=function()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    data=sim.unpackTable(data)
    return(sim.boolAnd32(data['bitCoded'],64)>0)
end

--[[
ragnar_getAllTrackedLocations=function()
    return getAllPartsInTrackingWindow(locationTrackingWindowHandle)
end
--]]

prepareStatisticsDialog=function(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="Average cycle time: 0.00 [s]" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                <label id="3" text="Average loss time: 0.00 [s]" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                <label id="2" text="Idle time: 100.0 [%]" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end
--[[
updateStatisticsDialog=function(cycleTime,auxCycleTime,totalCycleTime)
    if statUi then
        if totalCycleTime>0 then
            local ct=0
            local cnt=0
            if cycleTime then
                ct=ct+cycleTime
                cnt=cnt+1
            end
            if auxCycleTime then
                ct=ct+auxCycleTime
                cnt=cnt+1
            end
            ct=ct/cnt
            local t=sim.getSimulationTime()-startTime
            simUI.setLabelText(statUi,1,simBWF.format("Average cycle time: %.2f [s]",ct),true)
            simUI.setLabelText(statUi,2,simBWF.format("Idle time: %.1f [%%]",100*(t-totalCycleTime)/t),true)
        else
            simUI.setLabelText(statUi,1,"Average cycle time: 0.00 [s]",true)
            simUI.setLabelText(statUi,2,"Idle time: 100.0 [%]",true)
        end
    end
end
--]]
updateMotionParameters=function()
    -- 1. Read the current motion settings for the Ragnar:
    local ragnarSettings=sim.unpackTable(sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR))
    local mVel=ragnarSettings['maxVel']
    local mAccel=ragnarSettings['maxAccel']
    MaxVAJ = {mVel,mVel,mVel,angularVelocity,mAccel,mAccel,mAccel,angularAccel,2000,2000,2000,1000} -- pos,vel,acc128*dt
--    local trackingTimeShift=ragnarSettings['trackingTimeShift']
    
    dwellTime=ragnarSettings['dwellTime']

    -- Read the current tool offsets:
    pickOffset=ragnarSettings['pickOffset']
    placeOffset=ragnarSettings['placeOffset']
    -- And rounding, nulling and approachHeight:
    pickRounding=ragnarSettings['pickRounding']
    placeRounding=ragnarSettings['placeRounding']
    pickNulling=ragnarSettings['pickNulling']
    placeNulling=ragnarSettings['placeNulling']
    pickApproachHeight=ragnarSettings['pickApproachHeight']
    placeApproachHeight=ragnarSettings['placeApproachHeight']

    -- 2. Update the max vel/accel/jerk vector:
end

function sysCall_threadmain()
    -- Begin of the thread code:
    sim.setThreadAutomaticSwitch(false)
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    mainIkTask=sim.getIkGroupHandle('Ragnar')
    ikModeTipDummy=sim.getObjectHandle('Ragnar_InvKinTip')
    ikModeTargetDummy=sim.getObjectHandle('Ragnar_InvKinTarget')
    -- Following are the joints that we control when in FK mode:
    fkDrivingJoints={-1,-1,-1,-1}
    fkDrivingJoints[1]=sim.getObjectHandle('Ragnar_A1DrivingJoint1')
    fkDrivingJoints[2]=sim.getObjectHandle('Ragnar_A1DrivingJoint2')
    fkDrivingJoints[3]=sim.getObjectHandle('Ragnar_A1DrivingJoint3')
    fkDrivingJoints[4]=sim.getObjectHandle('Ragnar_A1DrivingJoint4')
    -- Following are the joints that we control when in IK mode (we use joints in order to be able to use the sim.moveToJointPositions command here too):
    ikDrivingJoints={-1,-1,-1,-1}
    ikDrivingJoints[1]=sim.getObjectHandle('Ragnar_T_X')
    ikDrivingJoints[2]=sim.getObjectHandle('Ragnar_T_Y')
    ikDrivingJoints[3]=sim.getObjectHandle('Ragnar_T_X')
    ikDrivingJoints[4]=sim.getObjectHandle('Ragnar_T_TH')

    local ragnarSettings=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    ragnarSettings=sim.unpackTable(ragnarSettings)
    toolHandle,stacking,stackingShift=getToolHandleAndStacking()

    staticPartWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICWINDOW1_REF)
    partTrackingWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING1_REF)
    auxPartTrackingWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_PARTTRACKING2_REF)
    staticTargetWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_STATICTARGETWINDOW1_REF)
    locationTrackingWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING1_REF)
    auxLocationTrackingWindowHandle=simBWF.getReferencedObjectHandle(model,simBWF.OLDRAGNAR_TARGETTRACKING2_REF)
    allSelectedLocationsOrBuckets=getListOfSelectedLocationsOrBuckets(ragnarSettings)
    lastPartWinType=1
    attachedParts={}


    _totalPickTime=0
    _totalPlaceTime=0
    _totalMovementTime=0
    _totalCycleTime=0

    -- Following 6 are the moving averages:
    _auxTrackingWindowPickTime=0
    _trackingWindowPickTime=0
    _targetTrackingWindowPlaceTime=0
    _otherLocationPlaceTime=0
    _cycleTime=0
    _lossTime=0 


    if sim.boolAnd32(ragnarSettings['bitCoded'],4096)>0 then
        setFkMode()
        simRemoteApi.start(19999)
    else



    totalCycleTime=0
    --totalCycles=0
    prepareStatisticsDialog(sim.boolAnd32(ragnarSettings['bitCoded'],128)>0)

    kinematicsFailedDialogHandle=-1
    angularVelocity=2.84*math.pi
    angularAccel=15.5*math.pi
    linearVelocity=4--0.1        
    linearAccel=12.5--0.145
    angularGain =0.0 -- FixedPlatfom1

    --proMov = 0.9 -- compeltion of approach/depart
    --appHight = .1 --meters

    -- First, make sure we are in initial position:
    setFkMode()
    sim.moveToJointPositions(fkDrivingJoints,{0,0,0,0},angularVelocity,angularAccel)
    setIkMode()
    initialPosition=sim.getObjectPosition(ikModeTipDummy,model)

    --Now with rotation rotation part - meaning DOF 4
    curPVA = {initialPosition[1],initialPosition[2],initialPosition[3],0.0,
              0.0,0.0,0.0,0.0,
              0.0,0.0,0.0,0.0} -- pos,vel,acc
    nextPVA = curPVA
    MaxVAJ = {linearVelocity,linearVelocity,linearVelocity,angularVelocity,
                linearAccel,linearAccel,linearAccel,angularAccel,
                2000,2000,2000,1000} -- pos,vel,acc128*dt
    selVec = {1,1,1,1}


        -- 3. Do we have a tracking window and a pick-and-place algorithm?
        local pickAndPlaceAlgo=ragnarSettings['algorithm']

        if (partTrackingWindowHandle>=0 or auxPartTrackingWindowHandle>=0 or staticPartWindowHandle>=0) and pickAndPlaceAlgo then
            -- 4. Load and run the pick-and-place algorithm:
            local algo=assert(loadstring(pickAndPlaceAlgo))
    --        cycleStartTime=-1
            algo() -- We stay in here until the end
    --[[
            if cycleStartTime>=0 then
                if not startTime then
                    startTime=cycleStartTime
                end
                didSomething=true
                local thisCycleTime=sim.getSimulationTime()-cycleStartTime
                totalCycleTime=totalCycleTime+thisCycleTime
                if lastPartWinType then
                    if auxCycleTime then
                        auxCycleTime=(9*auxCycleTime+thisCycleTime)/10 -- kind of moving average calc
                    else
                        auxCycleTime=thisCycleTime
                    end
                else
                    if cycleTime then
                        cycleTime=(9*cycleTime+thisCycleTime)/10 -- kind of moving average calc
                    else
                        cycleTime=thisCycleTime
                    end
                end
            end
            --]]
        end

        --[[
        -- 5. Avoid using too much processor time when idle
        if not didSomething then
            sim.switchThread()
        end

        updateStatisticsDialog(cycleTime,auxCycleTime,totalCycleTime)
        communicateCycleTimeToAssociatedTrackingWindows(cycleTime,auxCycleTime) 
        --]]
    --end
    end
end

