function model.simulationPause(isPause)
    if model.robotPlot.ui then
        if isPause and lastDataFromRagnar then
            model.robotPlot.setData(lastDataFromRagnar,1)
            model.robotPlot.setData(lastDataFromRagnar,2)
            model.robotPlot.setData(lastDataFromRagnar,3)
        end
        simUI.setMouseOptions(model.robotPlot.ui,1,isPause,isPause,isPause,isPause)
        simUI.setMouseOptions(model.robotPlot.ui,2,isPause,isPause,isPause,isPause)
        simUI.setMouseOptions(model.robotPlot.ui,3,isPause,isPause,isPause,isPause)
    end
    if model.clearancePlot.ui then
        if isPause and model.lastClearanceData then
            model.clearancePlot.setData(model.lastClearanceData,1)
        end
        simUI.setMouseOptions(model.clearancePlot.ui,1,isPause,isPause,isPause,isPause)
    end
end

function model.executeIk()
    if model.platform>=0 then
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

function model.setPlatformPose(pos,orient)
    if model.platform>=0 then
        sim.setObjectPosition(model.platform,sim.handle_parent,pos)
        sim.setObjectOrientation(model.platform,sim.handle_parent,orient)
        model.executeIk()
    end
end

function model.getPlatformPose()
    if model.platform>=0 then
        local pos=sim.getObjectPosition(model.platform,sim.handle_parent)
        local orient=sim.getObjectOrientation(model.platform,sim.handle_parent)
        return pos,orient
    end
end

function model.enableRagnar()
    if not savedJoints then
        savedJoints={}
        local allJoints=sim.getObjectsInTree(model.handle,sim.object_joint_type,1)
        for i=1,#allJoints,1 do
            savedJoints[allJoints[i]]=sim.getJointPosition(allJoints[i])
        end    
        local data={}
        data.id=model.handle
        data.bufferSize=model.connectionBufferSize
        simBWF.query('ragnar_connect',data)
    end
end

function model.disableRagnar()
    if savedJoints then
        local data={}
        data.id=model.handle
        simBWF.query('ragnar_disconnect',data)
        for key,value in pairs(savedJoints) do
            sim.setJointPosition(key,value)
        end
        savedJoints=nil
    end
end

function model.getAndApplyRagnarState()
    if savedJoints then
    
        if not gripperActionBuffer then
            gripperActionBuffer={}
            gripperActionBuffer.close={t={},v={}}
            gripperActionBuffer.open={t={},v={}}
        end

        local getData=false
        if model.online then    
            local dt=sim.getSystemTimeInMs(model.lastMoveVisualizeUpdateTimeInMs)
            if dt>moveVisUpdateFrequMs then
                getData=true
                model.lastMoveVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)
            end
        else
            local t=(sim.getSimulationTime()+sim.getSimulationTimeStep())*1000
            if t+1>model.lastMoveVisualizeUpdateTimeInMs+moveVisUpdateFrequMs then
                getData=true
                model.lastMoveVisualizeUpdateTimeInMs=t
            end
        end
        
        if getData then
            local updatePlot=false
            if model.online then    
                local dt=sim.getSystemTimeInMs(model.lastPlotVisualizeUpdateTimeInMs)
                if dt>plotVisUpdateFrequMs then
                    updatePlot=true
                    model.lastPlotVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)
                end
            else
                local t=(sim.getSimulationTime()+sim.getSimulationTimeStep())*1000
                if t+1>model.lastPlotVisualizeUpdateTimeInMs+plotVisUpdateFrequMs then
                    updatePlot=true
                    model.lastPlotVisualizeUpdateTimeInMs=t
                end
            end
        
            if not model.showTrajectory then
                sim.addDrawingObjectItem(model.graspCloseDrawingObject,nil) -- empty the cont.
                sim.addDrawingObjectItem(model.graspOpenDrawingObject,nil) -- empty the cont.
            end

            local data={}
            data.id=model.handle
            data.stateCount=model.connectionBufferSize
            data.posMultiplier=1000 -- for string buffers only!
            data.angleMultiplier=180/math.pi -- for string buffers only!
            local res,retData=simBWF.query('ragnar_getStates',data)
            
            if res=='ok' then
                dataFromRagnar=retData
            else
                if simBWF.isInTestMode() then
                    -- Generate fake data:
                    if not blabla then
                        blabla=0
                    end
                    blabla=blabla+0.05
                    dataFromRagnar={}
                    dataFromRagnar.timeStamps={}
                    dataFromRagnar.motorAngles={{},{},{},{},{}}
                    dataFromRagnar.motorErrors={{},{},{},{},{}}
                    dataFromRagnar.platformPose={{},{},{},{},{},{}}
                    local blabli=0
                    for i=1,model.connectionBufferSize,1 do
                        dataFromRagnar.timeStamps[i]=blabla+0.01*i
                        for j=1,5,1 do
                            dataFromRagnar.motorAngles[j][i]=math.sin(dataFromRagnar.timeStamps[i]*(1+0.1*j))
                            dataFromRagnar.motorErrors[j][i]=math.sin(dataFromRagnar.timeStamps[i]*(1+0.1*j))
                        end
                        
                        blabli=blabli+0.05
                        dataFromRagnar.platformPose[1][i]=math.sin(blabli+blabla*1.0)*model.primaryArmLengthInMM/1000
                        dataFromRagnar.platformPose[2][i]=math.sin(blabli+blabla*1.3)*model.primaryArmLengthInMM*0.7/1000
                        dataFromRagnar.platformPose[3][i]=-(model.secondaryArmLengthInMM-model.primaryArmLengthInMM+300)/1000+math.sin(blabla*0.6)*0.1
                        dataFromRagnar.platformPose[4][i]=0
                        dataFromRagnar.platformPose[5][i]=0
                        dataFromRagnar.platformPose[6][i]=0
                    end
                    dataFromRagnar.stateCount=model.connectionBufferSize
                    
                    -- Pack the data:
                    dataFromRagnar.timeStamps=sim.packFloatTable(dataFromRagnar.timeStamps)
                    for i=1,5,1 do
                        dataFromRagnar.motorAngles[i]=sim.packFloatTable(dataFromRagnar.motorAngles[i])
                        dataFromRagnar.motorErrors[i]=sim.packFloatTable(dataFromRagnar.motorErrors[i])
                    end
                    for i=1,6,1 do
                        dataFromRagnar.platformPose[i]=sim.packFloatTable(dataFromRagnar.platformPose[i])
                    end
                    
                else
                    dataFromRagnar=nil
                end
            end
            if dataFromRagnar then
                -- Unpack the data and scale it appropriately:
                dataFromRagnar.timeStamps=sim.unpackFloatTable(dataFromRagnar.timeStamps)
                for i=1,5,1 do
                    dataFromRagnar.motorAngles[i]=sim.transformBuffer(dataFromRagnar.motorAngles[i],sim.buffer_float,data.angleMultiplier,0,sim.buffer_float)
                    dataFromRagnar.motorAngles[i]=sim.unpackFloatTable(dataFromRagnar.motorAngles[i])
                end
                for i=1,5,1 do
                    dataFromRagnar.motorErrors[i]=sim.transformBuffer(dataFromRagnar.motorErrors[i],sim.buffer_float,data.angleMultiplier,0,sim.buffer_float)
                    dataFromRagnar.motorErrors[i]=sim.unpackFloatTable(dataFromRagnar.motorErrors[i])
                end
                for i=1,3,1 do
                    dataFromRagnar.platformPose[i]=sim.transformBuffer(dataFromRagnar.platformPose[i],sim.buffer_float,data.posMultiplier,0,sim.buffer_float)
                    dataFromRagnar.platformPose[i]=sim.unpackFloatTable(dataFromRagnar.platformPose[i])
                end
                for i=4,6,1 do
                    dataFromRagnar.platformPose[i]=sim.transformBuffer(dataFromRagnar.platformPose[i],sim.buffer_float,data.angleMultiplier,0,sim.buffer_float)
                    dataFromRagnar.platformPose[i]=sim.unpackFloatTable(dataFromRagnar.platformPose[i])
                end
            end
            
            if dataFromRagnar and #dataFromRagnar.timeStamps>0 then
                local pp=dataFromRagnar.platformPose
                local dataCnt=#dataFromRagnar.timeStamps
                
                if updatePlot then -- clearance
                   if model.previousClearanceFlag and model.previousClearanceEveryStepFlag then
                        model.lastClearanceData.times={}
                        model.lastClearanceData.clearances={}
                        local collection1=model.handles.robotArmCollection
                        if model.previousClearanceIncludePlatformFlag then
                            collection1=model.handles.robotArmAndPlatformCollection
                        end
                        for i=1,dataCnt,1 do
                            local pos={pp[1][i]/data.posMultiplier,pp[2][i]/data.posMultiplier,pp[3][i]/data.posMultiplier}
                            local orient={pp[4][i]/data.angleMultiplier,pp[5][i]/data.angleMultiplier,pp[6][i]/data.angleMultiplier}
                            model.setPlatformPose(pos,orient)
                            local res,distData=sim.checkDistance(collection1,model.handles.robotObstaclesCollection,0)
                            if res>0 then
                                model.lastClearanceData.times[i]=dataFromRagnar.timeStamps[i]
                                model.lastClearanceData.clearances[i]=distData[7]
                            end
                        end
                    end
                end
                
                
                local currentPos={pp[1][dataCnt]/data.posMultiplier,pp[2][dataCnt]/data.posMultiplier,pp[3][dataCnt]/data.posMultiplier}
                local currentOrient={pp[4][dataCnt]/data.angleMultiplier,pp[5][dataCnt]/data.angleMultiplier,pp[6][dataCnt]/data.angleMultiplier}
                model.setPlatformPose(currentPos,currentOrient)
                if updatePlot then
                    sim.addDrawingObjectItem(model.trajectoryDrawingObject,nil) -- empty the cont.
                    if model.showTrajectory then
                        local m=sim.getObjectMatrix(model.handles.ragnarRef,-1)
                        local w={dataFromRagnar.platformPose[1][dataCnt]/data.posMultiplier,dataFromRagnar.platformPose[2][dataCnt]/data.posMultiplier,dataFromRagnar.platformPose[3][dataCnt]/data.posMultiplier}
                        w=sim.multiplyVector(m,w)
                        local mm=math.max(dataCnt-100,1)
                        for i=dataCnt-1,mm,-1 do
                            local v={dataFromRagnar.platformPose[1][i]/data.posMultiplier,dataFromRagnar.platformPose[2][i]/data.posMultiplier,dataFromRagnar.platformPose[3][i]/data.posMultiplier}
                            v=sim.multiplyVector(m,v)
                            local dt=dataFromRagnar.timeStamps[i+1]-dataFromRagnar.timeStamps[i]
                            local l=simBWF.getPtPtDistance(v,w)
                            local speed=0
                            if dt>0 then
                                speed=l/dt
                            end
                            local c=model.getColorFromIntensity(speed/maxSpeed)
                            local data={v[1],v[2],v[3],w[1],w[2],w[3],c[1],c[2],c[3]}
                            sim.addDrawingObjectItem(model.trajectoryDrawingObject,data)
                            w=v
                        end
                    end
                    
                    if model.robotPlot.ui then
                        dataFromRagnar.gripperOpen=gripperActionBuffer.open
                        dataFromRagnar.gripperClose=gripperActionBuffer.close
                        model.robotPlot.setData(dataFromRagnar,simUI.getCurrentTab(model.robotPlot.ui,77)+1)
                        lastDataFromRagnar=dataFromRagnar
                    end
                end
            end
        end
        -- Now take care of the gripper actions (in each simulation step):
        if not dataFromRagnar then
            dataFromRagnar={}
        end
        local data={}
        data.id=model.handle
        local res,retData=simBWF.query('ragnar_getGripperAction',data)
        if res~='ok' then
            if simBWF.isInTestMode() then
                -- Generate fake data:
                if not fakeActionStage then
                    fakeActionStage=0
                end
                retData.gripperAction=-1 -- means no action
                if sim.getSimulationTime()>5 and fakeActionStage==0 then
                    retData.gripperAction=1
                    fakeActionStage=1
                end
                if sim.getSimulationTime()>15 and fakeActionStage==1 then
                    retData.gripperAction=0
                    fakeActionStage=2
                end
                local currentPos,currentOrient=model.getPlatformPose()
                retData.platformPosAtAction=currentPos
                retData.platformYprAtAction=currentOrient
                retData.gripperActionTime=0
                retData.timeStamp=0
            else
                retData=nil
            end
        end
        if retData then
            if retData.actions then
                for i=1,#retData.actions,1 do
                    local actionData = retData.actions[i]
                    if actionData.gripperAction~=-1 then
                        -- Gripping state changed
                        local dat={}
                        dat[1]=actionData.gripperAction
                        dat[2]={actionData.platformPosAtAction[1],actionData.platformPosAtAction[2],actionData.platformPosAtAction[3]}
                        dat[3]={actionData.platformYprAtAction[1],actionData.platformYprAtAction[2],actionData.platformYprAtAction[3]}
                        dat[4]=actionData.timeStamp-actionData.gripperActionTime
                        dat[5]=model.platform
                        dat[6]=model.handles.ragnarRef
                        simBWF.callCustomizationScriptFunction('model.ext.attachOrDetachDetectedPart',model.gripper,dat)
                        if model.showTrajectory then
                            local m=sim.getObjectMatrix(model.handles.ragnarRef,-1)
                            local p=sim.multiplyVector(m,dat[2])
                            if actionData.gripperAction==1 then
                                sim.addDrawingObjectItem(model.graspCloseDrawingObject,p)
                            else
                                sim.addDrawingObjectItem(model.graspOpenDrawingObject,p)
                            end
                        end
                        
                        -- Buffer the open/close actions and their time/pos:
                        local co=nil
                        if actionData.gripperAction==1 then
                            co=gripperActionBuffer.close
        --                    setPlatformColor({1,0,1})
                        end
                        if actionData.gripperAction==0 then
                            co=gripperActionBuffer.open
        --                    setPlatformColor(platformOriginalCol)
                        end
                        if co then
                            co.t[#co.t+1]=actionData.gripperActionTime
                            co.v[#co.v+1]=actionData.platformPosAtAction[1]
                            
                            co.t[#co.t+1]=actionData.gripperActionTime
                            co.v[#co.v+1]=actionData.platformPosAtAction[2]
                            
                            co.t[#co.t+1]=actionData.gripperActionTime
                            co.v[#co.v+1]=actionData.platformPosAtAction[3]
                        end

                        -- Remove old gripper actions in the buffer:
                        if lastDataFromRagnar and lastDataFromRagnar.timeStamps and #lastDataFromRagnar.timeStamps>0 then
                            while #gripperActionBuffer.close.t>0 and gripperActionBuffer.close.t[1]<lastDataFromRagnar.timeStamps[1] do
                                table.remove(gripperActionBuffer.close.t,1)
                                table.remove(gripperActionBuffer.close.v,1)
                            end
                            while #gripperActionBuffer.open.t>0 and gripperActionBuffer.open.t[1]<lastDataFromRagnar.timeStamps[1] do
                                table.remove(gripperActionBuffer.open.t,1)
                                table.remove(gripperActionBuffer.open.v,1)
                            end
                        end
                    end
                end
            end

            
        end
    end
end

function model.getColorFromIntensity(intensity)
    local col={0.16,0.16,0.16,0.16,0.16,1,1,0.16,0.16,1,1,0.16}
    if intensity>1 then intensity=1 end
    if intensity<0 then intensity=0 end
    intensity=math.exp(4*(intensity-1))
    local d=math.floor(intensity*3)
    if (d>2) then d=2 end
    local r=(intensity-d/3)*3
    local coll={}
    coll[1]=col[3*d+1]*(1-r)+col[3*(d+1)+1]*r
    coll[2]=col[3*d+2]*(1-r)+col[3*(d+1)+2]*r
    coll[3]=col[3*d+3]*(1-r)+col[3*(d+1)+3]*r
    return coll
end

function sysCall_init()
    model.codeVersion=1
    
    model.platform=simBWF.callCustomizationScriptFunction('model.ext.getPlatform',model.handle)
    model.gripper=simBWF.callCustomizationScriptFunction('model.ext.getGripper',model.handle)
    
    model.online=simBWF.isSystemOnline()
    model.simOrRealIndex=1
    model.lastMoveVisualizeUpdateTimeInMs=-1000
    model.lastPlotVisualizeUpdateTimeInMs=-1000
    model.lastClearancePlotVisualizeUpdateTimeInMs=-1000
    if model.online then
        model.simOrRealIndex=2
        model.lastMoveVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)-1000
        model.lastPlotVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)-1000
        model.lastClearancePlotVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)-1000
    end
    local data=model.readInfo()
    model.connectionBufferSize=data['connectionBufferSize'][model.simOrRealIndex]
    model.primaryArmLengthInMM=data['primaryArmLengthInMM']
    model.secondaryArmLengthInMM=data['secondaryArmLengthInMM']
    model.trajectoryDrawingObject=sim.addDrawingObject(sim.drawing_lines+sim.drawing_itemcolors+sim.drawing_emissioncolor+sim.drawing_cyclic,3,0,-1,1000)
    model.graspCloseDrawingObject=sim.addDrawingObject(sim.drawing_spherepoints+sim.drawing_cyclic,0.005,0,-1,1,{1,0,1})
    model.graspOpenDrawingObject=sim.addDrawingObject(sim.drawing_spherepoints+sim.drawing_cyclic,0.005,0,-1,1,{0,1,1})
    model.lastClearanceData={}
    model.lastClearanceData.times={}
    model.lastClearanceData.clearances={}
    model.robotPlot.wasClosed=false
    model.clearancePlot.wasClosed=false
    model.previousClearanceFlag=data.clearance[model.simOrRealIndex]
    model.previousClearanceIncludePlatformFlag=data.clearanceWithPlatform[model.simOrRealIndex]
    model.previousClearanceEveryStepFlag=data.clearanceForAllSteps[model.simOrRealIndex]
    model.previousClearanceValue=0
    model.enableRagnar()
end


function sysCall_customCallback1()
    local data=model.readInfo()
    model.showTrajectory=data.showTrajectory[model.simOrRealIndex]
    maxSpeed=data.maxVel
    local delaysInMs={50,200,200}
    moveVisUpdateFrequMs=delaysInMs[data.visualizeUpdateFrequ[model.simOrRealIndex]+1]
    local delaysInMs={50,200,1000}
    plotVisUpdateFrequMs=delaysInMs[data.visualizeUpdateFrequ[model.simOrRealIndex]+1]
    if data.showPlot[model.simOrRealIndex] then
        if not model.robotPlot.wasClosed then
            model.robotPlot.startShowing()
        end
    else
        model.robotPlot.wasClosed=false
        model.robotPlot.stopShowing()
    end
    if data.clearance[model.simOrRealIndex] then
        if not model.clearancePlot.wasClosed then
            model.clearancePlot.startShowing()
        end
    else
        model.clearancePlot.wasClosed=false
        model.clearancePlot.stopShowing()
    end
    model.getAndApplyRagnarState()
end


function sysCall_sensing()
    if not model.online then
        local data=model.readInfo()
        local clearBuff=false
        if model.previousClearanceIncludePlatformFlag~=data.clearanceWithPlatform[model.simOrRealIndex] then
            clearBuff=true
        end
        if model.previousClearanceEveryStepFlag~=data.clearanceForAllSteps[model.simOrRealIndex] then
            clearBuff=true
        end
        model.previousClearanceFlag=data.clearance[model.simOrRealIndex]
        model.previousClearanceIncludePlatformFlag=data.clearanceWithPlatform[model.simOrRealIndex]
        model.previousClearanceEveryStepFlag=data.clearanceForAllSteps[model.simOrRealIndex]
        if clearBuff then
            model.lastClearanceData.times={}
            model.lastClearanceData.clearances={}
        end
        
        if data.clearance[model.simOrRealIndex] and not data.clearanceForAllSteps[model.simOrRealIndex] then
            local collection1=model.handles.robotArmCollection
            if data.clearanceWithPlatform[model.simOrRealIndex] then
                collection1=model.handles.robotArmAndPlatformCollection
            end
            local res,distData=sim.checkDistance(collection1,model.handles.robotObstaclesCollection,0)
            local clearanceValue=distData[7]
            if res>0 then
                model.lastClearanceData.times[#model.lastClearanceData.times+1]=sim.getSimulationTime()
                model.lastClearanceData.clearances[#model.lastClearanceData.clearances+1]=clearanceValue
                while #model.lastClearanceData.times>200 do
                    table.remove(model.lastClearanceData.times,1)
                    table.remove(model.lastClearanceData.clearances,1)
                end
                if data.clearanceWarning[model.simOrRealIndex]>0 then
                    if model.previousClearanceValue>data.clearanceWarning[model.simOrRealIndex] and clearanceValue<=data.clearanceWarning[model.simOrRealIndex] then
                        local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
                        simBWF.outputMessage("WARNING (run-time): Clearance threshold triggered"..nm,simBWF.MSG_WARN)
                    end
                end
                model.previousClearanceValue=clearanceValue
            end
        end
        local updatePlot=false
        local t=(sim.getSimulationTime()+sim.getSimulationTimeStep())*1000
        if t+1>model.lastClearancePlotVisualizeUpdateTimeInMs+plotVisUpdateFrequMs then
            updatePlot=true
            model.lastClearancePlotVisualizeUpdateTimeInMs=t
        end
        if updatePlot then
            model.clearancePlot.setData(model.lastClearanceData,1)
        end
    end
end
--[[
function setPlatformColor(col)
    if platformShape then
        for i=1,3,1 do
            if col[i]~=lastPlatformColor then
                sim.setShapeColor(platformShape,'RAGNARPLATFORM',sim.colorcomponent_ambient,col)
                lastPlatformColor=col
                break
            end
        end
    end
end
--]]

function sysCall_suspend()
    model.simulationPause(true)
end

function sysCall_resume()
    model.simulationPause(false)
end

function sysCall_cleanup()
--    setPlatformColor(platformOriginalCol)
    model.robotPlot.stopShowing()
    model.clearancePlot.stopShowing()
    model.disableRagnar()
end

