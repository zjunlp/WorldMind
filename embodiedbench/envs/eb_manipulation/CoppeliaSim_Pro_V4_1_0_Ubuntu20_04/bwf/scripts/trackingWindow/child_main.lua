function model.getAssociatedRobotHandle()
    local ragnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
    for i=1,#ragnars,1 do
        if simBWF.callCustomizationScriptFunction('model.ext.checkIfRobotIsAssociatedWithLocationFrameOrTrackingWindow',ragnars[i],model.handle) then
            return ragnars[i]
        end
    end
    return -1
end

function sysCall_init()
    model.codeVersion=1
    
    local data=model.readInfo()
    model.showPoints=sim.boolAnd32(data.bitCoded,4)>0
    model.isPick=(data.type==0)
    model.createParts=model.isPick and (sim.boolAnd32(data.bitCoded,8)>0)
    model.robot=model.getAssociatedRobotHandle()
    if model.robot>=0 then
        model.robotRef=simBWF.callCustomizationScriptFunction('model.ext.getReferenceObject',model.robot)
        model.robotRefM=sim.getObjectMatrix(model.robotRef,-1)
        local mRelRef=sim.getObjectMatrix(model.handle,model.robotRef)
        local dat=simBWF.callCustomizationScriptFunction('model.ext.getCalibrationDataForCurrentMode',model.handle)
        model.calibrationMDat=dat.matrix -- data.calibrationMatrix
        model.calibrationM=model.calibrationMDat
        if not model.calibrationMDat then
            model.calibrationM=mRelRef
        end
    end
    model.m=sim.getObjectMatrix(model.handle,-1)
    model.mi=sim.getObjectMatrix(model.handle,-1)
    sim.invertMatrix(model.mi)
    model.sphereContainer=sim.addDrawingObject(sim.drawing_spherepoints,0.007,0,-1,9999,{1,0,1})
    model.online=simBWF.isSystemOnline()
    model.createdPartsInOnlineMode={}
    model.allProducedPartsInOnlineMode={}
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()
    sim.addDrawingObjectItem(model.sphereContainer,nil)
   
    if model.robot>0 then
        model.robotRefM=sim.getObjectMatrix(model.robotRef,-1)
        local mRelRef=sim.getObjectMatrix(model.handle,model.robotRef)
        if not model.calibrationMDat then
            model.calibrationM=mRelRef
        end
        local data={}
        data.id=model.handle
        local reply,retData=simBWF.query('trackingWindow_getPoints',data)
        if simBWF.isInTestMode() then
            reply='ok'
            retData={}
            retData.points={{0,0,0.3}}
            retData.pointIds={1}
            retData.partIds={sim.getObjectHandle('genericBox#')}
        end
        if reply=='ok' then
            local pts=retData.points
            local ptIds=retData.pointIds
            if model.online then
                local partIds=retData.partIds
                for i=1,#pts,1 do
                    local ptRel=pts[i]
                    local ptMRel=sim.buildMatrix(ptRel,{0,0,0})
                    -- We transform the position of the point in order to correct for calibration errors:
                    local dist=-ptMRel[4]
                    local tr=1-(dist-0.3)/0.3
                    if tr>1 then tr=1 end
                    if tr<0 then tr=0 end
                    if model.showPoints then
                        local mFar=sim.multiplyMatrices(model.m,ptMRel)
                        local mClose=sim.multiplyMatrices(model.calibrationM,ptMRel)
                        local mClose=sim.multiplyMatrices(model.robotRefM,mClose)
                        local theMatr=sim.interpolateMatrices(mFar,mClose,tr)
                        local dat={theMatr[4],theMatr[8],theMatr[12]}
                        sim.addDrawingObjectItem(model.sphereContainer,dat)
                    end
 --                   print(ptIds[i],partIds[i],sim.getObjectName(partIds[i])
 --                   -- We create parts that were detected in the real world (but only when the position is correct in simulation):
--                    print(model.createParts,partIds,partIds[i]>=0,tr)
                    if model.createParts and partIds and partIds[i]>=0 and tr==1 and model.createdPartsInOnlineMode[ptIds[i]]==nil then
                        local partData=simBWF.readPartInfo(partIds[i])
                        local vertMinMax=partData.vertMinMax
                        ptRel[1]=ptRel[1]-0.5*vertMinMax[1][2]-0.5*vertMinMax[1][1]
                        ptRel[2]=ptRel[2]-0.5*vertMinMax[2][2]-0.5*vertMinMax[2][1]
                        ptRel[3]=ptRel[3]-vertMinMax[3][2]
                        local itemPosition=sim.multiplyVector(model.m,ptRel)
                        local itemOrientation=sim.getEulerAnglesFromMatrix(model.m)
                        simBWF.instanciatePart(partIds[i],itemPosition,itemOrientation,nil,nil,nil,false)
                        model.createdPartsInOnlineMode[ptIds[i]]=true
                    end
                end
            else
                 if model.showPoints then
                     for i=1,#pts,1 do
                        local ptRel=pts[i]
                        local ptMRel=sim.buildMatrix(ptRel,{0,0,0})
                        -- We transform the position of the point in order to correct for calibration errors:
                        local dist=-ptMRel[4]
                        local mFar=sim.multiplyMatrices(model.m,ptMRel)
                        local mClose=sim.multiplyMatrices(model.calibrationM,ptMRel)
                        local mClose=sim.multiplyMatrices(model.robotRefM,mClose)
                        local tr=1-(dist-0.3)/0.3
                        if tr>1 then tr=1 end
                        if tr<0 then tr=0 end
                        local theMatr=sim.interpolateMatrices(mFar,mClose,tr)
                        local dat={theMatr[4],theMatr[8],theMatr[12]}
                        sim.addDrawingObjectItem(model.sphereContainer,dat)
                    end
                end
            end
        end
    end
end

