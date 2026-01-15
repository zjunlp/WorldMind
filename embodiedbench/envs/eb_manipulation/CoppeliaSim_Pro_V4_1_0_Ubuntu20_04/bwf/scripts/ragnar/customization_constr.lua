function model.applyCalibrationData()
    -- We basically need to adjust the position and orientation of Ragnar, if it is connected to
    -- at least one calibrated tracking window:

    local info=model.readInfo()
    local windAndCal={}
    for i=1,C.CIC,1 do
        local id=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)
        if id>=0 then
            local m=simBWF.callCustomizationScriptFunction('model.ext.getCalibrationMatrix',id)
            if m~=nil then
                windAndCal[#windAndCal+1]={id,m}
            end
        end
        local id=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)
        if id>=0 then
            local m=simBWF.callCustomizationScriptFunction('model.ext.getCalibrationMatrix',id)
            if m~=nil then
                windAndCal[#windAndCal+1]={id,m}
            end
        end
    end
    if #windAndCal>0 then
        for i=1,#windAndCal,1 do
            local winId=windAndCal[i][1]
            local m=windAndCal[i][2]
            -- Here we keep the tracking window in place (i.e. its red calibration ball), and adjust the position/orientation of the robot instead:
            sim.invertMatrix(m)
            local mWindow=sim.getObjectMatrix(winId,-1)
            local newAbsRefM=sim.multiplyMatrices(mWindow,m)
            windAndCal[i][3]=newAbsRefM -- this is the desired abs transf. of the Ragnar ref. (for this tracking window)
        end

        local m
        for i=1,#windAndCal,1 do
            local winId=windAndCal[i][1]
            local matr=windAndCal[i][3]
            if i==1 then
                m=matr
            else
--                m=matr
                m=sim.interpolateMatrices(m,matr,1/i)
            end
        end

        -- If we just want to slightly adjust the X/Y position of the robot (no orientation, nor Z change):
        local allowFullAdjustment=true
        local allowZAdjustment=true
        if allowFullAdjustment then
            local locRefMInv=sim.getObjectMatrix(model.handles.ragnarRef,model.handle)
            sim.invertMatrix(locRefMInv)
            local toApplyM=sim.multiplyMatrices(m,locRefMInv)
            sim.setObjectMatrix(model.handle,-1,toApplyM)
        else
            local absRefV=sim.getObjectPosition(model.handles.ragnarRef,-1)
            local p=sim.getObjectPosition(model.handle,-1)
            local nAbsRefV={m[4],m[8],absRefV[3]}
            if allowZAdjustment then
                nAbsRefV[3]=m[12]
                p[3]=p[3]+nAbsRefV[3]-absRefV[3]
            end
            p[1]=p[1]+nAbsRefV[1]-absRefV[1]
            p[2]=p[2]+nAbsRefV[2]-absRefV[2]
            sim.setObjectPosition(model.handle,-1,p)
        end

--        local r,p=sim.getObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions)
--        r=sim.boolOr32(r,1+4+16+32)-(1+4) -- forbid rotation and translation when simulation is not running
--        sim.setObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions,r)
--    else
--        local r,p=sim.getObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions)
--        r=sim.boolOr32(r,1+4+16+32) -- allow rotation and translation when simulation is not running
--        sim.setObjectInt32Parameter(model.handle,sim.objintparam_manipulation_permissions,r)
    end
end

function model.setAttachedLocationFramesIntoCalibrationPose()
    for i=1,C.CIC,1 do
        local frameHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)
        if frameHandle>=0 then
            simBWF.callCustomizationScriptFunction('model.ext.setLocationFrameIntoOnlineCalibrationPose',frameHandle)
        end
        local frameHandle=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)
        if frameHandle>=0 then
            simBWF.callCustomizationScriptFunction('model.ext.setLocationFrameIntoOnlineCalibrationPose',frameHandle)
        end
    end
end
