model.ext={}

function model.ext.setCameraPoseFromCalibrationBallDetections(ragnarVisionHandle,absMatrix)
    if model.online then
        model.allDesiredModelPoses[ragnarVisionHandle]=absMatrix
        local m=nil
        local cnt=1
        for key,value in pairs(model.allDesiredModelPoses) do
            if m==nil then
                m=value
            else
                m=sim.interpolateMatrices(m,value,1/cnt)
            end
            cnt=cnt+1
        end
        if m then
            sim.setObjectMatrix(model.handles.sensor,-1,m)
        else
            sim.setObjectMatrix(model.handles.sensor,-1,model.sensorInitialMatrix)
        end
    end
end
