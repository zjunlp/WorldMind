function model.enableSimulatedCamera()
--[[        local data={}
        data.id=model.handle
        data.imageProcessingParameters=imgProcessingParams
        simBWF.query('ragnarVision_connectSimulated',data)--]]
end

function model.disableSimulatedCamera()
--[[        local data={}
        data.id=model.handle
        simBWF.query('ragnarVision_disconnectSimulated',data)--]]
end

function model.enableRealCamera()
--[[
    if not realCamera_initialCameraTransform then
        realCamera_initialCameraTransform=sim.getObjectMatrix(model.handles.sensor,model.handle)
        local data={}
        data.id=model.handle
        data.imageProcessingParameters=imgProcessingParams
        simBWF.query('ragnarVision_connectReal',data)
    end
    --]]
end

function model.disableRealCamera()
--[[
    if realCamera_initialCameraTransform then
        local data={}
        data.id=model.handle
        simBWF.query('ragnarVision_disconnectReal',data)
        sim.setObjectMatrix(model.handles.sensor,model.handle,realCamera_initialCameraTransform)
        realCamera_initialCameraTransform=nil
    end
    --]]
end

function sysCall_init()
    model.codeVersion=1
    
    model.sensorInitialMatrix=sim.getObjectMatrix(model.handles.sensor,-1)
    model.allDesiredModelPoses={}

    model.online=simBWF.isSystemOnline()
    model.simOrRealIndex=1
    model.lastImgUpdateTimeInMs=-1000
    if model.online then
        model.simOrRealIndex=2
        model.lastImgUpdateTimeInMs=sim.getSystemTimeInMs(-1)-1000
    end
    local data=model.readInfo()
    model.sensorResolution=data.resolution
    
    model.imgToDisplay=data.imgToDisplay[model.simOrRealIndex]
    model.imgSizeToDisplay=data.imgSizeToDisplay[model.simOrRealIndex]
    
    model.lastImgToDisplay=-1
    model.lastImgSizeToDisplay=model.imgSizeToDisplay-- -1
    
    if model.online then
        model.enableRealCamera()
    else
        model.enableSimulatedCamera()
    end
end

function sysCall_sensing()
    local data=model.readInfo()
    model.imgToDisplay=data.imgToDisplay[model.simOrRealIndex]
    model.imgSizeToDisplay=data.imgSizeToDisplay[model.simOrRealIndex]
    local delaysInMs={50,200,1000}
    model.imgUpdateFrequMs=delaysInMs[data.imgUpdateFrequ[model.simOrRealIndex]+1]

    if model.online then
        if model.lastImgToDisplay~=model.imgToDisplay or model.lastImgSizeToDisplay~=model.imgSizeToDisplay then
            model.realCamDisp.removeDisplay()
            if model.lastImgSizeToDisplay~=model.imgSizeToDisplay then
                simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosReal",nil)
            end
            if model.realCamDisp.createDisplay() then
                model.lastImgToDisplay=model.imgToDisplay
                model.lastImgSizeToDisplay=model.imgSizeToDisplay
            end
        end
        model.realCamDisp.updateDisplay()
    else
        sim.handleVisionSensor(model.handles.sensor)
        if model.lastImgToDisplay~=model.imgToDisplay or model.lastImgSizeToDisplay~=model.imgSizeToDisplay then
            model.simCamDisp.removeDisplay()
            if model.lastImgSizeToDisplay~=model.imgSizeToDisplay then
                simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosSim",nil)
            end
            if model.simCamDisp.createDisplay() then
                model.lastImgToDisplay=model.imgToDisplay
                model.lastImgSizeToDisplay=model.imgSizeToDisplay
            end
        end
        model.simCamDisp.updateDisplay()
    end
end


function sysCall_cleanup()
    if model.online then
        model.realCamDisp.removeDisplay()
        model.disableRealCamera()
    else
        model.simCamDisp.removeDisplay()
        model.disableSimulatedCamera()
    end
    sim.setObjectMatrix(model.handles.sensor,-1,model.sensorInitialMatrix)
end

