model.simCamDisp={}

function model.simCamDisp.createDisplay()
    if model.imgToDisplay>0 then
        if not model.simCamDisp.ui then
            local resDividers={4,2,1}
            local div=resDividers[model.imgSizeToDisplay+1]
            local w=model.sensorResolution[1]/div
            local h=model.sensorResolution[2]/div
            local xml='<image id="1" width="'..w..'" height="'..h..'"/>'
            local prevPos,prevSiz=simBWF.readSessionPersistentObjectData(model.handle,"visionImgDlgPosSim")
            if prevPos and prevSiz~=model.imgSizeToDisplay then
                prevPos=nil
                simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosSim",nil)
            end
            if not prevPos then
                prevPos='bottomRight'
            end
            model.simCamDisp.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle),prevPos,true,'model.simCamDisp.removeDisplay'--[[,modal,resizable,activate,additionalUiAttribute--]])
            return true
        end
    end
    return false
end

function model.simCamDisp.removeDisplay()
    if model.simCamDisp.ui then
        local x,y=simUI.getPosition(model.simCamDisp.ui)
        simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosSim",{x,y},model.imgSizeToDisplay)
        simUI.destroy(model.simCamDisp.ui)
        model.simCamDisp.ui=nil
    end
end

function model.simCamDisp.updateDisplay()
    local res,nclipp=sim.getObjectFloatParameter(model.handles.sensor,sim.visionfloatparam_near_clipping)
    local res,fclipp=sim.getObjectFloatParameter(model.handles.sensor,sim.visionfloatparam_far_clipping)
    local rgbRaw=sim.getVisionSensorCharImage(model.handles.sensor)
    local depthRaw=sim.getVisionSensorDepthBuffer(model.handles.sensor+sim.handleflag_codedstring)
    local depth=sim.transformBuffer(depthRaw,sim.buffer_float,1000*(fclipp-nclipp),1000*nclipp,sim.buffer_uint16)
    local data={}
    data.id=model.handle
    data.resolution=model.sensorResolution
    data.depth=depth
    data.color=rgbRaw
    simBWF.query('ragnarCamera_setSimulatedImage',data)

    local t=(sim.getSimulationTime()+sim.getSimulationTimeStep())*1000
    if (t+1>model.lastImgUpdateTimeInMs+model.imgUpdateFrequMs) and model.simCamDisp.ui then
        model.lastImgUpdateTimeInMs=t
        local resDividers={4,2,1}
        local div=resDividers[model.imgSizeToDisplay+1]
        local image=nil
        if model.imgToDisplay==1 then -- rgb
            image=rgbRaw
        end
        if model.imgToDisplay==2 then -- depth
            image=sim.transformBuffer(depthRaw,sim.buffer_float,255,0,sim.buffer_uint8rgb)
        end
        if div~=1 then -- Scaling
            image=sim.getScaledImage(image,{model.sensorResolution[1],model.sensorResolution[2]},{model.sensorResolution[1]/div,model.sensorResolution[2]/div},0)
        end
        simUI.setImageData(model.simCamDisp.ui,1,image,model.sensorResolution[1]/div,model.sensorResolution[2]/div)
    end
end
