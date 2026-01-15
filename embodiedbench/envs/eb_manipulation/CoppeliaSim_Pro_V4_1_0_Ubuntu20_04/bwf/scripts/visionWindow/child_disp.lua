-- Following currently not used anymore, but keep in case people change their mind again:

--[=[
model.disp={}

function model.disp.createDisplay()
    if model.camera>=0 then
        if model.imgToDisplay>0 then
            if not model.disp.ui then
                -- We need the resolution of the real camera:
                local data={}
                data.id=model.handle
                data.type='none'
                data.frequency=-2+1000/model.imgUpdateFrequMs
                local result,retData=simBWF.query('ragnarVision_getImage',data)
                if result=='ok' then
                    model.sensorResolution=retData.resolution
                else
                    if simBWF.isInTestMode() then
                        if not model._staticVarTesting then
                            model._staticVarTesting=0
                        end
                        model._staticVarTesting=model._staticVarTesting+1
                        if model._staticVarTesting==5 then
                            model.sensorResolution={640,480}
                            result='ok'
                        end
                    end
                end
                if result=='ok' then
                    local resDividers={4,2,1}
                    local div=resDividers[model.imgSizeToDisplay+1]
                    local w=model.sensorResolution[1]/div
                    local h=model.sensorResolution[2]/div
                    local xml='<image id="1" width="'..w..'" height="'..h..'"/>'
                    local prevPos,prevSiz=simBWF.readSessionPersistentObjectData(model.handle,"visionImgDlgPos")
                    if prevPos and prevSiz~=model.imgSizeToDisplay then
                        prevPos=nil
                        simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPos",nil)
                    end
                    if not prevPos then
                        prevPos='bottomRight'
                    end
                    model.disp.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle),prevPos,true,'model.disp.removeDisplay'--[[,modal,resizable,activate,additionalUiAttribute--]])
                    return true
                end
            end
        end
    end
    return false
end

function model.disp.removeDisplay()
    if model.disp.ui then
        local x,y=simUI.getPosition(model.disp.ui)
        simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPos",{x,y},model.imgSizeToDisplay)
        simUI.destroy(model.disp.ui)
        model.disp.ui=nil
    end
end

function model.disp.updateDisplay()
    if model.camera>=0 and model.disp.ui then
--        local c=sim.unpackTable(sim.readCustomDataBlock(model.handle,simBWF.modelTags.VISIONWINDOW))
        local data={}
        data.id=model.handle
        local tp={'none','processed'}
       
        local dt=sim.getSystemTimeInMs(model.lastImgUpdateTimeInMs)
        if (dt>model.imgUpdateFrequMs) and (model.disp.ui~=nil) then
            data.type=tp[model.imgToDisplay+1]
            model.lastImgUpdateTimeInMs=sim.getSystemTimeInMs(-1)
        else
            data.type=tp[1] -- i.e. 'none'
        end
        if model.imgToDisplay==0 then
            data.frequency=0
        else
            data.frequency=-2+1000/model.imgUpdateFrequMs
        end
        local result,retData=simBWF.query('ragnarVision_getImage',data)
        local testing=simBWF.isInTestMode()
        local image=nil
        if not testing then
            if result=="ok" then
                if retData.ball1 then
                    local m=simBWF.getMatrixFromCalibrationBallPositions(retData.ball1,retData.ball2,retData.ball3,true)
                    sim.invertMatrix(m)
                    m=sim.multiplyMatrices(sim.getObjectMatrix(model.handle,-1),m)
                    simBWF.callChildScriptFunction_noError("model.ext.setCameraPoseFromCalibrationBallDetections",model.camera,model.handle,m)
                else
                    simBWF.callChildScriptFunction_noError("model.ext.setCameraPoseFromCalibrationBallDetections",model.camera,model.handle,nil)
                end
                image=retData.image
--                if image then
--                    sim.transformImage(image,model.sensorResolution,2) -- we receive the image flipped along the x axis. Correct that
--                end
            end
        else
            local ampl=0.01
            local off=-ampl*0.5
            local p1={-0.2-off+math.random()*ampl,-0.075-off+math.random()*ampl,0.4553-off+math.random()*ampl}
            local p2={-0.2-off+math.random()*ampl,0.1-off+math.random()*ampl,0.455-off+math.random()*ampl}
            local p3={0.16-off+math.random()*ampl,0.09-off+math.random()*ampl,0.455-off+math.random()*ampl}
            local m=simBWF.getMatrixFromCalibrationBallPositions(p1,p2,p3,true)
            sim.invertMatrix(m)
            m=sim.multiplyMatrices(sim.getObjectMatrix(model.handle,-1),m)
            simBWF.callChildScriptFunction_noError("model.ext.setCameraPoseFromCalibrationBallDetections",model.camera,model.handle,m)
        end
        if model.disp.ui then
            if data.type~='none' then
                if not image then
                    if not model.sstaticBla then model.sstaticBla=0 end
                    model.sstaticBla=model.sstaticBla+60
                    if model.sstaticBla>255 then model.sstaticBla=model.sstaticBla-255 end
                    local ttmp=sim.packUInt8Table({model.sstaticBla,0,255-model.sstaticBla})
                    image=string.rep(ttmp,model.sensorResolution[1]*model.sensorResolution[2])
                end
                local resDividers={4,2,1}
                local div=resDividers[model.imgSizeToDisplay+1]
                local w=model.sensorResolution[1]
                local h=model.sensorResolution[2]
                if div~=1 then
                    image=sim.getScaledImage(image,{w,h},{w/div,h/div},0)
                end
                simUI.setImageData(model.disp.ui,1,image,w/div,h/div)
            end
        end
    end
end
--]=]
