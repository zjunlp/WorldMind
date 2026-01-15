model.realCamDisp={}

function model.realCamDisp.createDisplay()
    if model.imgToDisplay>0 then
        if not model.realCamDisp.ui then
            -- We need the resolution of the real camera:
            local data={}
            data.id=model.handle
            data.type='none'
            data.frequency=-2+1000/model.imgUpdateFrequMs
            local result,retData=simBWF.query('ragnarCamera_getRealImage',data)
            if result=='ok' then
                model.sensorResolution=retData.resolution
            else
                if simBWF.isInTestMode() then
                    if not _staticVarTesting then
                        _staticVarTesting=0
                    end
                    _staticVarTesting=_staticVarTesting+1
                    if _staticVarTesting==5 then
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
                local prevPos,prevSiz=simBWF.readSessionPersistentObjectData(model.handle,"visionImgDlgPosReal")
                if prevPos and prevSiz~=model.imgSizeToDisplay then
                    prevPos=nil
                    simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosReal",nil)
                end
                if not prevPos then
                    prevPos='bottomRight'
                end
                model.realCamDisp.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle),prevPos,true,'model.realCamDisp.removeDisplay'--[[,modal,resizable,activate,additionalUiAttribute--]])
                return true
            end
        end
    end
    return false
end

function model.realCamDisp.removeDisplay()
    if model.realCamDisp.ui then
        local x,y=simUI.getPosition(model.realCamDisp.ui)
        simBWF.writeSessionPersistentObjectData(model.handle,"visionImgDlgPosReal",{x,y},model.imgSizeToDisplay)
        simUI.destroy(model.realCamDisp.ui)
        model.realCamDisp.ui=nil
    end
end

function model.realCamDisp.updateDisplay()
    if model.realCamDisp.ui then -- realCamera_initialCameraTransform
        local c=model.readInfo()
        local data={}
        data.id=model.handle
        local tp={'none','color','depth'}
        data.type=tp[model.imgToDisplay+1]
        data.frequency=-2+1000/model.imgUpdateFrequMs
       
        local dt=sim.getSystemTimeInMs(model.lastImgUpdateTimeInMs)
        if (dt>model.imgUpdateFrequMs) and (model.realCamDisp.ui~=nil) and data.type~='none' then
            model.lastImgUpdateTimeInMs=sim.getSystemTimeInMs(-1)

            
            local result,retData=simBWF.query('ragnarCamera_getRealImage',data)
            local testing=simBWF.isInTestMode()
            local image=nil
            if not testing then
                if result=="ok" then
--                    if retData.ball1 then
--                        local m=simBWF.getMatrixFromCalibrationBallPositions(retData.ball1,retData.ball2,retData.ball3,true)
--                        sim.invertMatrix(m)
--                        sim.setObjectMatrix(model.handles.sensor,calibrationBalls[1],m)
--                    end
                    image=retData.image
--                    if image then
--                        sim.transformImage(image,model.sensorResolution,2) -- we receive the image flipped along the x axis. Correct that
--                    end
                end
            else
            --[[
                local ampl=0.01
                local off=-ampl*0.5
                local p1={-0.2-off+math.random()*ampl,-0.075-off+math.random()*ampl,0.4553-off+math.random()*ampl}
                local p2={-0.2-off+math.random()*ampl,0.1-off+math.random()*ampl,0.455-off+math.random()*ampl}
                local p3={0.16-off+math.random()*ampl,0.09-off+math.random()*ampl,0.455-off+math.random()*ampl}
                local m=simBWF.getMatrixFromCalibrationBallPositions(p1,p2,p3,true)
                sim.invertMatrix(m)
                sim.setObjectMatrix(model.handles.sensor,calibrationBalls[1],m)
                --]]
            end
            if not image then
                if not sstaticBla then sstaticBla=0 end
                sstaticBla=sstaticBla+60
                if sstaticBla>255 then sstaticBla=sstaticBla-255 end
                local ttmp=sim.packUInt8Table({sstaticBla,0,255-sstaticBla})
                image=string.rep(ttmp,model.sensorResolution[1]*model.sensorResolution[2])
            end
            local resDividers={4,2,1}
            local div=resDividers[model.imgSizeToDisplay+1]
            local w=model.sensorResolution[1]
            local h=model.sensorResolution[2]
            if div~=1 then
                image=sim.getScaledImage(image,{w,h},{w/div,h/div},0)
            end
            simUI.setImageData(model.realCamDisp.ui,1,image,w/div,h/div)
        end
    end
end
