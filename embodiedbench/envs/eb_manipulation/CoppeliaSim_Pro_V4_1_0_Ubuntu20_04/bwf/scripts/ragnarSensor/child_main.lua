function model.getAndApplySensorState()
    
    if model.online then
        local data={}
        data.id=model.handle
        local res,retData=simBWF.query('ragnarSensor_getTriggers',data)
        
        if res=='ok' then
            sensorTriggerTimesFromPlugin=retData
        else
            if simBWF.isInTestMode() then
                -- Generate fake data:
                if not blabla then
                    blabla=0
                    blabli=0
                end
                blabla=blabla+0.05
                sensorTriggerTimesFromPlugin={}
                if blabla>5 then
                        sensorTriggerTimesFromPlugin[#sensorTriggerTimesFromPlugin+1]=blabli+5
                        blabli=blabli+blabla
                        blabla=0
                end
            else
                sensorTriggerTimesFromPlugin={}
            end
        end
        model.sensorTriggerData.lastTime=simBWF.getSimulationOrOnlineTime()
    else
        model.sensorTriggerData.lastTime=sim.getSimulationTime()
    end
    if sensorTriggerTimesFromPlugin and #sensorTriggerTimesFromPlugin>0 then
        for i=1,#sensorTriggerTimesFromPlugin,1 do
            model.sensorTriggerData.triggerTimes[#model.sensorTriggerData.triggerTimes+1]=sensorTriggerTimesFromPlugin[i]
        end
    end
    if model.simSensorTriggers and #model.simSensorTriggers>0 then
        for i=1,#model.simSensorTriggers,1 do
            model.sensorTriggerData.triggerTimes[#model.sensorTriggerData.triggerTimes+1]=model.simSensorTriggers[i]
        end
        model.simSensorTriggers={}
    end
    -- Remove triggers that lay more then 10 seconds back:
    local ind=1
    while ind<=#model.sensorTriggerData.triggerTimes do
        if model.sensorTriggerData.triggerTimes[ind]<model.sensorTriggerData.lastTime-10 then
            table.remove(model.sensorTriggerData.triggerTimes,ind)
        else
            ind=ind+1
        end
    end
    if model.plot.ui then
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
        if model.plot.ui and updatePlot then
            model.plot.updateData(model.sensorTriggerData,1)
        end
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.online=simBWF.isSystemOnline()
    model.simOrRealIndex=1
    model.lastPlotVisualizeUpdateTimeInMs=-1000
    if model.online then
        model.simOrRealIndex=2
        model.lastPlotVisualizeUpdateTimeInMs=sim.getSystemTimeInMs(-1)-1000
    end
    model.sensorTriggerData={}
    model.sensorTriggerData.lastTime=0
    model.sensorTriggerData.triggerTimes={}
    model.plot.wasClosed=false
    model.lastSimSensorReading=false
    model.simSensorTriggers={}
end

function sysCall_suspend()
    if model.plot.ui then
        if model.lastDataFromSensor then
            model.plot.updateData(model.sensorTriggerData,1)
        end
        simUI.setMouseOptions(model.plot.ui,1,true,true,true,true)
    end
end

function sysCall_resume()
    if model.plot.ui then
        simUI.setMouseOptions(model.plot.ui,1,false,false,false,false)
    end
end

function sysCall_customCallback1()
    local data=model.readInfo()
    local delaysInMs={50,200,1000}
    plotVisUpdateFrequMs=delaysInMs[data.plotUpdateFrequ[model.simOrRealIndex]+1]
    
    if data.showPlot[model.simOrRealIndex] then
        if not model.plot.wasClosed then
            model.plot.showPlot()
        end
    else
        model.plot.wasClosed=false
        model.plot.closePlot()
    end
    model.getAndApplySensorState()
end


function sysCall_sensing()
    if not model.online then
        local res,d,pt,obj=sim.handleProximitySensor(model.handles.sensor)
        if res~=model.lastSimSensorReading then
            if res>0 then
                model.simSensorTriggers[#model.simSensorTriggers+1]=sim.getSimulationTime()
                local data={}
                data.id=model.handle
                data.dirOffset=pt[1]
                simBWF.query('ragnarSensor_trigger',data)
            end
            model.lastSimSensorReading=res
        end
    end
end


function sysCall_cleanup()
    model.plot.closePlot()
    if not model.online then
        sim.resetProximitySensor(model.handles.sensor)
    end
end

