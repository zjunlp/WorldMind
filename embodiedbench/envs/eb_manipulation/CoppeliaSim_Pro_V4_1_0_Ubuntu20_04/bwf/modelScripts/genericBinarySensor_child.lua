prepareStatisticsDialog=function(enabled)
    if enabled then
        local xml = [[
                <label id="1" text="BLA" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

updateStatisticsDialog=function()
    if statUi then
        simUI.setLabelText(statUi,1,statText..simBWF.format("%.0f",incrementer),true)
    end
end

function getAllParts()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    for i=1,#l,1 do
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
        if isInstanciated then
            retL[#retL+1]=l[i]
        end
    end
    return retL
end

function isPartDetected(partHandle)
    local shapesToTest={}
    if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_model)>0 then
        -- We have a single shape which is not a model. Is the shape detectable?
        if sim.boolAnd32(sim.getObjectSpecialProperty(partHandle),sim.objectspecialproperty_detectable_all)>0 then
            shapesToTest[1]=partHandle -- yes, it is detectable
        end
    else
        -- We have a model. Does the model have the detectable flags overridden?
        if sim.boolAnd32(sim.getModelProperty(partHandle),sim.modelproperty_not_detectable)==0 then
            -- No, now take all model shapes that are detectable:
            local t=sim.getObjectsInTree(partHandle,sim.object_shape_type,0)
            for i=1,#t,1 do
                if sim.boolAnd32(sim.getObjectSpecialProperty(t[i]),sim.objectspecialproperty_detectable_all)>0 then
                    shapesToTest[#shapesToTest+1]=t[i]
                end
            end
        end
    end
    for i=1,#shapesToTest,1 do
        if sim.checkProximitySensor(sensor,shapesToTest[i])>0 then
            return true
        end
    end
    return false
end

isEnabled=function()
    local data=sim.readCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO')
    data=sim.unpackTable(data)
    return sim.boolAnd32(data['bitCoded'],1)>0
end

setDetectionState=function(v)
    local data=sim.readCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO')
    data=sim.unpackTable(data)
    data['detectionState']=v
    sim.writeCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO',sim.packTable(data))
end

checkSensor=function()
    if detectPartsOnly then
        local p=getAllParts()
        for i=1,#p,1 do
            if isPartDetected(p[i]) then
                return true
            end
        end
    else
        local result=sim.handleProximitySensor(sensor)
        return result>0
    end
    return false
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    sensor=sim.getObjectHandle('genericBinarySensor_sensor')
    local data=sim.readCustomDataBlock(model,'XYZ_BINARYSENSOR_INFO')
    data=sim.unpackTable(data)
    detectPartsOnly=sim.boolAnd32(data['bitCoded'],2)>0
    onRise=sim.boolAnd32(data['bitCoded'],8)>0
    onFall=sim.boolAnd32(data['bitCoded'],16)>0
    countForTrigger=data['countForTrigger']
    showStats=sim.boolAnd32(data['bitCoded'],32)>0
    statText=data['statText']
    delayForTrigger=data['delayForTrigger']
    prepareStatisticsDialog(showStats)
    previousDetectionState=false
    detectionCount=0
    incrementTimes={}
    incrementer=0
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    local dt=sim.getSimulationTimeStep()
    if isEnabled() then
        local r=checkSensor()
        if r~=previousDetectionState then
            if r then
                if onRise then
                   detectionCount=detectionCount+1
                end
            else
                if onFall then
                   detectionCount=detectionCount+1
                end
            end
            if detectionCount>=countForTrigger then
                detectionCount=0
                incrementTimes[#incrementTimes+1]=t+delayForTrigger
            end
        end
        previousDetectionState=r
    end

    local changeCol=false
    local i=1
    while i<=#incrementTimes do
        if incrementTimes[i]<t+dt then
            incrementer=incrementer+1
            setDetectionState(incrementer)
            changeCol=true
            table.remove(incrementTimes,i)
        else
            i=i+1
        end
    end

    if changeCol then
        sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{1,0,0})
    else
        sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{0,0,1})
    end
    updateStatisticsDialog()
end

function sysCall_cleanup()
        sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{0,0,1})
end