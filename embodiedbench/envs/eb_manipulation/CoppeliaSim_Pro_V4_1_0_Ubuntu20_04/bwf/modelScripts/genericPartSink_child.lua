destroyPartIfPart=function(objH)
    if objH and objH>=0 then
        local isPart,isInstanciated=simBWF.isObjectPartAndInstanciated(objH)
        if isPart then
            if isInstanciated then
                local p=sim.getModelProperty(objH)
                if sim.boolAnd32(p,sim.modelproperty_not_model)>0 then
                    sim.removeObject(objH)
                else
                    sim.removeModel(objH)
                end
                return true
            else
                return false
            end
        else
            while objH>=0 do
                objH=sim.getObjectParent(objH)
                if objH>=0 then
                    isPart,isInstanciated=simBWF.isObjectPartAndInstanciated(objH)
                    if isPart then
                        if isInstanciated then
                            sim.removeModel(objH)
                            return true
                        else
                            return false
                        end
                    end
                end
            end
        end
    end
    return false
end

prepareStatisticsDialog=function(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="Part destruction count: 0" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

updateStatisticsDialog=function(enabled)
    if statUi then
        simUI.setLabelText(statUi,1,"Part destruction count: "..destructionCount,true)
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    sensor=sim.getObjectHandle('genericPartSink_sensor')
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTSINK)
    data=sim.unpackTable(data)
    operational=data['status']~='disabled'
    destructionCount=0
    prepareStatisticsDialog(sim.boolAnd32(data['bitCoded'],128)>0)
end


function sysCall_actuation()
    if operational then
        local shapes=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type)
        for i=1,#shapes,1 do
            if sim.isHandleValid(shapes[i])>0 then
                if sim.boolAnd32(sim.getObjectSpecialProperty(shapes[i]),sim.objectspecialproperty_detectable_all)>0 then
                    local r=sim.checkProximitySensor(sensor,shapes[i])
                    if r>0 then
                        if destroyPartIfPart(shapes[i]) then
                            destructionCount=destructionCount+1
                            local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTSINK)
                            data=sim.unpackTable(data)
                            data['destroyedCnt']=data['destroyedCnt']+1
                            sim.writeCustomDataBlock(model,simBWF.modelTags.PARTSINK,sim.packTable(data))
                        end
                    end
                end
            end
        end
    end
    updateStatisticsDialog()
end
