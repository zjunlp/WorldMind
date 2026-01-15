getDistributionValue=function(distribution)
    if (type(distribution[1])=='table') and (#distribution[1]==2) then
        local cnt=0
        for i=1,#distribution,1 do
           cnt=cnt+distribution[i][1] 
        end
        local p=sim.getFloatParameter(sim.floatparam_rand)*cnt
        cnt=0
        for i=1,#distribution,1 do
            if cnt+distribution[i][1]>=p then
                return distribution[i][2]
            end
            cnt=cnt+distribution[i][1] 
        end
    else
        local cnt=#distribution
        local p=1+math.floor(sim.getFloatParameter(sim.floatparam_rand)*cnt-0.0001)
        return distribution[p]
    end
end

function getAllParts()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.PART)
        if data then
            retL[#retL+1]=l[i]
        end
    end
    return retL
end

function isPartDetected(partHandle)
    --[[ We don't use a proximity sensor anymore, we simply check the position relative to the box
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
    --]]
    local p=sim.getObjectPosition(partHandle,model)
    return math.abs(p[1])<width*0.5 and math.abs(p[2])<length*0.5 and math.abs(p[3])<height*0.5
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
--    sensor=sim.getObjectHandle('genericPartTagger_sensor')
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTTAGGER)
    data=sim.unpackTable(data)
    if sim.boolAnd32(data['bitCoded'],8)>0 then
        console=sim.auxiliaryConsoleOpen('Tagged Parts',1000,4,nil,{600,300},nil,{0.9,0.9,1})
    end
    changeName=(sim.boolAnd32(data['bitCoded'],2)>0)
    changeDestination=(sim.boolAnd32(data['bitCoded'],4)>0)
    changeColor=(sim.boolAnd32(data['bitCoded'],16)>0)
    changeAnything=changeName or changeDestination or changeColor
    width=data['width']
    length=data['length']
    height=data['height']
    alreadyDetectedPartsList={}
    counter=0
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    if changeAnything then
        local line=''
        local p=getAllParts()
        local newNameDistribution
        local newDestinationNameDistribution
        local newColorDistribution
        if #p>0 then
            local data=sim.readCustomDataBlock(model,simBWF.modelTags.PARTTAGGER)
            data=sim.unpackTable(data)
            newNameDistribution='{'..data['partNameDistribution']..'}'
            local f=loadstring("return "..newNameDistribution)
            newNameDistribution=f()
            newDestinationNameDistribution='{'..data['partDestinationNameDistribution']..'}'
            f=loadstring("return "..newDestinationNameDistribution)
            newDestinationNameDistribution=f()
            newColorDistribution='{'..data['partColorDistribution']..'}'
            f=loadstring("return "..newColorDistribution)
            newColorDistribution=f()
        end
        for i=1,#p,1 do
            if isPartDetected(p[i]) then
                if not alreadyDetectedPartsList[p[i]] then
                    alreadyDetectedPartsList[p[i]]=t
                    local nameChanged=false
                    local destinationChanged=false
                    local colorChanged=false
                    local data=sim.readCustomDataBlock(p[i],simBWF.modelTags.PART)
                    data=sim.unpackTable(data)
                    local lline="Object name '"..sim.getObjectName(p[i]).."':\n"
                    lline=lline.."    --> from part with name '"..data['name'].."' and destination '"..data['destination'].."'\n" 
                    if changeName then
                        local newName=getDistributionValue(newNameDistribution)
                        if newName and newName~='<DEFAULT>' then
                            nameChanged=true
                            data['name']=newName
                        end
                    end
                    if changeDestination then
                        local newDestination=getDistributionValue(newDestinationNameDistribution)
                        if newDestination and newDestination~='<DEFAULT>' then
                            destinationChanged=true
                            data['destination']=newDestination
                        end
                    end
                    if changeColor then
                        local newColor=getDistributionValue(newColorDistribution)
                        if newColor and newColor~='<DEFAULT>' then
                            colorChanged=true
                            local l=sim.getObjectsInTree(p[i],sim.object_shape_type)
                            for i=1,#l,1 do
                                sim.setShapeColor(l[i],nil,sim.colorcomponent_ambient_diffuse,newColor)
                            end
                        end
                    end
                    lline=lline.."    --> to part with name '"..data['name'].."' and destination '"..data['destination'].."'\n"
                    if colorChanged then
                        lline=lline.."    --> color was changed\n"
                    end
                    lline=lline.."\n"
                    if nameChanged or destinationChanged or colorChanged then
                        line=line..lline
                        sim.writeCustomDataBlock(p[i],simBWF.modelTags.PART,sim.packTable(data))
                    end
                    counter=counter+1
                end
            else
                alreadyDetectedPartsList[p[i]]=nil
            end
        end
        if console and line~='' then
            sim.auxiliaryConsolePrint(console,line)
        end
    end
end
