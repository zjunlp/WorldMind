function sim.getObjectHandle_noErrorNoSuffixAdjustment(name)
    local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local suff=sim.getNameSuffix(nil)
    sim.setNameSuffix(-1)
    retVal=sim.getObjectHandle(name)
    sim.setNameSuffix(suff)
    sim.setInt32Parameter(sim.intparam_error_report_mode,err)
    return retVal
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
    local p=sim.getObjectPosition(partHandle,model)
    return math.abs(p[1])<xSize*0.5 and math.abs(p[2])<ySize*0.5 and math.abs(p[3])<zSize*0.5
end

isEnabled=function()
    local data=sim.readCustomDataBlock(model,'XYZ_PARTTELEPORTER_INFO')
    data=sim.unpackTable(data)
    return sim.boolAnd32(data['bitCoded'],1)>0
end

getSensorPart=function()
    local p=getAllParts()
    for i=1,#p,1 do
        if isPartDetected(p[i]) then
            return p[i]
        end
    end
    return -1
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,'XYZ_PARTTELEPORTER_INFO')
    data=sim.unpackTable(data)
    isSource=sim.boolAnd32(data['bitCoded'],2)>0
    xSize=data['width']
    ySize=data['length']
    zSize=data['height']
    destinationPod=simBWF.getReferencedObjectHandle(model,simBWF.TELEPORTER_DESTINATION_REF)
    if destinationPod>=0 then
        local dataD=sim.readCustomDataBlock(destinationPod,'XYZ_PARTTELEPORTER_INFO')
        if dataD then
            dataD=sim.unpackTable(dataD)
            if sim.boolAnd32(dataD['bitCoded'],2)==0 then
                xSizeD=dataD['width']
                ySizeD=dataD['length']
                zSizeD=dataD['height']
            else
                destinationPod=-1
            end
        else
            destinationPod=-1
        end
    end
end

function sysCall_actuation()
    if isSource and destinationPod>=0 and isEnabled() then
        local part=getSensorPart()
        if part>=0 then
            local m=sim.getObjectMatrix(part,model)
            m[4]=xSizeD*m[4]/xSize
            m[8]=ySizeD*m[8]/ySize
            m[12]=zSizeD*m[12]/zSize
            sim.setObjectMatrix(part,destinationPod,m)
            local p=sim.getModelProperty(part)
            if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
                -- We have a model
                local l=sim.getObjectsInTree(part)
                for i=1,#l,1 do
                    sim.resetDynamicObject(l[i])
                end
            else
                -- We have a shape
                sim.resetDynamicObject(part)
            end
        end
    end
end
