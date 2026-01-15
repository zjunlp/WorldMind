function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDLOCATION)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDLOCATION,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDLOCATION,'')
    end
end

function getAvailableBuckets()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_BUCKET_INFO')
        if data then
            retL[#retL+1]=l[i]
        end
    end
    return retL
end

function getAvailableTransporters()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_TRANSPORTER_INFO')
        if data then
            retL[#retL+1]=l[i]
        end
    end
    return retL
end

function isABucketWithinRange()
    for i=1,#buckets,1 do
        local p=sim.getObjectPosition(buckets[i],model)
        local d=math.sqrt(p[1]*p[1]+p[2]*p[2])
        if d<0.1 then
            return true
        end
    end
    return false
end

function isATransporterWithinRange()
    for i=1,#transporters,1 do
        local p=sim.getObjectPosition(transporters[i],model)
        local d=math.sqrt(p[1]*p[1]+p[2]*p[2])
        if d<0.1 then
            return true
        end
    end
    return false
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    buckets=getAvailableBuckets()
    transporters=getAvailableTransporters()
end

function sysCall_sensing()
    local data=readInfo()
    if isABucketWithinRange() or isATransporterWithinRange() then
        data['status']='occupied'
    else
        if data['status']~='reserved' then
            data['status']='free'
        end
    end
    writeInfo(data)
end


function sysCall_cleanup()

	-- Put some restoration code here

end