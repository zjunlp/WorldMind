function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    version=sim.getInt32Parameter(sim.intparam_program_version)
    local data=sim.unpackTable(sim.readCustomDataBlock(model,simBWF.modelTags.OLDOVERRIDE))
    if sim.boolAnd32(data['bitCoded'],8)>0 then
        if version>30303 then
            -- Check if some models are coincident
            -- We check only models with tags that start with 'XYZ_' or tags that are simBWF.modelTags.RAGNAR or simBWF.modelTags.CONVEYOR
            local objects=sim.getObjectsInTree(sim.handle_scene)
            local modelsWithTags={}
            for i=1,#objects,1 do
                local h=objects[i]
                local p=sim.getModelProperty(h)
                if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
                    local tags=sim.readCustomDataBlockTags(h)
                    if tags then
                        for j=1,#tags,1 do
                            local tag=tags[j]
                            if string.find(tag,'XYZ_')==1 or tag==simBWF.modelTags.RAGNAR or tag==simBWF.modelTags.CONVEYOR then
                                if tag~=simBWF.modelTags.PART and tag~='XYZ_PARTLABEL_INFO' then
                                    if modelsWithTags[tag]==nil then
                                        modelsWithTags[tag]={}
                                    end
                                    modelsWithTags[tag][#modelsWithTags[tag]+1]=h
                                    break
                                end
                            end
                        end
                    end
                end
            end
            -- We now loop through models with same tags and check if they are coincident (i.e. with a tolerance)
            local hh1=-1
            local hh2=-1
            for key,value in pairs(modelsWithTags) do
                if #value>1 then
                    for i=1,#value-1,1 do
                        local h1=value[i]
                        for j=i+1,#value,1 do
                            local h2=value[j]
                            local p=sim.getObjectPosition(h1,h2)
                            local e=sim.getObjectOrientation(h1,h2)
                            local dl=math.sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3])
                            local da=math.max(e[1],math.max(e[2],e[3]))
                            if dl<0.01 and da<5*math.pi/180 then
                                hh1=h1
                                hh2=h2
                                break
                            end
                        end
                        if hh1>=0 then
                            break
                        end
                    end
                end
            end
            if hh1>=0 then
                local msg="Detected at least two coincident models: '"..sim.getObjectName(hh1).."' and '"..sim.getObjectName(hh2).."'. Simulation might not run as expected."
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Coincident Models",msg)
            end
        end
    end
end
