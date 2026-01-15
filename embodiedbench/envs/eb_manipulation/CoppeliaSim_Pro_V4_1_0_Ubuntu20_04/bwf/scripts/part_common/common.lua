-- Functions:
-------------------------------------------------------
function model.readInfo()
    -- Read all the data stored in the model
    
    local data=sim.readCustomDataBlock(model.handle,model.tagName)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    
    -- All the data stored in the model. Set-up default values, and remove unused values
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']=model.partType
    end
    model.completeDataPartSpecific(data)
    if model.partType=='box' or model.partType=='pillowBag' or model.partType=='shippingBox' then
        -- For backward compatibility:
        if data['width'] then
            data.partSpecific.width=data.width
            data['width']=nil
        end
        if data['length'] then
            data.partSpecific.length=data.length
            data['length']=nil
        end
        if data['height'] then
            data.partSpecific.height=data.height
            data['height']=nil
        end
        if data['mass'] then
            data.partSpecific.mass=data.mass
            data['mass']=nil
        end
        if data['bitCoded'] then
            data.partSpecific.bitCoded=data.bitCoded
            data['bitCoded']=nil
        end
    end
    if model.partType=='cylinder' then
        -- For backward compatibility:
        if data['diameter'] then
            data.partSpecific.diameter=data.diameter
            data['diameter']=nil
        end
        if data['count'] then
            data.partSpecific.count=data.count
            data['count']=nil
        end
        if data['height'] then
            data.partSpecific.height=data.height
            data['height']=nil
        end
        if data['mass'] then
            data.partSpecific.mass=data.mass
            data['mass']=nil
        end
        if data['bitCoded'] then
            data.partSpecific.bitCoded=data.bitCoded
            data['bitCoded']=nil
        end
        if data['offset'] then
            data.partSpecific.offset=data.offset
            data['offset']=nil
        end
    end
    if model.partType=='sphere' then
        -- For backward compatibility:
        if data['diameter'] then
            data.partSpecific.diameter=data.diameter
            data['diameter']=nil
        end
        if data['count'] then
            data.partSpecific.count=data.count
            data['count']=nil
        end
        if data['mass'] then
            data.partSpecific.mass=data.mass
            data['mass']=nil
        end
        if data['bitCoded'] then
            data.partSpecific.bitCoded=data.bitCoded
            data['bitCoded']=nil
        end
        if data['offset'] then
            data.partSpecific.offset=data.offset
            data['offset']=nil
        end
    end
    if model.partType=='packingBox' then
        -- For backward compatibility:
        if data['width'] then
            data.partSpecific.width=data.width
            data['width']=nil
        end
        if data['length'] then
            data.partSpecific.length=data.length
            data['length']=nil
        end
        if data['height'] then
            data.partSpecific.height=data.height
            data['height']=nil
        end
        if data['mass'] then
            data.partSpecific.mass=data.mass
            data['mass']=nil
        end
        if data['bitCoded'] then
            data.partSpecific.bitCoded=data.bitCoded
            data['bitCoded']=nil
        end
        if data['thickness'] then
            data.partSpecific.thickness=data.thickness
            data['thickness']=nil
        end
        if data['closePartALength'] then
            data.partSpecific.closePartALength=data.closePartALength
            data['closePartALength']=nil
        end
        if data['closePartAWidth'] then
            data.partSpecific.closePartAWidth=data.closePartAWidth
            data['closePartAWidth']=nil
        end
        if data['closePartBLength'] then
            data.partSpecific.closePartBLength=data.closePartBLength
            data['closePartBLength']=nil
        end
        if data['closePartBWidth'] then
            data.partSpecific.closePartBWidth=data.closePartBWidth
            data['closePartBWidth']=nil
        end
        if data['inertiaFactor'] then
            data.partSpecific.inertiaFactor=data.inertiaFactor
            data['inertiaFactor']=nil
        end
        if data['lidTorque'] then
            data.partSpecific.lidTorque=data.lidTorque
            data['lidTorque']=nil
        end
        if data['lidSpring'] then
            data.partSpecific.lidSpring=data.lidSpring
            data['lidSpring']=nil
        end
        if data['lidDamping'] then
            data.partSpecific.lidDamping=data.lidDamping
            data['lidDamping']=nil
        end
    end
    if model.partType=='tray' then
        -- For backward compatibility:
        if data['width'] then
            data.partSpecific.width=data.width
            data['width']=nil
        end
        if data['length'] then
            data.partSpecific.length=data.length
            data['length']=nil
        end
        if data['height'] then
            data.partSpecific.height=data.height
            data['height']=nil
        end
        if data['mass'] then
            data.partSpecific.mass=data.mass
            data['mass']=nil
        end
        if data['bitCoded'] then
            data.partSpecific.bitCoded=data.bitCoded
            data['bitCoded']=nil
        end
        if data['borderHeight'] then
            data.partSpecific.borderHeight=data.borderHeight
            data['borderHeight']=nil
        end
        if data['borderThickness'] then
            data.partSpecific.borderThickness=data.borderThickness
            data['borderThickness']=nil
        end
        if data['pocketType'] then
            data.partSpecific.pocketType=data.pocketType
            data['pocketType']=nil
        end
        if data['linePocket'] then
            data.partSpecific.linePocket=data.linePocket
            data['linePocket']=nil
        end
        if data['honeyPocket'] then
            data.partSpecific.honeyPocket=data.honeyPocket
            data['honeyPocket']=nil
        end
        if data['placeOffset'] then
            data.partSpecific.placeOffset=data.placeOffset
            data['placeOffset']=nil
        end
    end
    
    return data
end

function model.writeInfo(data)
    -- Write all the data stored in the model. Before writing, make sure to always first read with readInfo()
    
    if data then
        sim.writeCustomDataBlock(model.handle,model.tagName,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model.handle,model.tagName,'')
    end
end


-- referenced object slots (do not modify):
-------------------------------------------------------



-- Handles:
-------------------------------------------------------
model.handles={}

