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
        data['subtype']='partHolder'
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


-- Referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}


-- Handles:
-------------------------------------------------------
model.handles={}
