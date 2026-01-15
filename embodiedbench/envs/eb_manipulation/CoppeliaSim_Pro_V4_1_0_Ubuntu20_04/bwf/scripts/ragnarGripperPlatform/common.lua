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
        data['subtype']='platform'
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=visualize the gripper state with a different gripper platform color
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


-- Ragnar gripper platform referenced object slots (do not modify):
-------------------------------------------------------



-- Handles:
-------------------------------------------------------
model.handles={}

model.handles.ikPts={}
for i=1,4,1 do
    model.handles.ikPts[i]=sim.getObjectHandle('RagnarGripperPlatform_ikPt'..i)
end

model.handles.gripperAttachmentPoint=sim.getObjectHandle('RagnarGripperPlatform_toolAttachment')
model.handles.tracing=sim.getObjectHandle('RagnarGripperPlatform_tracing')
