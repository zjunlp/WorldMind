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
        data['subtype']='frame'
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



-- Handles:
-------------------------------------------------------
model.handles={}

model.handles.centralResizeShapes={}
model.handles.centralResizeShapes[1]=sim.getObjectHandle('RagnarFrame_centralTop')

model.handles.widthJoints={}
model.handles.widthJoints[1]=sim.getObjectHandle('RagnarFrame_widthJ1')
model.handles.widthJoints[2]=sim.getObjectHandle('RagnarFrame_widthJ2')

model.handles.heightJoints={}
model.handles.heightJoints[1]=sim.getObjectHandle('RagnarFrame_heightJ1')
model.handles.heightJoints[2]=sim.getObjectHandle('RagnarFrame_heightJ2')
model.handles.heightJoints[3]=sim.getObjectHandle('RagnarFrame_heightJ3')
model.handles.heightJoints[4]=sim.getObjectHandle('RagnarFrame_heightJ4')
model.handles.heightJoints[5]=sim.getObjectHandle('RagnarFrame_heightJ5')

model.handles.doorJoints={}
model.handles.doorJoints[1]=sim.getObjectHandle('RagnarFrame_doorJ1')
model.handles.doorJoints[2]=sim.getObjectHandle('RagnarFrame_doorJ2')

model.handles.doorShapes={}
model.handles.doorShapes[1]=sim.getObjectHandle('RagnarFrame_topRightDoor')
model.handles.doorShapes[2]=sim.getObjectHandle('RagnarFrame_topLeftDoor')
model.handles.doorShapes[3]=sim.getObjectHandle('RagnarFrame_bottomRightDoor')
model.handles.doorShapes[4]=sim.getObjectHandle('RagnarFrame_bottomLeftDoor')
