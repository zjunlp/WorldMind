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
    data['size']=nil
    
    if not data['version'] then
        data['version']=1
    end
    if not data['gripperType'] then
        data['gripperType']={0,0,3,0,0,1,2,0,1} -- == xxx.yyy.abc = xxx=type (3=4fingers), yyy=notUsed, a=material, b=0, c=nails/noNails (1=steel nails)
    end
    if not data['subtype'] then
        data['subtype']=getGripperTypeString(data['gripperType'])
    end
    if not data['stacking'] then
        data['stacking']=1
    end
    if not data['stackingShift'] then
        data['stackingShift']=0.01
    end
    if not data['kinematricsParams'] then
        data['kinematricsParams']={0.15,30*math.pi/180,120*math.pi/180} -- i.e. r, gamma1 and gamma2 (needed to compute the workspace for instance)
    end
    -- Following groups part pick/place settings. both can be overridden by a part or pallet item
    if not data['pickAndPlaceInfo'] then
        data['pickAndPlaceInfo']={}
    end
    simBWF._getPickPlaceSettingsDefaultInfoForNonExistingFields(data.pickAndPlaceInfo)
    
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



-- Ragnar gripper referenced object slots (do not modify):
-------------------------------------------------------



-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.hand=sim.getObjectHandle('RagnarGripper_hand')
model.handles.nails=sim.getObjectHandle('RagnarGripper_nails')
model.handles.sensor=sim.getObjectHandle('RagnarGripper_sensor')
model.handles.attachPt=sim.getObjectHandle('RagnarGripper_attachPt')
