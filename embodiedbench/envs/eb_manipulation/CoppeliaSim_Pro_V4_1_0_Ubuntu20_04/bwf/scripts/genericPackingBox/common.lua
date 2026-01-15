-- Functions:
-------------------------------------------------------
function model.completeDataPartSpecific(data)
    if not data.partSpecific then
        data.partSpecific={}
    end
    if not data.partSpecific['width'] then
        data.partSpecific['width']=0.3
    end
    if not data.partSpecific['length'] then
        data.partSpecific['length']=0.3
    end
    if not data.partSpecific['height'] then
        data.partSpecific['height']=0.3
    end
    if not data.partSpecific['bitCoded'] then
        data.partSpecific['bitCoded']=1+2+4 -- 1=partA, 2=partB, 4=textured
    end
    if not data.partSpecific['mass'] then
        data.partSpecific['mass']=0.5
    end
    if not data.partSpecific['thickness'] then
        data.partSpecific['thickness']=0.003
    end
    if not data.partSpecific['closePartALength'] then
        data.partSpecific['closePartALength']=0.5
    end
    if not data.partSpecific['closePartAWidth'] then
        data.partSpecific['closePartAWidth']=1
    end
    if not data.partSpecific['closePartBLength'] then
        data.partSpecific['closePartBLength']=0.5
    end
    if not data.partSpecific['closePartBWidth'] then
        data.partSpecific['closePartBWidth']=0.9
    end
    if not data.partSpecific['inertiaFactor'] then
        data.partSpecific['inertiaFactor']=1
    end
    if not data.partSpecific['lidTorque'] then
        data.partSpecific['lidTorque']=0.1
    end
    if not data.partSpecific['lidSpring'] then
        data.partSpecific['lidSpring']=1
    end
    if not data.partSpecific['lidDamping'] then
        data.partSpecific['lidDamping']=0
    end
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

model.specHandles.bb=sim.getObjectHandle('genericPackingBox_bb')
model.specHandles.sideConnection=sim.getObjectHandle('genericPackingBox_sideConnection')
model.specHandles.sides=sim.getObjectChild(model.specHandles.sideConnection,0)
model.specHandles.joints={}
model.specHandles.lids={}
for i=1,4,1 do
    model.specHandles.joints[i]=sim.getObjectHandle('genericPackingBox_j'..i)
    model.specHandles.lids[i]=sim.getObjectChild(model.specHandles.joints[i],0)
end
