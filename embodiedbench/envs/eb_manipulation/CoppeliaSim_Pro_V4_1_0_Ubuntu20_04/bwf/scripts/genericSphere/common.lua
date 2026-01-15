-- Functions:
-------------------------------------------------------
function model.completeDataPartSpecific(data)
    if not data.partSpecific then
        data.partSpecific={}
    end
    if not data.partSpecific['diameter'] then
        data.partSpecific['diameter']=0.1
    end
    if not data.partSpecific['count'] then
        data.partSpecific['count']=1
    end
    if not data.partSpecific['offset'] then
        data.partSpecific['offset']=1
    end
    if not data.partSpecific['bitCoded'] then
        data.partSpecific['bitCoded']=0 -- all free
    end
    if not data.partSpecific['mass'] then
        data.partSpecific['mass']=0.5
    end
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

model.specHandles.auxSpheres={}
for i=1,3,1 do
    model.specHandles.auxSpheres[i]=sim.getObjectHandle('genericSphere_auxSphere'..i)
end
