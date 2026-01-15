-- Functions:
-------------------------------------------------------
function model.completeDataPartSpecific(data)
    if not data.partSpecific then
        data.partSpecific={}
    end
    if not data.partSpecific['width'] then
        data.partSpecific['width']=0.4
    end
    if not data.partSpecific['length'] then
        data.partSpecific['length']=1
    end
    if not data.partSpecific['height'] then
        data.partSpecific['height']=0.1
    end
    if not data.partSpecific['bitCoded'] then
        data.partSpecific['bitCoded']=1 -- 1:base+borders have same color
    end
    if not data.partSpecific['mass'] then
        data.partSpecific['mass']=0.5
    end
    if not data.partSpecific['borderHeight'] then
        data.partSpecific['borderHeight']=0.05
    end
    if not data.partSpecific['borderThickness'] then
        data.partSpecific['borderThickness']=0.005
    end
    if not data.partSpecific['pocketType'] then
        data.partSpecific['pocketType']=0 -- 0=none, 1=rectangle, 2=honeycomb
    end
    if not data.partSpecific['linePocket'] then
        data.partSpecific['linePocket']={0.01,0.005,3,3} -- height, thickness, row, col
    end
    if not data.partSpecific['honeyPocket'] then
        data.partSpecific['honeyPocket']={0.01,0.005,3,3,false} -- height, thickness, row, col, first is odd
    end
    if not data.partSpecific['placeOffset'] then
        data.partSpecific['placeOffset']={0,0,0}
    end
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

model.specHandles.connection=sim.getObjectHandle('genericTray_borderConnection')
model.specHandles.border=sim.getObjectChild(model.specHandles.connection,0)
model.specHandles.borderElement=sim.getObjectHandle('genericTray_borderElement')
