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
        data.partSpecific['bitCoded']=0 -- all free
    end
    if not data.partSpecific['mass'] then
        data.partSpecific['mass']=0.5
    end
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

