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
        data.partSpecific['bitCoded']=0 -- 1,2: free, 4:textured
    end
    if not data.partSpecific['mass'] then
        data.partSpecific['mass']=0.5
    end
end


function model.readPartInfo()
    local data=simBWF.readPartInfo(model.handle)
    -- Additional part data (label specific):
    
    if not data['labelData'] then
        data['labelData']={}
    end
    if not data['labelData']['bitCoded'] then
        data['labelData']['bitCoded']=0 -- 1,2,4: free, 8=label1,16=label2,32=label3,64=largeLabel1,128=largeLabel2,256=largeLabel3
    end
    if not data['labelData']['placementCode'] then
        data['labelData']['placementCode']={'{0,0,0},{0,0,0}','{0,0,0},{0,0,0}','{0,0,0},{0,0,0}'}
    end
    if not data['labelData']['smallLabelSize'] then
        data['labelData']['smallLabelSize']={0.075,0.038}
    end
    if not data['labelData']['largeLabelSize'] then
        data['labelData']['largeLabelSize']={0.075,0.1125}
    end
    if not data['labelData']['boxSize'] then
        data['labelData']['boxSize']={0.1,0.1,0.1}
    end
    return data
end

function model.writePartInfo(data)
    return simBWF.writePartInfo(model.handle,data)
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

model.specHandles.texture=sim.getObjectHandle('genericShippingBox_texture')
model.specHandles.smallLabel=sim.getObjectHandle('genericShippingBox_smallLabel')
model.specHandles.largeLabel=sim.getObjectHandle('genericShippingBox_largeLabel')

