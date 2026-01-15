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
    data['width']=nil
    data['depth']=nil
    data['height']=nil
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']='camera'
    end
    if not data['size'] then
        data['size']={0.03,0.12,0.03}
    end
    if not data['resolution'] then
        data['resolution']={640,480}
    end
    if not data['clippPlanes'] then
        data['clippPlanes']={0.04,0.75}
    end
    if not data['fov'] then
        data['fov']=60*math.pi/180
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- not used
    end
    if not data['imgToDisplay'] then
        data['imgToDisplay']={0,0} -- simulation and real parameters. 0=none, 1=rgb, 2=depth, 3=processed
    end
    if not data['imgSizeToDisplay'] then
        data['imgSizeToDisplay']={0,0} -- simulation and real parameters. 0=small, 1=medium, 2=large
    end
    if not data['imgUpdateFrequ'] then
        data['imgUpdateFrequ']={0,0} -- simulation and real parameters. 0=always, 1=medium (every 200ms), 2=rare (every 1s)
    end
    if not data['deviceId'] then
        data['deviceId']=simBWF.NONE_TEXT
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

function model.copyModelParamsToJob(data,jobName)
    -- Copy the model parameters that are job-related to a specific job
    -- Job must already exist!
    local job=data.jobData.jobs[jobName]
--    job.refreshMode=data.refreshMode
end

function model.copyJobToModelParams(data,jobName)
    -- Copy a specific job to the model parameters that are job-related
    local job=data.jobData.jobs[jobName]
--    data.refreshMode=job.refreshMode
end


-- referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
--model.objRefIdx.SOMETHING=1
model.objRefJobInfo={1} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.body=sim.getObjectHandle('RagnarCamera_body')
model.handles.arrows=sim.getObjectHandle('RagnarCamera_arrows')
model.handles.sensor=sim.getObjectHandle('RagnarCamera_sensor')

