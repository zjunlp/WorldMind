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
        data['subtype']='hub'
    end
    if not data['iohubSerial'] then
        data['iohubSerial']=simBWF.NONE_TEXT
    end
    if not data['bitCoded'] then
        data['bitCoded']=1 -- 1=hidden during simulation
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


-- IOHUB referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}

model.objRefIdx.INPUT1=1 -- actually conveyor1
model.objRefIdx.INPUT2=2 -- actually conveyor2
model.objRefIdx.INPUT3=3
model.objRefIdx.INPUT4=4
model.objRefIdx.INPUT5=5
model.objRefIdx.INPUT6=6
model.objRefIdx.INPUT7=7
model.objRefIdx.INPUT8=8

model.objRefIdx.OUTPUT1=11
model.objRefIdx.OUTPUT2=12
model.objRefIdx.OUTPUT3=13
model.objRefIdx.OUTPUT4=14
model.objRefIdx.OUTPUT5=15
model.objRefIdx.OUTPUT6=16
model.objRefIdx.OUTPUT7=17
model.objRefIdx.OUTPUT8=18

model.objRefJobInfo={22} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope

-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.body=sim.getObjectHandle('IOhub_obj')

