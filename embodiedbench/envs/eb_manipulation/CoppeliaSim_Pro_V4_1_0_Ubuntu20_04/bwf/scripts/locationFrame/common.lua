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
        data['subtype']='pick'
    end
    if not data['type'] then
        data['type']=0 -- 0=pick, 1=place
    end
    
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['refreshMode'] then
        data['refreshMode']=1 -- 0='lua', 1='auto' or 2='vision'
    end
    ----------------------------------------------------------------------------------------------

    data['palletId']=nil
    if not data['bitCoded'] then
        data['bitCoded']=1 -- 1=hidden during sim, 2=calibration balls hidden during sim, 4=free, 8=create parts (online mode), 16=show associated pallet, 32=pick also without target in sight (deprecated)
    end
    if not data['calibration'] then
        data['calibration']=nil -- either nil, or {ball1RelPos,ball2RelPos,ball3RelPos}
    end
    if not data['calibrationMatrix'] then
        data['calibrationMatrix']=nil -- either nil, or the calibration matrix
    end

    if not data.jobData then
        data.jobData={version=1,activeJobInModel=nil,objRefJobInfo=model.objRefJobInfo,jobs={}}
    end
    
    -- to correct a bug (6/12/2017):
    if data.calibration and not data.calibrationMatrix then 
        local m=simBWF.getMatrixFromCalibrationBallPositions(data.calibration[1],data.calibration[2],data.calibration[3])
        data.calibrationMatrix=m
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
    model.copyJobRelatedDataFromTo(data,job)
end

function model.copyJobToModelParams(data,jobName)
    -- Copy a specific job to the model parameters that are job-related
    local job=data.jobData.jobs[jobName]
    model.copyJobRelatedDataFromTo(job,data)
end

function model.copyJobRelatedDataFromTo(origin,destination)
    destination.refreshMode=origin.refreshMode
end



-- Location frame referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.PALLET=1
model.objRefJobInfo={2,1} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope

-- Handles:
-------------------------------------------------------
model.handles={}

local isPick=(model.readInfo()['type']==0)
if isPick then
    model.handles.frameShape=sim.getObjectHandle('pickLocationFrame_shape')
else
    model.handles.frameShape=sim.getObjectHandle('placeLocationFrame_shape')
end

model.handles.calibrationBalls={}
for i=1,3,1 do
    if isPick then
        model.handles.calibrationBalls[i]=sim.getObjectHandle('pickLocationFrame_calibrationBall'..i)
    else
        model.handles.calibrationBalls[i]=sim.getObjectHandle('placeLocationFrame_calibrationBall'..i)
    end
end
