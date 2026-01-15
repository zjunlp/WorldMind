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
        data['subtype']='window'
    end
    
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['detectorDiameter'] then
        data['detectorDiameter']=0.001
    end
    if not data['detectorHeight'] then
        data['detectorHeight']=0.3
    end
    if not data['detectorHeightOffset'] then
        data['detectorHeightOffset']=-0.025
    end
    ----------------------------------------------------------------------------------------------
    
    if not data['bitCoded'] then
        data['bitCoded']=1 -- 1=hide detector box during simulation, 2=flipped 180 in rel. to conveyor frame, 4=show detections
    end
    if not data['calibrationBallDistance'] then
        data['calibrationBallDistance']=1
    end
    
    if not data.jobData then
        data.jobData={version=1,activeJobInModel=nil,objRefJobInfo=model.objRefJobInfo,jobs={}}
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
    destination.detectorDiameter=origin.detectorDiameter
    destination.detectorHeight=origin.detectorHeight
    destination.detectorHeightOffset=origin.detectorHeightOffset
end


-- Ragnar detector referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.CONVEYOR=1
model.objRefIdx.INPUT=2
model.objRefIdx.PALLET=3
model.objRefJobInfo={6,3} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.detectorBox=sim.getObjectHandle('RagnarDetector_detectorBox')
model.handles.detectorSensor=sim.getObjectHandle('RagnarDetector_detectorSensor')
model.handles.calibrationBalls={model.handle}
for i=2,3,1 do
    model.handles.calibrationBalls[i]=sim.getObjectHandle('RagnarDetector_calibrationBall'..i)
end
