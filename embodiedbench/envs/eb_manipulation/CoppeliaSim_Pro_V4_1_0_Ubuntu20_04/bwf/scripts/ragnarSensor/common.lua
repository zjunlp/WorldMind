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
    if not data.version then
        data.version=1
    end
    if not data.subtype then
        data.subtype='sensor'
    end
    
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data.detectionOffset then
        data.detectionOffset={{0,0.1,0},{0,0.1,0}} -- first is simulation, next is online
    end
    ----------------------------------------------------------------------------------------------
    
    if not data.bitCoded then
        data.bitCoded=0 -- 2=flipped 180 in rel. to conveyor frame
    end

    data.measurementLength=nil

    if not data.showPlot then
        data.showPlot={false,false} -- first is simulation, next is online
    end
    if not data.plotUpdateFrequ then
        data.plotUpdateFrequ={0,0} -- simulation and real parameters. 0=always, 1=medium (every 200ms), 2=rare (every 1s)
    end
    if not data.calibrationBallDistance then
        data.calibrationBallDistance=1
    end
    if not data.detectionWidth then
        data.detectionWidth=0.005
    end
    data.deviceId=nil
    
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
    destination.detectionOffset=origin.detectionOffset
end


-- Ragnar sensor referenced object slots (do not modify):
model.objRefIdx={}
model.objRefIdx.CONVEYOR=1
model.objRefIdx.INPUT=2
model.objRefIdx.PALLET=3
model.objRefJobInfo={6,3} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.sensor=sim.getObjectHandle('RagnarSensor_sensor')
model.handles.blueBall=sim.getObjectHandle('RagnarSensor_blueBall')
