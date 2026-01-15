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
    data['resolution']=nil
    data['clippPlanes']=nil
    data['fov']=nil
    data['detectionPolygonSimulation']=nil
    data['showDetections']=nil
    data['deviceId']=nil
    data['detectorExtraLength']=nil

    -- Following currently not used anymore, but keep in case people change their mind again:
    if not data['imgProcessingParams'] then
        data['imgProcessingParams']={{},{}} -- simulation and real parameters
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
    ------
    
    
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']='vision'
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
        data['bitCoded']=1 -- 1=hide detector box during simulation, 2=flipped 180 in rel. to conveyor frame
    end
    if not data['calibrationBallDistance'] then
        data['calibrationBallDistance']=1
    end
    
    -- to correct a bug (6/12/2017):
    if data.calibration and not data.calibrationMatrix then 
        local m=simBWF.getMatrixFromCalibrationBallPositions(data.calibration[1],data.calibration[2],data.calibration[3])
        data.calibrationMatrix=m
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



-- Ragnar vision referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.CONVEYOR=1
-- model.objRefIdx.CAMERA=2 not used anymore
model.objRefIdx.INPUT=3
model.objRefIdx.PALLET=4
model.objRefJobInfo={6,4} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.calibrationBalls={model.handle}
model.handles.detectorBox=sim.getObjectHandle('VisionWindow_detectorBox@silentError')
model.handles.detectorSensor=sim.getObjectHandle('VisionWindow_detectorSensor@silentError')
for i=2,3,1 do
    model.handles.calibrationBalls[i]=sim.getObjectHandle('VisionWindow_calibrationBall'..i..'@silentError')
end
if model.handles.detectorBox==-1 then
    -- for backward compatibility:
    model.handles.detectorBox=sim.getObjectHandle('RagnarVision_detectorBox')
    model.handles.detectorSensor=sim.getObjectHandle('RagnarVision_detectorSensor')
    for i=2,3,1 do
        model.handles.calibrationBalls[i]=sim.getObjectHandle('RagnarVision_calibrationBall'..i)
    end
end