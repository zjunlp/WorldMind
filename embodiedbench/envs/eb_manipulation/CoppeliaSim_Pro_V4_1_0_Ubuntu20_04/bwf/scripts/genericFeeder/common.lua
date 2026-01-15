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
    data['destinationDistribution']=nil
    data['deactivationTime']=nil
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']='feeder'
    end
    
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['size'] then
        data['size']={0.3,0,0}
    end
    if not data['frequency'] then
        data['frequency']=1
    end
    if not data['algorithm'] then
        data['algorithm']=''
    end
    if not data['conveyorDist'] then
        data['conveyorDist']=0.2
    end
    if not data['maxProductionCnt'] then
        data['maxProductionCnt']=0
    end
    if not data['partDistribution'] then
        data['partDistribution']="{1,'BOX'}"
    end
    if not data['shiftDistribution'] then
        data['shiftDistribution']="{1,-0.5},{2,-0.25},{4,0},{2,0.25},{1,0.5}"
    end
    if not data['rotationDistribution'] then
        data['rotationDistribution']="{1,-math.pi/2},{2,-math.pi/4},{4,0},{2,math.pi/4},{1,math.pi/2}"
    end
    if not data['weightDistribution'] then
        data['weightDistribution']="{1,'<DEFAULT>'}"
    end
    if not data['labelDistribution'] then
        data['labelDistribution']="{1,1+2+4}"
    end
    if not data['isoSizeScalingDistribution'] then
        data['isoSizeScalingDistribution']="{1,1}"
    end
    if not data['nonIsoSizeScalingDistribution'] then
        data['nonIsoSizeScalingDistribution']="{1,{1,1,1}}"
    end
    if not data['sizeScaling'] then
        data['sizeScaling']=0 -- 0:none, 1=iso, 2=non-iso
    end
    if not data['bitCoded'] then
        data['bitCoded']=3 -- 1=hidden, 2=enabled, 4-31:0=frequency, 4=sensor triggered, 8=user, 12=conveyorTriggered, 16=multi-feeder triggered, 20=manual trigger, 128=show statistics
    end
    if not data['multiFeederTriggerCnt'] then
        data['multiFeederTriggerCnt']=0
    end
    ----------------------------------------------------------------------------------------------
    
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
    destination.size=origin.size
    destination.frequency=origin.frequency
    destination.algorithm=origin.algorithm
    destination.conveyorDist=origin.conveyorDist
    destination.maxProductionCnt=origin.maxProductionCnt
    destination.partDistribution=origin.partDistribution
    destination.shiftDistribution=origin.shiftDistribution
    destination.rotationDistribution=origin.rotationDistribution
    destination.weightDistribution=origin.weightDistribution
    destination.labelDistribution=origin.labelDistribution
    destination.isoSizeScalingDistribution=origin.isoSizeScalingDistribution
    destination.nonIsoSizeScalingDistribution=origin.nonIsoSizeScalingDistribution
    destination.sizeScaling=origin.sizeScaling
    destination.bitCoded=origin.bitCoded
    destination.multiFeederTriggerCnt=origin.multiFeederTriggerCnt
end

-- Feeder referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.SENSOR=1
model.objRefIdx.CONVEYOR=2
model.objRefIdx.STOPSIGNAL=3
model.objRefIdx.STARTSIGNAL=4
model.objRefJobInfo={6,1,2,3,4} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}

model.handles.producedPartsDummy=sim.getObjectHandle('genericFeeder_ownedParts')


