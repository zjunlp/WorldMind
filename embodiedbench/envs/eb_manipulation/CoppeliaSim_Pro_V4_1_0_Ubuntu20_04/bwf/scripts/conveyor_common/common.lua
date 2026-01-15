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
    data['subtype']=model.conveyorType

    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['velocity'] then
        data['velocity']=0.1
    end
    if not data['acceleration'] then
        data['acceleration']=0.01
    end
    ----------------------------------------------------------------------------------------------

    if not data['length'] then
        data['length']=1
    end
    if not data['width'] then
        data['width']=0.3
    end
    if not data['height'] then
        data['height']=0.1
    end

    if data['bitCoded'] then
        data['enabled']=sim.boolAnd32(data['bitCoded'],64)>0
    end

    if not data['enabled'] then
        data['enabled']=true
    end

    if not data['stopRequests'] then
        data['stopRequests']={}
    end
    if not data['calibration'] then
        data['calibration']=0.04    -- in mm/pulse
    end
    if not data['stopCmd'] then
        data['stopCmd']='M860'
    end
    if not data['startCmd'] then
        data['startCmd']='M861'
    end
    if not data['triggerDistance'] then
        data['triggerDistance']=0
    end


    data.deviceId=nil
    --[[
    if not data.deviceId then
        data.deviceId=simBWF.NONE_TEXT
    end
    --]]

    model.completeDataConveyorSpecific(data)
    if model.conveyorType=='A' then
        -- For backward compatibility:
        if data['borderHeight'] then
            data.conveyorSpecific.borderHeight=data.borderHeight
            data['borderHeight']=nil --0.2
        end
        if data['wallThickness'] then
            data.conveyorSpecific.wallThickness=data.wallThickness
            data['wallThickness']=nil --0.005
        end
        if data['bitCoded'] then
            data.conveyorSpecific.bitCoded=data.bitCoded
            data.conveyorSpecific.bitCoded=sim.boolOr32(data.conveyorSpecific.bitCoded,64)-64
            data['bitCoded']=nil -- 1+2+4+8
        end
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
    destination.velocity=origin.velocity
    destination.acceleration=origin.acceleration
end


-- Conveyor/Pingpong/thermoformer referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.STOPSIGNAL=1
model.objRefIdx.STARTSIGNAL=2
model.objRefIdx.MASTERCONVEYOR=3
model.objRefIdx.OUTPUTBOX=4
model.objRefJobInfo={8} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope

-- Handles:
-------------------------------------------------------
model.handles={}
