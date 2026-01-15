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
    data['subtype']='thermoformer'

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
    
    if not data.thermo_rowColCnt then
        data.thermo_rowColCnt={3,2}
    end
    if not data.thermo_rowColStep then
        data.thermo_rowColStep={0.15,0.250}
    end
    if not data.thermo_extrusionSize then
        data.thermo_extrusionSize={0.2,0.1,0.08}
    end
    if not data.thermo_wallThickness then
        data.thermo_wallThickness=0.002
    end
    if not data.thermo_stationCnt then
        data.thermo_stationCnt=4
    end
    if not data.thermo_stationSpacing then
        data.thermo_stationSpacing=0.2
    end
    if not data.thermo_dwellTime then
        data.thermo_dwellTime=1
    end
    if not data.thermo_color then
        data.thermo_color={0.8,0.8,1}
    end

    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['velocity'] then
        data['velocity']=0.1
    end
    if not data['acceleration'] then
        data['acceleration']=0.01
    end
    if not data.detectionOffset then
        data.detectionOffset={{0,0.1,0},{0,0.1,0}} -- first is simulation, next is online
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
    destination.velocity=origin.velocity
    destination.acceleration=origin.acceleration
    destination.detectionOffset=origin.detectionOffset
end


-- Conveyor/Pingpong/thermoformer referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.STOPSIGNAL=1
model.objRefIdx.STARTSIGNAL=2
model.objRefIdx.MASTERCONVEYOR=3
model.objRefIdx.OUTPUTBOX=4
model.objRefIdx.PALLET=5
model.objRefJobInfo={8,5} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope

-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.base=sim.getObjectHandle('genericThermoformer_base')
model.handles.station=sim.getObjectHandle('genericThermoformer_station')
model.handles.otherStations=sim.getObjectHandle('genericThermoformer_otherStations')
model.handles.boxes=sim.getObjectHandle('genericThermoformer_boxes')
model.handles.trigger=sim.getObjectHandle('genericThermoformer_trigger')


