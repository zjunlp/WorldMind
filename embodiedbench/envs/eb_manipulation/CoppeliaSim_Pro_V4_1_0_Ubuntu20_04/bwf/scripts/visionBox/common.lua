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
        data['subtype']='visionBox'
    end
    data['visionBoxSerial']=nil
    if not data['visionBoxServerName'] then
        data['visionBoxServerName']=simBWF.NONE_TEXT
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=hidden during simulation
    end
    if data['visionBoxJson'] then
        data['visionBoxJsonOnline']=data['visionBoxJson']
        data['visionBoxJsonOffline']=data['visionBoxJson']
        data['visionBoxJson']=nil
    end
    if data['visionBoxJsonOnline']=='' and data['visionBoxJsonOffline']~='' then
        data['visionBoxJsonOnline']=data['visionBoxJsonOffline']
    end
    if data['visionBoxJsonOffline']=='' and data['visionBoxJsonOnline']~='' then
        data['visionBoxJsonOffline']=data['visionBoxJsonOnline']
    end
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['visionBoxJsonOnline'] then
        data['visionBoxJsonOnline']=''
    end
    if not data['visionBoxJsonOffline'] then
        data['visionBoxJsonOffline']=''
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
    destination.visionBoxJsonOnline=origin.visionBoxJsonOnline
    destination.visionBoxJsonOffline=origin.visionBoxJsonOffline
end


-- Various constants
-------------------------------------------------------
C={}
C.CAMERACNT=4
C.VISIONWINDOWCNT=4
C.LOCATIONFRAMECNT=4

-- IOHUB referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.CAMERA1=1
model.objRefIdx.CAMERA2=2
model.objRefIdx.CAMERA3=3
model.objRefIdx.CAMERA4=4
model.objRefIdx.VISIONWINDOW1=11
model.objRefIdx.VISIONWINDOW2=12
model.objRefIdx.VISIONWINDOW3=13
model.objRefIdx.VISIONWINDOW4=14
model.objRefIdx.LOCATIONFRAME1=21
model.objRefIdx.LOCATIONFRAME2=22
model.objRefIdx.LOCATIONFRAME3=23
model.objRefIdx.LOCATIONFRAME4=24

model.objRefJobInfo={30} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
model.handles={}
model.handles.body=sim.getObjectHandle('visionBox_obj')
