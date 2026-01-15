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
        data['type']=0 -- 0 is pick, 1 is place
    end
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['sizes'] then
        data['sizes']={0.4,0.3,0.4}
    end
    if not data['offsets'] then
        data['offsets']={-0.2,0,0}
    end
    if not data['stopLinePos'] then
        data['stopLinePos']=0.1
    end
    if not data['startLinePos'] then
        data['startLinePos']=data['stopLinePos']-0.01
    end
    if not data['upstreamMarginPos'] then
        data['upstreamMarginPos']=0.1
    end
    if not data['bitCoded'] then
        data['bitCoded']=1 -- 1=hidden during sim, 2=calibration balls hidden during sim, 4=showPts, 8=create parts (online mode), 16=startStopLine enable, 32=pick also without target in sight (deprecated)
    end
    ----------------------------------------------------------------------------------------------
    data['palletId']=nil
    if not data['calibrationBallOffset'] then
        if not data['calibrationBallDistance'] then
            data['calibrationBallOffset']={1,0,0}
        else
            data['calibrationBallOffset']={data['calibrationBallDistance'],0,0}
            data['calibrationBallDistance']=nil
        end
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
    destination.sizes=origin.sizes
    destination.offsets=origin.offsets
    destination.stopLinePos=origin.stopLinePos
    destination.startLinePos=origin.startLinePos
    destination.upstreamMarginPos=origin.upstreamMarginPos
    destination.bitCoded=origin.bitCoded
end


-- Tracking window referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
--model.objRefIdx.PALLET=1 REMOVED on 7/8/2018
model.objRefIdx.INPUT=2
model.objRefIdx.PARTTYPE=3
model.objRefJobInfo={5,3} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope


-- Handles:
-------------------------------------------------------
local isPick=(model.readInfo()['type']==0)
model.handles={}

if isPick then
    model.handles.trackBox1=sim.getObjectHandle('pickTrackingWindow_box1')
    model.handles.trackBox2=sim.getObjectHandle('pickTrackingWindow_box2')
    model.handles.stopLineBox=sim.getObjectHandle('pickTrackingWindow_stopLine')
    model.handles.startLineBox=sim.getObjectHandle('pickTrackingWindow_startLine')
    model.handles.refFrame=sim.getObjectHandle('pickTrackingWindow_refFrame')
    model.handles.upstreamMarginBox=sim.getObjectHandle('pickTrackingWindow_upstreamMargin')
else
    model.handles.trackBox1=sim.getObjectHandle('placeTrackingWindow_box1')
    model.handles.trackBox2=sim.getObjectHandle('placeTrackingWindow_box2')
    model.handles.stopLineBox=sim.getObjectHandle('placeTrackingWindow_stopLine')
    model.handles.startLineBox=sim.getObjectHandle('placeTrackingWindow_startLine')
    model.handles.refFrame=sim.getObjectHandle('placeTrackingWindow_refFrame')
    model.handles.upstreamMarginBox=sim.getObjectHandle('placeTrackingWindow_upstreamMargin')
end
model.handles.calibrationBalls={}
for i=1,3,1 do
    if isPick then
        model.handles.calibrationBalls[i]=sim.getObjectHandle('pickTrackingWindow_calibrationBall'..i)
    else
        model.handles.calibrationBalls[i]=sim.getObjectHandle('placeTrackingWindow_calibrationBall'..i)
    end
end
