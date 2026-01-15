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
    data['dwellTime']=nil
    data['algorithm']=nil
    data['pickOffset']=nil
    data['placeOffset']=nil
    data['pickRounding']=nil
    data['placeRounding']=nil
    data['pickNulling']=nil
    data['placeNulling']=nil
    data['pickApproachHeight']=nil
    data['placeApproachHeight']=nil
    data['gripperActionsWithColorChange']=nil
    data['dynamics']=nil
    
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']='ragnar'
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=wsbox visible, 2=free, 4=wsbox visible during run,8= free, 16=free, 64=enabled, 128=free, 256=show ws, 512=show ws also during simulation 1024=free  2048=free 4096=free
    end
    -- Job related (see also functions model.copyModelParamsToJob and model.copyJobToModelParams):
    ----------------------------------------------------------------------------------------------
    if not data['maxVel'] then
        data['maxVel']=1
    end
    if not data['maxAccel'] then
        data['maxAccel']=1
    end
    if not data['waitLocAfterPickOrPlace'] then
        data['waitLocAfterPickOrPlace']={{0,0,-0.35},{0,0,-0.35}}
    end
    if not data.jobBitCoded then
        data.jobBitCoded=data.bitCoded -- 8= hideRobotBase 16=hideFrame 1024=attach part to target via a force sensor, 2048=pick part without target in sight,4096=ignore part destinations
    end
    ----------------------------------------------------------------------------------------------
    if not data['primaryArmLengthInMM'] then
        data['primaryArmLengthInMM']=300
    end
    if not data['secondaryArmLengthInMM'] then
        data['secondaryArmLengthInMM']=550
    end
    if not data['frameHeightInMM'] then
        data['frameHeightInMM']=1800 -- nominal
    end
    if not data['connectionBufferSize'] then
        data['connectionBufferSize']={1000,1000} -- simulation and real
    end
    if not data['showPlot'] then
        data['showPlot']={false,false} -- simulation and real
    end
    if not data['showTrajectory'] then
        data['showTrajectory']={false,false} -- simulation and real
    end
    if not data['visualizeUpdateFrequ'] then
        data['visualizeUpdateFrequ']={0,0} -- simulation and real parameters. 0=always, 1=medium (every 200ms), 2=rare (every 1s)
    end
    if not data['motorType'] then
        data['motorType']=C.MOTORTYPELIST[1] -- 0, i.e. standard
    end
    if not data['exteriorType'] then
        data['exteriorType']=C.EXTERIORTYPELIST[1] -- 0, i.e. standard
    end
    if not data['frameType'] then
        data['frameType']=C.FRAMETYPELIST[1] -- 0, i.e. experimental
    end
    if not data['frameDoor'] then
        data['frameDoor']=C.FRAMEDOORSTATELIST[1] -- 0, i.e. closed
    end
    if not data['clearance'] then
        data['clearance']={false,false} -- simulation and real parameters.
    end
    if not data['clearanceWithPlatform'] then
        data['clearanceWithPlatform']={false,false} -- simulation and real parameters.
    end
    if not data['clearanceForAllSteps'] then
        data['clearanceForAllSteps']={false,false} -- simulation and real parameters.
    end
    if not data['clearanceWarning'] then
        data['clearanceWarning']={0,0} -- simulation and real parameters.
    end
    data['deviceId']=nil
    if not data['wsBox'] then
        data['wsBox']={{-0.3,-0.4,-0.8},{0.3,0.4,-0.3}}
    end
    data.lastStoredJob=nil
    data.jobs=nil
    if not data.jobData then
        data.jobData={version=1,activeJobInModel=nil,jobs={}}
    end
    -- for backward compatibility:
    -----------------------------------------------------------
    if not data.jobData.objRefJobInfo then
        data.jobData.objRefJobInfo=model.objRefJobInfo
    end
    local cnt=1
    for key,value in pairs(data.jobData.jobs) do
        if value.jobIndex==nil then
            value.jobIndex=cnt
        end
        cnt=cnt+1
    end
    -----------------------------------------------------------
    
    if not data['robotAlias'] then
        data['robotAlias']=simBWF.NONE_TEXT
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


function model.readJobModelInfo()
    -- Read all the data stored in the model
    
    local data=sim.readCustomDataBlock(model.handle,C.JOBMODELDATATAG)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    
    return data
end

function model.writeJobModelInfo(data)
    -- Write all the data stored in the model. Before writing, make sure to always first read with readInfo()
    
    if data then
        sim.writeCustomDataBlock(model.handle,C.JOBMODELDATATAG,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model.handle,C.JOBMODELDATATAG,'')
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
    destination.maxVel=origin.maxVel
    destination.maxAccel=origin.maxAccel
    destination.waitLocAfterPickOrPlace=origin.waitLocAfterPickOrPlace
    destination.jobBitCoded=origin.jobBitCoded
end


-- Various constants
-------------------------------------------------------
C={}
C.CIC=4 -- Connected Item Count (for each category: pick tracking window, place tracking window, pick location frame, place location frame)
C.ECIC_PiW= 2 -- Exposed Connected Item Count, Pick Window
C.ECIC_PlW= 2 -- Exposed Connected Item Count, Place Window
C.ECIC_PiL= 4 -- Exposed Connected Item Count, Pick Location
C.ECIC_PlL= 4 -- Exposed Connected Item Count, Place Location

C.MOTORTYPELIST={0,1,2}
C.MOTORTYPES={}
C.MOTORTYPES[C.MOTORTYPELIST[1]]={text='standard',pricingText='standard',maxVel=5,maxAccel=35}
C.MOTORTYPES[C.MOTORTYPELIST[2]]={text='high-power',pricingText='high-power',maxVel=2.5,maxAccel=25}
C.MOTORTYPES[C.MOTORTYPELIST[3]]={text='high-torque',pricingText='high-torque',maxVel=2.5,maxAccel=25}

C.EXTERIORTYPELIST={0,1,2}
C.EXTERIORTYPES={}
C.EXTERIORTYPES[C.EXTERIORTYPELIST[1]]={text='standard',pricingText='std'}
C.EXTERIORTYPES[C.EXTERIORTYPELIST[2]]={text='wash-down',pricingText='wd'}
C.EXTERIORTYPES[C.EXTERIORTYPELIST[3]]={text='hygienic',pricingText='hg'}

C.FRAMETYPELIST={0,1}
C.FRAMETYPES={}
C.FRAMETYPES[C.FRAMETYPELIST[1]]={text='experimental',pricingText='experimental'}
C.FRAMETYPES[C.FRAMETYPELIST[2]]={text='industrial',pricingText='industrial'}

C.FRAMEDOORSTATELIST={0,1,2}
C.FRAMEDOORSTATES={}
C.FRAMEDOORSTATES[C.FRAMEDOORSTATELIST[1]]={text='closed'}
C.FRAMEDOORSTATES[C.FRAMEDOORSTATELIST[2]]={text='open'}
C.FRAMEDOORSTATES[C.FRAMEDOORSTATELIST[3]]={text='hidden'}

C.JOBMODELDATATAG='RAGNAR_JOBMODELS_INFO'

-- Ragnar referenced object slots (do not modify):
-------------------------------------------------------
model.objRefIdx={}
model.objRefIdx.PICKTRACKINGWINDOW1=1
model.objRefIdx.PICKTRACKINGWINDOW2=2
model.objRefIdx.PICKTRACKINGWINDOW3=3
model.objRefIdx.PICKTRACKINGWINDOW4=4
model.objRefIdx.PLACETRACKINGWINDOW1=11
model.objRefIdx.PLACETRACKINGWINDOW2=12
model.objRefIdx.PLACETRACKINGWINDOW3=13
model.objRefIdx.PLACETRACKINGWINDOW4=14
model.objRefIdx.PICKFRAME1=21
model.objRefIdx.PICKFRAME2=22
model.objRefIdx.PICKFRAME3=23
model.objRefIdx.PICKFRAME4=24
model.objRefIdx.PLACEFRAME1=31
model.objRefIdx.PLACEFRAME2=32
model.objRefIdx.PLACEFRAME3=33
model.objRefIdx.PLACEFRAME4=34

model.objRefIdx.CONVEYOR1=41
model.objRefIdx.CONVEYOR2=42

model.objRefIdx.INPUT1=41 -- actually conveyor1
model.objRefIdx.INPUT2=42 -- actually conveyor2
model.objRefIdx.INPUT3=43
model.objRefIdx.INPUT4=44
model.objRefIdx.INPUT5=45
model.objRefIdx.INPUT6=46
model.objRefIdx.INPUT7=47
model.objRefIdx.INPUT8=48

model.objRefIdx.OUTPUT1=51
model.objRefIdx.OUTPUT2=52
model.objRefIdx.OUTPUT3=53
model.objRefIdx.OUTPUT4=54
model.objRefIdx.OUTPUT5=55
model.objRefIdx.OUTPUT6=56
model.objRefIdx.OUTPUT7=57
model.objRefIdx.OUTPUT8=58

model.objRefJobInfo={70} -- information about jobs stored in object references. Item 1 is where job related obj refs start, other items are obj ref indices that are in the job scope

-- Handles:
-------------------------------------------------------
model.handles={}

model.handles.ragnarRef=sim.getObjectHandle('Ragnar_ref')
model.handles.ragnarGripperPlatformAttachment=sim.getObjectHandle('Ragnar_gripperPlatformAttachment')
model.handles.ragnarWs=sim.getObjectHandle('Ragnar_ws')
model.handles.ragnarWsBox=sim.getObjectHandle('Ragnar_wsBox')

model.handles.alphaOffsetJ1=sim.getObjectHandle('Ragnar_zRotLeftFront')
model.handles.betaOffsetJ1=sim.getObjectHandle('Ragnar_xRotLeftFront')
model.handles.xOffsetJ1=sim.getObjectHandle('Ragnar_yOffsetLeft')
model.handles.yOffsetJ1=sim.getObjectHandle('Ragnar_xOffsetLeftFront')

model.handles.ikTips={}
for i=1,4,1 do
    model.handles.ikTips[i]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'a_tip')
end

model.handles.motorJoints={}
for i=1,4,1 do
    model.handles.motorJoints[i]=sim.getObjectHandle('Ragnar_motor'..i)
end

model.handles.tiltAdjustmentAngles={}
model.handles.tiltAdjustmentAngles[1]=-sim.getJointPosition(sim.getObjectHandle('Ragnar_xRotLeftFront'))
model.handles.tiltAdjustmentAngles[2]=sim.getJointPosition(sim.getObjectHandle('Ragnar_xRotRightFront'))
model.handles.tiltAdjustmentAngles[3]=sim.getJointPosition(sim.getObjectHandle('Ragnar_xRotRightRear'))
model.handles.tiltAdjustmentAngles[4]=-sim.getJointPosition(sim.getObjectHandle('Ragnar_xRotLeftRear'))
model.handles.panAdjustmentAngles={}
model.handles.panAdjustmentAngles[1]=-sim.getJointPosition(sim.getObjectHandle('Ragnar_zRotLeftFront'))
model.handles.panAdjustmentAngles[2]=sim.getJointPosition(sim.getObjectHandle('Ragnar_zRotRightFront'))
model.handles.panAdjustmentAngles[3]=-sim.getJointPosition(sim.getObjectHandle('Ragnar_zRotRightRear'))
model.handles.panAdjustmentAngles[4]=sim.getJointPosition(sim.getObjectHandle('Ragnar_zRotLeftRear'))

model.handles.ikGroups={}
for i=1,4,1 do
    model.handles.ikGroups[i]=sim.getIkGroupHandle('ragnarIk_arm'..i)
end

model.handles.primaryArms={}
model.handles.secondaryArms={}

model.handles.primaryArmsEndAdjust={}
model.handles.secondaryArmsEndAdjust={}

model.handles.primaryArmsLAdjust={}
model.handles.secondaryArmsLAdjust={}

model.handles.leftAndRightSideAdjust={sim.getObjectHandle('Ragnar_yOffsetLeft'),sim.getObjectHandle('Ragnar_yOffsetRight')}

for i=1,4,1 do
    model.handles.primaryArms[#model.handles.primaryArms+1]=sim.getObjectHandle('Ragnar_primaryArm'..i..'_part2')
    model.handles.secondaryArms[#model.handles.secondaryArms+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'a_part2')
    model.handles.secondaryArms[#model.handles.secondaryArms+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'b_part2')

    model.handles.primaryArmsEndAdjust[#model.handles.primaryArmsEndAdjust+1]=sim.getObjectHandle('Ragnar_primaryArm'..i..'_adjustJ2')
    model.handles.secondaryArmsEndAdjust[#model.handles.secondaryArmsEndAdjust+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'a_adjustJ2')
    model.handles.secondaryArmsEndAdjust[#model.handles.secondaryArmsEndAdjust+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'b_adjustJ2')

    model.handles.primaryArmsLAdjust[#model.handles.primaryArmsLAdjust+1]=sim.getObjectHandle('Ragnar_primaryArm'..i..'_adjustJ1')
    model.handles.secondaryArmsLAdjust[#model.handles.secondaryArmsLAdjust+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'a_adjustJ1')
    model.handles.secondaryArmsLAdjust[#model.handles.secondaryArmsLAdjust+1]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'b_adjustJ1')
end

model.handles.centralCover={}
model.handles.centralCover[1]=sim.getObjectHandle('Ragnar_centralCover')
model.handles.nameElement={}
model.handles.nameElement[1]=sim.getObjectHandle('Ragnar_frontName')
model.handles.nameElement[2]=sim.getObjectHandle('Ragnar_rearName')

model.handles.housingItems={}
for i=1,#model.handles.centralCover,1 do
    model.handles.housingItems[#model.handles.housingItems+1]=model.handles.centralCover[i]
end
for i=1,#model.handles.nameElement,1 do
    model.handles.housingItems[#model.handles.housingItems+1]=model.handles.nameElement[i]
end
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_lightTower')
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_topCoverBump')
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_downstreamCover')
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_downstreamConnector')
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_upstreamCover')
model.handles.housingItems[#model.handles.housingItems+1]=sim.getObjectHandle('Ragnar_upstreamConnector')

model.handles.robotArmCollection=sim.getCollectionHandle('RagnarArms')
model.handles.robotArmAndPlatformCollection=sim.getCollectionHandle('RagnarArmsAndPlatform')
model.handles.robotObstaclesCollection=sim.getCollectionHandle('RagnarObstacles')

model.handles.frameModel=sim.getObjectHandle('RagnarFrame')
