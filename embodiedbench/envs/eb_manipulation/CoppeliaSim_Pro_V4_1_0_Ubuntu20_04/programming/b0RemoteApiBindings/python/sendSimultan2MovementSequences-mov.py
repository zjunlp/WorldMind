import b0RemoteApi
import time
import math

with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi',60) as client:    
    client.executedMovId1='notReady'
    client.executedMovId2='notReady'

    targetArm1='threadedBlueArm'
    targetArm2='nonThreadedRedArm'

    stringSignalName1=targetArm1+'_executedMovId'
    stringSignalName2=targetArm2+'_executedMovId'

    def waitForMovementExecuted1(id):
        while client.executedMovId1!=id:
            client.simxSpinOnce()

    def waitForMovementExecuted2(id):
        while client.executedMovId2!=id:
            client.simxSpinOnce()

    def executedMovId1_callback(msg):
        if type(msg[1])==bytes:
            msg[1]=msg[1].decode('ascii') # python2/python3 differences
        client.executedMovId1=msg[1]

    def executedMovId2_callback(msg):
        if type(msg[1])==bytes:
            msg[1]=msg[1].decode('ascii') # python2/python3 differences
        client.executedMovId2=msg[1]

    # Subscribe to stringSignalName1 and stringSignalName2 string signals:
    client.simxGetStringSignal(stringSignalName1,client.simxDefaultSubscriber(executedMovId1_callback));
    client.simxGetStringSignal(stringSignalName2,client.simxDefaultSubscriber(executedMovId2_callback));

    # Set-up some movement variables:
    mVel=100*math.pi/180
    mAccel=150*math.pi/180
    maxVel=[mVel,mVel,mVel,mVel,mVel,mVel]
    maxAccel=[mAccel,mAccel,mAccel,mAccel,mAccel,mAccel]
    targetVel=[0,0,0,0,0,0]

    # Start simulation:
    client.simxStartSimulation(client.simxServiceCall())

    # Wait until ready:
    waitForMovementExecuted1('ready') 
    waitForMovementExecuted2('ready') 

    # Send first movement sequence:
    targetConfig=[90*math.pi/180,90*math.pi/180,-90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180]
    movementData={"id":"movSeq1","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm1,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())
    client.simxCallScriptFunction('movementDataFunction@'+targetArm2,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())

    # Execute the first movement sequence:
    client.simxCallScriptFunction('executeMovement@'+targetArm1,'sim.scripttype_childscript','movSeq1',client.simxDefaultPublisher())
    client.simxCallScriptFunction('executeMovement@'+targetArm2,'sim.scripttype_childscript','movSeq1',client.simxDefaultPublisher())
    
    # Wait until above movement sequences finished executing:
    waitForMovementExecuted1('movSeq1') 
    waitForMovementExecuted2('movSeq1') 

    # Send second and third movement sequence:
    targetConfig=[-90*math.pi/180,45*math.pi/180,90*math.pi/180,135*math.pi/180,90*math.pi/180,90*math.pi/180]
    targetVel=[-60*math.pi/180,-20*math.pi/180,0,0,0,0]
    movementData={"id":"movSeq2","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm1,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())
    client.simxCallScriptFunction('movementDataFunction@'+targetArm2,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())
    targetConfig=[0,0,0,0,0,0]
    targetVel=[0,0,0,0,0,0]
    movementData={"id":"movSeq3","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm1,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())
    client.simxCallScriptFunction('movementDataFunction@'+targetArm2,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())

    # Execute second and third movement sequences, where third one should execute immediately after the second one:
    client.simxCallScriptFunction('executeMovement@'+targetArm1,'sim.scripttype_childscript','movSeq2',client.simxDefaultPublisher())
    client.simxCallScriptFunction('executeMovement@'+targetArm2,'sim.scripttype_childscript','movSeq2',client.simxDefaultPublisher())
    client.simxCallScriptFunction('executeMovement@'+targetArm1,'sim.scripttype_childscript','movSeq3',client.simxDefaultPublisher())
    client.simxCallScriptFunction('executeMovement@'+targetArm2,'sim.scripttype_childscript','movSeq3',client.simxDefaultPublisher())

    # Wait until above 2 movement sequences finished executing:
    waitForMovementExecuted1('movSeq3') 
    waitForMovementExecuted2('movSeq3') 
    client.simxStopSimulation(client.simxServiceCall())
