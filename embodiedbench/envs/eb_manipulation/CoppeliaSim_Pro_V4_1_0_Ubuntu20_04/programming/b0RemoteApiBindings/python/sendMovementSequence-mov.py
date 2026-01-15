import b0RemoteApi
import math

with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi',60) as client:    
    client.executedMovId='notReady'

    targetArm='threadedBlueArm'
    #targetArm='nonThreadedRedArm'

    stringSignalName=targetArm+'_executedMovId'

    def waitForMovementExecuted(id):
        while client.executedMovId!=id:
            client.simxSpinOnce()

    def executedMovId_callback(msg):
        if type(msg[1])==bytes:
            msg[1]=msg[1].decode('ascii') # python2/python3 differences
        client.executedMovId=msg[1]

    # Subscribe to stringSignalName string signal:
    client.simxGetStringSignal(stringSignalName,client.simxDefaultSubscriber(executedMovId_callback));

    # Set-up some movement variables:
    mVel=100*math.pi/180
    mAccel=150*math.pi/180
    maxVel=[mVel,mVel,mVel,mVel,mVel,mVel]
    maxAccel=[mAccel,mAccel,mAccel,mAccel,mAccel,mAccel]
    targetVel=[0,0,0,0,0,0]

    # Start simulation:
    client.simxStartSimulation(client.simxServiceCall())

    # Wait until ready:
    waitForMovementExecuted('ready') 

    # Send first movement sequence:
    targetConfig=[90*math.pi/180,90*math.pi/180,-90*math.pi/180,90*math.pi/180,90*math.pi/180,90*math.pi/180]
    movementData={"id":"movSeq1","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())

    # Execute first movement sequence:
    client.simxCallScriptFunction('executeMovement@'+targetArm,'sim.scripttype_childscript','movSeq1',client.simxDefaultPublisher())
    
    # Wait until above movement sequence finished executing:
    waitForMovementExecuted('movSeq1')

    # Send second and third movement sequence, where third one should execute immediately after the second one:
    targetConfig=[-90*math.pi/180,45*math.pi/180,90*math.pi/180,135*math.pi/180,90*math.pi/180,90*math.pi/180]
    targetVel=[-60*math.pi/180,-20*math.pi/180,0,0,0,0]
    movementData={"id":"movSeq2","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())
    targetConfig=[0,0,0,0,0,0]
    targetVel=[0,0,0,0,0,0]
    movementData={"id":"movSeq3","type":"mov","targetConfig":targetConfig,"targetVel":targetVel,"maxVel":maxVel,"maxAccel":maxAccel}
    client.simxCallScriptFunction('movementDataFunction@'+targetArm,'sim.scripttype_childscript',movementData,client.simxDefaultPublisher())

    # Execute second and third movement sequences:
    client.simxCallScriptFunction('executeMovement@'+targetArm,'sim.scripttype_childscript','movSeq2',client.simxDefaultPublisher())
    client.simxCallScriptFunction('executeMovement@'+targetArm,'sim.scripttype_childscript','movSeq3',client.simxDefaultPublisher())
    
    # Wait until above 2 movement sequences finished executing:
    waitForMovementExecuted('movSeq3')
    client.simxStopSimulation(client.simxServiceCall())
