# Make sure to have CoppeliaSim running, with followig scene loaded:
#
# scenes/synchronousImageTransmissionViaRemoteApi.ttt
#
# Do not launch simulation, and make sure that the B0 resolver
# is running. Then run this script
#
# The client side (i.e. this script) depends on:
#
# b0RemoteApi (Python script), which depends several libraries present
# in the CoppeliaSim folder

import b0RemoteApi
import time

with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi') as client:    
    client.doNextStep=True
    client.runInSynchronousMode=True

    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step done. Simulation time: ',simTime);
        client.doNextStep=True
        
    def imageCallback(msg):
        print('Received image.',msg[1])
        client.simxSetVisionSensorImage(passiveVisionSensorHandle[1],False,msg[2],client.simxDefaultPublisher())
        
    def stepSimulation():
        if client.runInSynchronousMode:
            while not client.doNextStep:
                client.simxSpinOnce()
            client.doNextStep=False
            client.simxSynchronousTrigger()
        else:
            client.simxSpinOnce()
    
    client.simxAddStatusbarMessage('Hello world!',client.simxDefaultPublisher())
    visionSensorHandle=client.simxGetObjectHandle('VisionSensor',client.simxServiceCall())
    passiveVisionSensorHandle=client.simxGetObjectHandle('PassiveVisionSensor',client.simxServiceCall())

    if client.runInSynchronousMode:
        client.simxSynchronous(True)
    
    #dedicatedSub=client.simxCreateSubscriber(imageCallback,1,True)
    #client.simxGetVisionSensorImage(visionSensorHandle[1],False,dedicatedSub)
    client.simxGetVisionSensorImage(visionSensorHandle[1],False,client.simxDefaultSubscriber(imageCallback))

    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone));
    client.simxStartSimulation(client.simxDefaultPublisher())
    
    startTime=time.time()
    while time.time()<startTime+5:
        stepSimulation()
        
    client.simxStopSimulation(client.simxDefaultPublisher())
