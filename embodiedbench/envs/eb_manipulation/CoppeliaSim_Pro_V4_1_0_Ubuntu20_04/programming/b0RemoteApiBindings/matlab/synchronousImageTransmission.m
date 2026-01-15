% Make sure to have CoppeliaSim running, with followig scene loaded:
%
% scenes/synchronousImageTransmissionViaRemoteApi.ttt
%
% Do not launch simulation, and make sure that the B0 resolver
% is running. Then run this script
%
% The client side (i.e. this script) depends on:
%
% b0RemoteApi (Matlab script), which depends on:
% msgpack-matlab (Matlab scripts)
% b0RemoteApiProto (Matlab script), which depends on:
% b0 (shared library), which depends on:
% boost_chrono (shared library)
% boost_system (shared library)
% boost_thread (shared library)
% libzmq (shared library)

function synchronousImageTransmission()
    doNextStep=true;
    runInSynchronousMode=true;
    
    function simulationStepStarted_CB(data)
        data=data{2};
        simTime=data('simulationTime');
        disp(strcat('Simulation step started. Simulation time: ',num2str(simTime)));
    end
    function simulationStepDone_CB(data)
        data=data{2};
        simTime=data('simulationTime');
        disp(strcat('Simulation step done. Simulation time: ',num2str(simTime)));
        doNextStep=true;
    end
    
    function image_CB(data)
        % disp(jsonencode(data)); % in order to explicitely display all data
        img=data{3};
        disp('Received image.');
        res=client.simxSetVisionSensorImage(passiveVisionSensorHandle{2},false,img,client.simxDefaultPublisher());
    end
    
    function stepSimulation()
        if runInSynchronousMode
            while doNextStep==false
                client.simxSpinOnce();
            end
            doNextStep=false;
            client.simxSynchronousTrigger();
        else
            client.simxSpinOnce();
        end
    end
    
    disp('Program started');
    try
        client=b0RemoteApi('b0RemoteApi_matlabClient','b0RemoteApi');
        client.simxAddStatusbarMessage('Hello world!',client.simxDefaultPublisher());
        visionSensorHandle=client.simxGetObjectHandle('VisionSensor',client.simxServiceCall());
        passiveVisionSensorHandle=client.simxGetObjectHandle('PassiveVisionSensor',client.simxServiceCall());
        
        if runInSynchronousMode
            client.simxSynchronous(true);
        end
        
        client.simxGetVisionSensorImage(visionSensorHandle{2},false,client.simxDefaultSubscriber(@image_CB));
        
        client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(@simulationStepStarted_CB));
        client.simxGetSimulationStepDone(client.simxDefaultSubscriber(@simulationStepDone_CB));
        
        client.simxStartSimulation(client.simxServiceCall());
        tic;
        while toc<5
            stepSimulation();
        end
        client.simxStopSimulation(client.simxDefaultPublisher());
        client.delete();
    catch me
        client.delete();
        rethrow(me);
    end
    disp('Program ended');
end
