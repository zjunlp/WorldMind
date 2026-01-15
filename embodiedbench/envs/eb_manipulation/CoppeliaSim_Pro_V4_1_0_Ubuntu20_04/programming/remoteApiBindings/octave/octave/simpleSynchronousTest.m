% This small example illustrates how to use the remote API
% synchronous mode. The synchronous mode needs to be
% pre-enabled on the server side. You would do this by
% starting the server (e.g. in a child script) with:
%
% simRemoteApi.start(19999,1300,false,true)
%
% But in this example we try to connect on port
% 19997 where there should be a continuous remote API
% server service already running and pre-enabled for
% synchronous mode.
%
% IMPORTANT: for each successful call to simxStart, there
% should be a corresponding call to simxFinish at the end!

function simpleSynchronousTest()
    disp('Program started');
    sim=remApiSetup();
    simxFinish(-1); % just in case, close all opened connections
    clientID=simxStart('127.0.0.1',19997,true,true,5000,5);

    if (clientID>-1)
        disp('Connected to remote API server');

        % enable the synchronous mode on the client:
        simxSynchronous(clientID,true);

        % start the simulation:
        simxStartSimulation(clientID,sim.simx_opmode_blocking);

        % Now step a few times:
        for i=0:10
            disp('Press a key to step the simulation!');
            pause;
            simxSynchronousTrigger(clientID);
        end

        % stop the simulation:
        simxStopSimulation(clientID,sim.simx_opmode_blocking);

        % Now close the connection to CoppeliaSim:    
        simxFinish(clientID);
    else
        disp('Failed connecting to remote API server');
    end
    disp('Program ended');
end
