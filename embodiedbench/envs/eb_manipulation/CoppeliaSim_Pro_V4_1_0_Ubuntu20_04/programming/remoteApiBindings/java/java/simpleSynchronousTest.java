// This small example illustrates how to use the remote API
// synchronous mode. The synchronous mode needs to be
// pre-enabled on the server side. You would do this by
// starting the server (e.g. in a child script) with:
//
// simRemoteApi.start(19999,1300,false,true)
//
// But in this example we try to connect on port
// 19997 where there should be a continuous remote API
// server service already running and pre-enabled for
// synchronous mode.
//
// IMPORTANT: for each successful call to simxStart, there
// should be a corresponding call to simxFinish at the end!

import coppelia.IntW;
import coppelia.IntWA;
import coppelia.remoteApi;

public class simpleSynchronousTest
{
    public static void main(String[] args)
    {
        System.out.println("Program started");
        remoteApi sim = new remoteApi();
        sim.simxFinish(-1); // just in case, close all opened connections
        int clientID = sim.simxStart("127.0.0.1",19997,true,true,5000,5);
        if (clientID!=-1)
        {
            System.out.println("Connected to remote API server");   

            // enable the synchronous mode on the client:
            sim.simxSynchronous(clientID,true);

            // start the simulation:
            sim.simxStartSimulation(clientID,sim.simx_opmode_blocking);

            // Now step a few times:
            for (int i=0;i<10;i++)
            {
                System.out.println("Press enter to step the simulation!");
                String input=System.console().readLine();
                sim.simxSynchronousTrigger(clientID);
            }

            // stop the simulation:
            sim.simxStopSimulation(clientID,sim.simx_opmode_blocking);

            // Now close the connection to CoppeliaSim:   
            sim.simxFinish(clientID);
        }
        else
            System.out.println("Failed connecting to remote API server");
        System.out.println("Program ended");
    }
}
            
