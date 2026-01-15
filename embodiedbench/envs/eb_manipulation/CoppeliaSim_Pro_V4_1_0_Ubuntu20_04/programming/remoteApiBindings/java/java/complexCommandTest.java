// This example illustrates how to execute complex commands from
// a remote API client. You can also use a similar construct for
// commands that are not directly supported by the remote API.
//
// Load the demo scene 'remoteApiCommandServerExample.ttt' in CoppeliaSim, then 
// start the simulation and run this program.
//
// IMPORTANT: for each successful call to simxStart, there
// should be a corresponding call to simxFinish at the end!

import coppelia.IntWA;
import coppelia.FloatWA;
import coppelia.StringWA;
import coppelia.remoteApi;

public class complexCommandTest
{
    public static void main(String[] args)
    {
        System.out.println("Program started");
        remoteApi sim = new remoteApi();
        sim.simxFinish(-1); // just in case, close all opened connections
        int clientID = sim.simxStart("127.0.0.1",19999,true,true,5000,5);
        if (clientID!=-1)
        {
            System.out.println("Connected to remote API server");   

            // 1. First send a command to display a specific message in a dialog box:
            StringWA inStrings=new StringWA(1);
            inStrings.getArray()[0]="Hello world!";
            StringWA outStrings=new StringWA(0);
            int result=sim.simxCallScriptFunction(clientID,"remoteApiCommandServer",sim.sim_scripttype_childscript,"displayText_function",null,null,inStrings,null,null,null,outStrings,null,sim.simx_opmode_blocking);
            if (result==sim.simx_return_ok)
                System.out.format("Returned message: %s\n",outStrings.getArray()[0]); // display the reply from CoppeliaSim (in this case, just a string)
            else
                System.out.format("Remote function call failed\n");

            // 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
            FloatWA inFloats=new FloatWA(3);
            inFloats.getArray()[0]=0.1f;
            inFloats.getArray()[1]=0.2f;
            inFloats.getArray()[2]=0.3f;
            inStrings=new StringWA(1);
            inStrings.getArray()[0]="MyDummyName";
            IntWA outInts=new IntWA(0);
            result=sim.simxCallScriptFunction(clientID,"remoteApiCommandServer",sim.sim_scripttype_childscript,"createDummy_function",null,inFloats,inStrings,null,outInts,null,null,null,sim.simx_opmode_blocking);
            if (result==sim.simx_return_ok)
                System.out.format("Dummy handle: %d\n",outInts.getArray()[0]); // display the reply from CoppeliaSim (in this case, the handle of the created dummy)
            else
                System.out.format("Remote function call failed\n");

            // 3. Now send a code string to execute some random functions:
            inStrings.getArray()[0]="local octreeHandle=simCreateOctree(0.5,0,1)\n" 
            + "simInsertVoxelsIntoOctree(octreeHandle,0,{0.1,0.1,0.1},{255,0,255})\n" 
            + "return 'done'";
            result=sim.simxCallScriptFunction(clientID,"remoteApiCommandServer",sim.sim_scripttype_childscript,"executeCode_function",null,null,inStrings,null,null,null,outStrings,null,sim.simx_opmode_blocking);
            if (result==sim.simx_return_ok)
                System.out.format("Code execution returned: %s\n",outStrings.getArray()[0]);
            else
                System.out.format("Remote function call failed\n");

            // Now close the connection to CoppeliaSim:   
            sim.simxFinish(clientID);
        }
        else
            System.out.println("Failed connecting to remote API server");
        System.out.println("Program ended");
    }
}
