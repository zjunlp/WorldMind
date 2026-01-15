// Make sure to have the server side running in CoppeliaSim: 
// in a child script of a CoppeliaSim scene, add following command
// to be executed just once, at simulation start:
//
// simRemoteApi.start(19999)
//
// then start simulation, and run this program.
//
// IMPORTANT: for each successful call to simxStart, there
// should be a corresponding call to simxFinish at the end!

#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "extApi.h"
}

int main(int argc,char* argv[])
{
    int clientID=simxStart((simxChar*)"127.0.0.1",19999,true,true,2000,5);
    if (clientID!=-1)
    {
        printf("Connected to remote API server\n");

        // Now try to retrieve data in a blocking fashion (i.e. a service call):
        int objectCount;
        int* objectHandles;
        int ret=simxGetObjects(clientID,sim_handle_all,&objectCount,&objectHandles,simx_opmode_blocking);
        if (ret==simx_return_ok)
            printf("Number of objects in the scene: %d\n",objectCount);
        else
            printf("Remote API function call returned with error code: %d\n",ret);

        extApi_sleepMs(2000);

        // Now retrieve streaming data (i.e. in a non-blocking fashion):
        int startTime=extApi_getTimeInMs();
        int mouseX;
        simxGetIntegerParameter(clientID,sim_intparam_mouse_x,&mouseX,simx_opmode_streaming); // Initialize streaming
        while (extApi_getTimeDiffInMs(startTime) < 5000)
        {
            ret=simxGetIntegerParameter(clientID,sim_intparam_mouse_x,&mouseX,simx_opmode_buffer); // Try to retrieve the streamed data
            if (ret==simx_return_ok) // After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
                printf("Mouse position x: %d\n",mouseX); // Mouse position x is actualized when the cursor is over CoppeliaSim's window
        }
        
        // Now send some data to CoppeliaSim in a non-blocking fashion:
        simxAddStatusbarMessage(clientID,"Hello CoppeliaSim!",simx_opmode_oneshot);

        // Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        int pingTime;
        simxGetPingTime(clientID,&pingTime);

        // Now close the connection to CoppeliaSim:   
        simxFinish(clientID);
    }
    return(0);
}
