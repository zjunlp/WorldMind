// This example illustrates how to execute complex commands from
// a remote API client. You can also use a similar construct for
// commands that are not directly supported by the remote API.
//
// Load the demo scene 'remoteApiCommandServerExample.ttt' in CoppeliaSim, then 
// start the simulation and run this program.
//
// IMPORTANT: for each successful call to simxStart, there
// should be a corresponding call to simxFinish at the end!

#include <stdio.h>
#include <stdlib.h>
#include <string>

extern "C" {
    #include "extApi.h"
}

int main(int argc,char* argv[])
{
    int clientID=simxStart((simxChar*)"127.0.0.1",19999,true,true,2000,5);
    if (clientID!=-1)
    {
        printf("Connected to remote API server\n");

        // 1. First send a command to display a specific message in a dialog box:
        int retStringCnt;
        char* retStrings;
        int result=simxCallScriptFunction(clientID,"remoteApiCommandServer",sim_scripttype_childscript,"displayText_function",0,NULL,0,NULL,1,"Hello world!",0,NULL,NULL,NULL,NULL,NULL,&retStringCnt,&retStrings,NULL,NULL,simx_opmode_blocking);
        if (result==simx_return_ok)
            printf("Returned message: %s\n",retStrings); // display the reply from CoppeliaSim (in this case, just a string)
        else
            printf("Remote function call failed\n");

        // 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
        float coords[3]={0.1f,0.2f,0.3f};
        int retIntCnt;
        int* retInts;
        result=simxCallScriptFunction(clientID,"remoteApiCommandServer",sim_scripttype_childscript,"createDummy_function",0,NULL,3,coords,1,"MyDummyName",0,NULL,&retIntCnt,&retInts,NULL,NULL,NULL,NULL,NULL,NULL,simx_opmode_blocking);
        if (result==simx_return_ok)
            printf("Dummy handle: %i\n",retInts[0]); // display the reply from CoppeliaSim (in this case, the handle of the created dummy)
        else
            printf("Remote function call failed\n");

        // 3. Now send a code string to execute some random functions:
        char* code="local octreeHandle=simCreateOctree(0.5,0,1)\n" \
            "simInsertVoxelsIntoOctree(octreeHandle,0,{0.1,0.1,0.1},{255,0,255})\n" \
            "return 'done'";
        result=simxCallScriptFunction(clientID,"remoteApiCommandServer",sim_scripttype_childscript,"executeCode_function",0,NULL,0,NULL,1,code,0,NULL,NULL,NULL,NULL,NULL,&retStringCnt,&retStrings,NULL,NULL,simx_opmode_blocking);
        if (result==simx_return_ok)
            printf("Code execution returned: %s\n",retStrings);
        else
            printf("Remote function call failed\n");

        extApi_sleepMs(4000);

        // Now close the connection to CoppeliaSim:   
        simxFinish(clientID);
    }
    return(0);
}