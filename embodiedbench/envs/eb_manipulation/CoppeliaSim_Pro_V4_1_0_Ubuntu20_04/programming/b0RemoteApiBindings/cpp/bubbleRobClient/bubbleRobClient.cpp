#include "b0RemoteApi.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static float simTime=0.0f;
static int sensorTrigger=0;
static long lastTimeReceived=0;
static b0RemoteApi* cl=nullptr;

void simulationStepStarted_CB(std::vector<msgpack::object>* msg)
{
    std::map<std::string,msgpack::object> data=msg->at(1).as<std::map<std::string,msgpack::object>>();
    std::map<std::string,msgpack::object>::iterator it=data.find("simulationTime");
    if (it!=data.end())
        simTime=it->second.as<float>();
}

void proxSensor_CB(std::vector<msgpack::object>* msg)
{
    sensorTrigger=b0RemoteApi::readInt(msg,1);
    lastTimeReceived=cl->simxGetTimeInMs();
    printf(".");
}

int main(int argc,char* argv[])
{
    int leftMotorHandle;
    int rightMotorHandle;
    int sensorHandle;

    if (argc>=4)
    {
        leftMotorHandle=atoi(argv[1]);
        rightMotorHandle=atoi(argv[2]);
        sensorHandle=atoi(argv[3]);
    }
    else
    {
        printf("Indicate following arguments: 'leftMotorHandle rightMotorHandle sensorHandle'!\n");
#ifdef _WIN32
        Sleep(5000);
#else
        usleep(5000000);
#endif
        return 0;
    }
    b0RemoteApi client("b0RemoteApi_c++Client","b0RemoteApi");
    cl=&client;

    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted_CB));
    client.simxReadProximitySensor(sensorHandle,client.simxDefaultSubscriber(proxSensor_CB,0));

    float driveBackStartTime=-99.0f;
    float motorSpeeds[2];
    lastTimeReceived=client.simxGetTimeInMs();

    while (client.simxGetTimeInMs()-lastTimeReceived<2000)
    {
        if (simTime-driveBackStartTime<3.0f)
        { // driving backwards while slightly turning:
            motorSpeeds[0]=-3.1415f*0.5f;
            motorSpeeds[1]=-3.1415f*0.25f;
        }
        else
        { // going forward:
            motorSpeeds[0]=3.1415f;
            motorSpeeds[1]=3.1415f;
            if (sensorTrigger>0)
                driveBackStartTime=simTime; // We detected something, and start the backward mode
        }
        client.simxSetJointTargetVelocity(leftMotorHandle,motorSpeeds[0],client.simxDefaultPublisher());
        client.simxSetJointTargetVelocity(rightMotorHandle,motorSpeeds[1],client.simxDefaultPublisher());
        client.simxSpinOnce();
        client.simxSleep(50);
    }
    std::cout << "Ended!" << std::endl;
    return(0);
}

