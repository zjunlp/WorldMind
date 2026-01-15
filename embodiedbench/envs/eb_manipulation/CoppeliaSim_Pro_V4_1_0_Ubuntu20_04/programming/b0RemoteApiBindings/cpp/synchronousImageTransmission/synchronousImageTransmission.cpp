// Make sure to have CoppeliaSim running, with followig scene loaded:
//
// scenes/synchronousImageTransmissionViaRemoteApi.ttt
//
// Do not launch simulation, and make sure that the B0 resolver
// is running. Then run this program
//
// The client side (i.e. this program) depends on:
//
// b0 (shared library), which depends on several other libraries (libzmq and several boost libraries)

 
#include "b0RemoteApi.h"

bool doNextStep=true;
bool runInSynchronousMode=true;
int sens1,sens2;
b0RemoteApi* cl=NULL;

void simulationStepStarted_CB(std::vector<msgpack::object>* msg)
{
    float simTime=0.0;
    std::map<std::string,msgpack::object> data=msg->at(1).as<std::map<std::string,msgpack::object>>();
    std::map<std::string,msgpack::object>::iterator it=data.find("simulationTime");
    if (it!=data.end())
        simTime=it->second.as<float>();
    std::cout << "Simulation step started. Simulation time: " << simTime << std::endl;
}

void simulationStepDone_CB(std::vector<msgpack::object>* msg)
{
    float simTime=0.0;
    std::map<std::string,msgpack::object> data=msg->at(1).as<std::map<std::string,msgpack::object>>();
    std::map<std::string,msgpack::object>::iterator it=data.find("simulationTime");
    if (it!=data.end())
        simTime=it->second.as<float>();
    std::cout << "Simulation step done. Simulation time: " << simTime << std::endl;
    doNextStep=true;
}

void image_CB(std::vector<msgpack::object>* msg)
{
    std::cout << "Received image." << std::endl;
    std::string img(b0RemoteApi::readByteArray(msg,2));
    cl->simxSetVisionSensorImage(sens2,false,img.c_str(),img.size(),cl->simxDefaultPublisher());
}

void stepSimulation()
{
    if (runInSynchronousMode)
    {
        while (!doNextStep)
            cl->simxSpinOnce();
        doNextStep=false;
        cl->simxSynchronousTrigger();
    }
    else
        cl->simxSpinOnce();
}

int main(int argc,char* argv[])
{
    b0RemoteApi client("b0RemoteApi_c++Client","b0RemoteApi");
    cl=&client;

    client.simxAddStatusbarMessage("Hello world!",client.simxDefaultPublisher());
    std::vector<msgpack::object>* reply=client.simxGetObjectHandle("VisionSensor",client.simxServiceCall());
    sens1=b0RemoteApi::readInt(reply,1);
    reply=client.simxGetObjectHandle("PassiveVisionSensor",client.simxServiceCall());
    sens2=b0RemoteApi::readInt(reply,1);

    if (runInSynchronousMode)
        client.simxSynchronous(true);

    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted_CB));
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone_CB));
    client.simxGetVisionSensorImage(sens1,false,client.simxDefaultSubscriber(image_CB));
//    client.simxGetVisionSensorImage(sens1,false,client.simxCreateSubscriber(image_CB,1,true));
    client.simxStartSimulation(client.simxDefaultPublisher());

    unsigned int st=client.simxGetTimeInMs();
    while (client.simxGetTimeInMs()<st+5000)
        stepSimulation();

    client.simxStopSimulation(client.simxDefaultPublisher());
    std::cout << "Ended!" << std::endl;
    return(0);
}

