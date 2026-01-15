// Make sure to have CoppeliaSim running, with followig scene loaded:
//
// scenes/blueZeroBasedRemoteApiDemo.ttt
//
// Do not launch simulation, and make sure that the B0 resolver
// is running. Then run "simpleTest"
//
// The client side (i.e. "simpleTest") depends on:
//
// b0 (shared library), which depends on:
// boost_chrono (shared library)
// boost_system (shared library)
// boost_thread (shared library)
// libzmq (shared library)


#include "b0RemoteApi.h"

b0RemoteApi* cl=NULL;

void callb(std::vector<msgpack::object>* msg)
{
    b0RemoteApi::print(msg);
}

int main(int argc,char* argv[])
{
    b0RemoteApi client("b0RemoteApi_c++Client","b0RemoteApi");
    cl=&client;

    std::string errorStr;
    std::vector<msgpack::object>* res;
    int colPurple[3]={255,0,255};
    int s1=b0RemoteApi::readInt(client.simxGetObjectHandle("shape1",client.simxServiceCall()),1);
    int s2=b0RemoteApi::readInt(client.simxGetObjectHandle("shape2",client.simxServiceCall()),1);
    int prox=b0RemoteApi::readInt(client.simxGetObjectHandle("prox",client.simxServiceCall()),1);
    int vis=b0RemoteApi::readInt(client.simxGetObjectHandle("vis",client.simxServiceCall()),1);
    int fs=b0RemoteApi::readInt(client.simxGetObjectHandle("fs",client.simxServiceCall()),1);
    int coll=b0RemoteApi::readInt(client.simxGetCollisionHandle("coll",client.simxServiceCall()),1);
    int dist=b0RemoteApi::readInt(client.simxGetDistanceHandle("dist",client.simxServiceCall()),1);
    /*
    client.simxAddStatusbarMessage("Hello",client.simxDefaultPublisher());
    int pos[2]={10,400};
    int size[2]={1024,100};
    float tCol[3]={1.0,1.0,0.0};
    float bCol[3]={0.0,0.0,0.0};
    int ch=b0RemoteApi::readInt(client.simxAuxiliaryConsoleOpen("theTitle",50,4,pos,size,tCol,bCol,client.simxServiceCall()),1);
    client.simxAuxiliaryConsolePrint(ch,"Hello World!!!\n",client.simxServiceCall());
    client.simxSleep(1000);
    client.simxAuxiliaryConsoleShow(ch,false,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxAuxiliaryConsoleShow(ch,true,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxAuxiliaryConsoleClose(ch,client.simxServiceCall());
    client.simxStartSimulation(client.simxServiceCall());
    client.simxStopSimulation(client.simxServiceCall());
    //*/
            /*
    float coords1[9]={0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0};
    res=client.simxAddDrawingObject_points(8,colPurple,coords1,3,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxRemoveDrawingObject(b0RemoteApi::readInt(res,1),client.simxServiceCall());
    int colRed[3]={255,0,0};
    res=client.simxAddDrawingObject_spheres(0.05,colRed,coords1,3,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxRemoveDrawingObject(b0RemoteApi::readInt(res,1),client.simxServiceCall());
    res=client.simxAddDrawingObject_cubes(0.05,colRed,coords1,3,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxRemoveDrawingObject(b0RemoteApi::readInt(res,1),client.simxServiceCall());
    int colGreen[3]={0,255,0};
    float coords2[18]={0.0,0.0,0.0,1.0,0.0,0.0, 1.0,0.0,0.0,0.0,0.0,1.0, 0.0,0.0,1.0,0.0,0.0,0.0};
    res=client.simxAddDrawingObject_segments(4,colGreen,coords2,3,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxRemoveDrawingObject(b0RemoteApi::readInt(res,1),client.simxServiceCall());
    int colOrange[3]={255,128,0};
    res=client.simxAddDrawingObject_triangles(colOrange,coords1,1,client.simxServiceCall());
    client.simxSleep(1000);
    client.simxRemoveDrawingObject(b0RemoteApi::readInt(res,1),client.simxServiceCall());
//*/
/*
    client.simxStartSimulation(client.simxServiceCall());
    b0RemoteApi::print(client.simxCheckCollision(s1,s2,client.simxServiceCall()));
    b0RemoteApi::print(client.simxCheckDistance(s1,s2,0,client.simxServiceCall()));
    b0RemoteApi::print(client.simxCheckProximitySensor(prox,s2,client.simxServiceCall()));
    b0RemoteApi::print(client.simxCheckVisionSensor(vis,s2,client.simxServiceCall()));
    b0RemoteApi::print(client.simxReadCollision(coll,client.simxServiceCall()));
    b0RemoteApi::print(client.simxReadDistance(dist,client.simxServiceCall()));
    b0RemoteApi::print(client.simxReadProximitySensor(prox,client.simxServiceCall()));
    b0RemoteApi::print(client.simxReadVisionSensor(vis,client.simxServiceCall()));
    b0RemoteApi::print(client.simxReadForceSensor(fs,client.simxServiceCall()));
    b0RemoteApi::print(client.simxBreakForceSensor(fs,client.simxServiceCall()));
    client.simxSleep(1000);
    client.simxStopSimulation(client.simxServiceCall());
            //*/
            /*
    client.simxSetFloatSignal("floatSignal",123.456f,client.simxServiceCall());
    client.simxSetIntegerSignal("integerSignal",59,client.simxServiceCall());
    client.simxSetStringSignal("stringSignal","Hello World",strlen("Hello world"),client.simxServiceCall());
    b0RemoteApi::print(client.simxGetFloatSignal("floatSignal",client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetIntegerSignal("integerSignal",client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetStringSignal("stringSignal",client.simxServiceCall()));
    client.simxSleep(1000);
    client.simxClearFloatSignal("floatSignal",client.simxServiceCall());
    client.simxClearIntegerSignal("integerSignal",client.simxServiceCall());
    client.simxClearStringSignal("stringSignal",client.simxServiceCall());
    client.simxSleep(1000);
    b0RemoteApi::print(client.simxGetFloatSignal("floatSignal",client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetIntegerSignal("integerSignal",client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetStringSignal("stringSignal",client.simxServiceCall()));

    client.simxCheckProximitySensor(prox,s2,client.simxDefaultSubscriber(callb));
    int startTime=client.simxGetTimeInMs();
    while (client.simxGetTimeInMs()<startTime+5000)
        client.simxSpinOnce();
//                */
    /*
    float pos[3]={0.0,0.0,0.2f};
    b0RemoteApi::print(client.simxSetObjectPosition(s1,-1,pos,client.simxServiceCall()));
    client.simxSleep(1000);
    b0RemoteApi::print(client.simxSetObjectOrientation(s1,-1,pos,client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetObjectOrientation(s1,-1,client.simxServiceCall()));
    client.simxSleep(1000);
    float quat[4]={0.0,0.0,0.2f,1.0f};
    b0RemoteApi::print(client.simxSetObjectQuaternion(s1,-1,quat,client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetObjectQuaternion(s1,-1,client.simxServiceCall()));
    client.simxSleep(1000);
    float pose[7]={0.1f,0.1f,0.0f,0.0f,0.0f,0.0f,1.0f};
    b0RemoteApi::print(client.simxSetObjectPose(s1,-1,pose,client.simxServiceCall()));
    b0RemoteApi::print(client.simxGetObjectPose(s1,-1,client.simxServiceCall()));
    client.simxSleep(1000);
    std::vector<msgpack::object>* matr=client.simxGetObjectMatrix(s1,-1,client.simxServiceCall());
    b0RemoteApi::print(matr);
    std::vector<float> m;
    b0RemoteApi::readFloatArray(matr,m,1);
    m[3]=0.0;
    m[7]=0.0;
    b0RemoteApi::print(client.simxSetObjectMatrix(s1,-1,&m[0],client.simxServiceCall()));
*/

    //*
    std::tuple<std::string,std::vector<int>,float> args1(
                "Hello World :)",
                std::vector<int>(colPurple,colPurple+3),
                42.123f);
    std::stringstream packedArgs1;
    msgpack::pack(packedArgs1,args1);
    b0RemoteApi::print(client.simxCallScriptFunction("myFunction@DefaultCamera","sim.scripttype_customizationscript",packedArgs1.str().c_str(),packedArgs1.str().size(),client.simxServiceCall()));
    std::tuple<std::string> args2("Hello World :)");
    std::stringstream packedArgs2;
    msgpack::pack(packedArgs2,args2);
    b0RemoteApi::print(client.simxCallScriptFunction("myFunction@DefaultCamera","sim.scripttype_customizationscript",packedArgs2.str().c_str(),packedArgs2.str().size(),client.simxServiceCall()));
    std::string msgPackNil;
    msgPackNil+=char(-64); // nil
    b0RemoteApi::print(client.simxCallScriptFunction("myFunction@DefaultCamera","sim.scripttype_customizationscript",msgPackNil.c_str(),msgPackNil.size(),client.simxServiceCall()));
//*/

    std::cout << "Ended!" << std::endl;
    return(0);
}

