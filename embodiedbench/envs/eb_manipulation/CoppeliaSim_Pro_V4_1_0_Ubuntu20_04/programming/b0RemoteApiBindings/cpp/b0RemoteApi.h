// -------------------------------------------------------
// Add your custom functions at the bottom of the file
// and the server counterpart to lua/b0RemoteApiServer.lua
// -------------------------------------------------------

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "msgpack.hpp"
#include <boost/function.hpp>
#include <boost/bind.hpp>
extern "C" {
    #include <c.h>
}
#ifndef _WIN32
    #define __cdecl
#endif

typedef boost::function<void(std::vector<msgpack::object>*)> CB_FUNC;

struct SHandleAndCb
{
    b0_subscriber* handle;
    bool dropMessages;
    CB_FUNC cb;
};


class b0RemoteApi
{
public:
    b0RemoteApi(const char* nodeName="b0RemoteApi_c++Client",const char* channelName="b0RemoteApi",int inactivityToleranceInSec=60,bool setupSubscribersAsynchronously=false,int timeout=3);
    virtual ~b0RemoteApi();

    const char* simxServiceCall();
    const char* simxDefaultPublisher();
    const char* simxDefaultSubscriber(CB_FUNC cb,int publishInterval=1);
    const char* simxCreatePublisher(bool dropMessages=false);
    const char* simxCreateSubscriber(CB_FUNC cb,int publishInterval=1,bool dropMessages=false);
    void simxRemoveSubscriber(const char* topic);
    void simxRemovePublisher(const char* topic);

    long simxGetTimeInMs();
    void simxSleep(int durationInMs);
    void simxSpin();
    void simxSpinOnce();

    static void print(const std::vector<msgpack::object>* msg);
    static bool hasValue(const std::vector<msgpack::object>* msg);

    static bool readBool(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static int readInt(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static float readFloat(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static double readDouble(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static std::string readString(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static std::string readByteArray(std::vector<msgpack::object>* msg,int pos,bool* success=nullptr);
    static bool readIntArray(std::vector<msgpack::object>* msg,std::vector<int>& array,int pos);
    static bool readFloatArray(std::vector<msgpack::object>* msg,std::vector<float>& array,int pos);
    static bool readDoubleArray(std::vector<msgpack::object>* msg,std::vector<double>& array,int pos);
    static bool readStringArray(std::vector<msgpack::object>* msg,std::vector<std::string>& array,int pos);

protected:
    std::vector<msgpack::object>* _handleFunction(const char* funcName,const std::string& packedArgs,const char* topic);
    void _handleReceivedMessage(const std::string packedData);
    void _pingCallback(std::vector<msgpack::object>* msg);

    std::string _serviceCallTopic;
    std::string _defaultPublisherTopic;
    std::string _defaultSubscriberTopic;
    std::vector<std::string> _allTopics;
    int _nextDefaultSubscriberHandle;
    int _nextDedicatedPublisherHandle;
    int _nextDedicatedSubscriberHandle;
    bool _pongReceived;
    bool _setupSubscribersAsynchronously;
    msgpack::unpacked _tmpUnpackedMsg;
    std::vector<msgpack::object> _tmpMsgPackObjects;
    std::string _channelName;
    b0_node* _node;
    std::string _clientId;
    b0_service_client* _serviceClient;
    b0_publisher* _defaultPublisher;
    b0_subscriber* _defaultSubscriber;
    std::map<std::string,SHandleAndCb> _allSubscribers;
    std::map<std::string,b0_publisher*> _allDedicatedPublishers;

public:

    void simxSynchronous(bool enable);
    void simxSynchronousTrigger();
    void simxGetSimulationStepDone(const char* topic);
    void simxGetSimulationStepStarted(const char* topic);
    std::vector<msgpack::object>* simxCallScriptFunction(const char* funcAtObjName,int scriptType,const char* packedData,size_t packedDataSize,const char* topic);
    std::vector<msgpack::object>* simxCallScriptFunction(const char* funcAtObjName,const char* scriptType,const char* packedData,size_t packedDataSize,const char* topic);


    std::vector<msgpack::object>* simxGetObjectHandle(
        const char* objectName,
        const char* topic);
    std::vector<msgpack::object>* simxAddStatusbarMessage(
        const char* msg,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectPosition(
        int objectHandle,
        int relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectPosition(
        int objectHandle,
        const char* relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectOrientation(
        int objectHandle,
        int relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectOrientation(
        int objectHandle,
        const char* relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectQuaternion(
        int objectHandle,
        int relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectQuaternion(
        int objectHandle,
        const char* relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectPose(
        int objectHandle,
        int relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectPose(
        int objectHandle,
        const char* relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectMatrix(
        int objectHandle,
        int relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectMatrix(
        int objectHandle,
        const char* relObjHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectPosition(
        int objectHandle,
        int relObjHandle,
        const float* position,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectPosition(
        int objectHandle,
        const char* relObjHandle,
        const float* position,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectOrientation(
        int objectHandle,
        int relObjHandle,
        const float* euler,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectOrientation(
        int objectHandle,
        const char* relObjHandle,
        const float* euler,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectQuaternion(
        int objectHandle,
        int relObjHandle,
        const float* quat,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectQuaternion(
        int objectHandle,
        const char* relObjHandle,
        const float* quat,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectPose(
        int objectHandle,
        int relObjHandle,
        const float* pose,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectPose(
        int objectHandle,
        const char* relObjHandle,
        const float* pose,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectMatrix(
        int objectHandle,
        int relObjHandle,
        const float* matr,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectMatrix(
        int objectHandle,
        const char* relObjHandle,
        const float* matr,
        const char* topic);
    std::vector<msgpack::object>* simxClearFloatSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxClearIntegerSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxClearStringSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxSetFloatSignal(
        const char* sigName,
        float sigValue,
        const char* topic);
    std::vector<msgpack::object>* simxSetIntSignal(
        const char* sigName,
        int sigValue,
        const char* topic);
    std::vector<msgpack::object>* simxSetStringSignal(
        const char* sigName,
        const char* sigValue_data,size_t sigValue_charCnt,
        const char* topic);
    std::vector<msgpack::object>* simxGetFloatSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxGetIntSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxGetStringSignal(
        const char* sigName,
        const char* topic);
    std::vector<msgpack::object>* simxAuxiliaryConsoleClose(
        int consoleHandle,
        const char* topic);
    std::vector<msgpack::object>* simxAuxiliaryConsolePrint(
        int consoleHandle,
        const char* text,
        const char* topic);
    std::vector<msgpack::object>* simxAuxiliaryConsoleShow(
        int consoleHandle,
        bool showState,
        const char* topic);
    std::vector<msgpack::object>* simxAuxiliaryConsoleOpen(
        const char* title,
        int maxLines,
        int mode,
        const int* position,
        const int* size,
        const int* textColor,
        const int* backgroundColor,
        const char* topic);
    std::vector<msgpack::object>* simxStartSimulation(
        const char* topic);
    std::vector<msgpack::object>* simxStopSimulation(
        const char* topic);
    std::vector<msgpack::object>* simxPauseSimulation(
        const char* topic);
    std::vector<msgpack::object>* simxGetVisionSensorImage(
        int objectHandle,
        bool greyScale,
        const char* topic);
    std::vector<msgpack::object>* simxSetVisionSensorImage(
        int objectHandle,
        bool greyScale,
        const char* img_data,size_t img_charCnt,
        const char* topic);
    std::vector<msgpack::object>* simxGetVisionSensorDepthBuffer(
        int objectHandle,
        bool toMeters,
        bool asByteArray,
        const char* topic);
    std::vector<msgpack::object>* simxAddDrawingObject_points(
        int size,
        const int* color,
        const float* coords_data,size_t coords_floatCnt,
        const char* topic);
    std::vector<msgpack::object>* simxAddDrawingObject_spheres(
        float size,
        const int* color,
        const float* coords_data,size_t coords_floatCnt,
        const char* topic);
    std::vector<msgpack::object>* simxAddDrawingObject_cubes(
        float size,
        const int* color,
        const float* coords_data,size_t coords_floatCnt,
        const char* topic);
    std::vector<msgpack::object>* simxAddDrawingObject_segments(
        int lineSize,
        const int* color,
        const float* segments_data,size_t segments_floatCnt,
        const char* topic);
    std::vector<msgpack::object>* simxAddDrawingObject_triangles(
        const int* color,
        const float* triangles_data,size_t triangles_floatCnt,
        const char* topic);
    std::vector<msgpack::object>* simxRemoveDrawingObject(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxGetCollisionHandle(
        const char* nameOfObject,
        const char* topic);
    std::vector<msgpack::object>* simxGetDistanceHandle(
        const char* nameOfObject,
        const char* topic);
    std::vector<msgpack::object>* simxReadCollision(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxReadDistance(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxCheckCollision(
        int entity1,
        int entity2,
        const char* topic);
    std::vector<msgpack::object>* simxCheckCollision(
        int entity1,
        const char* entity2,
        const char* topic);
    std::vector<msgpack::object>* simxCheckDistance(
        int entity1,
        int entity2,
        float threshold,
        const char* topic);
    std::vector<msgpack::object>* simxCheckDistance(
        int entity1,
        const char* entity2,
        float threshold,
        const char* topic);
    std::vector<msgpack::object>* simxReadProximitySensor(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxCheckProximitySensor(
        int handle,
        int entity,
        const char* topic);
    std::vector<msgpack::object>* simxCheckProximitySensor(
        int handle,
        const char* entity,
        const char* topic);
    std::vector<msgpack::object>* simxReadForceSensor(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxBreakForceSensor(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxReadVisionSensor(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxCheckVisionSensor(
        int handle,
        int entity,
        const char* topic);
    std::vector<msgpack::object>* simxCheckVisionSensor(
        int handle,
        const char* entity,
        const char* topic);
    std::vector<msgpack::object>* simxCopyPasteObjects(
        const int* objectHandles_data,size_t objectHandles_intCnt,
        int options,
        const char* topic);
    std::vector<msgpack::object>* simxRemoveObjects(
        const int* objectHandles_data,size_t objectHandles_intCnt,
        int options,
        const char* topic);
    std::vector<msgpack::object>* simxCloseScene(
        const char* topic);
    std::vector<msgpack::object>* simxSetStringParameter(
        int paramId,
        const char* paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetStringParameter(
        const char* paramId,
        const char* paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetFloatParameter(
        int paramId,
        float paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetFloatParameter(
        const char* paramId,
        float paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetArrayParameter(
        int paramId,
        const float* paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetArrayParameter(
        const char* paramId,
        const float* paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetIntParameter(
        int paramId,
        int paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetIntParameter(
        const char* paramId,
        int paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetBoolParameter(
        int paramId,
        bool paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxSetBoolParameter(
        const char* paramId,
        bool paramVal,
        const char* topic);
    std::vector<msgpack::object>* simxGetStringParameter(
        int paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetStringParameter(
        const char* paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetFloatParameter(
        int paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetFloatParameter(
        const char* paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetArrayParameter(
        int paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetArrayParameter(
        const char* paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetIntParameter(
        int paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetIntParameter(
        const char* paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetBoolParameter(
        int paramId,
        const char* topic);
    std::vector<msgpack::object>* simxGetBoolParameter(
        const char* paramId,
        const char* topic);
    std::vector<msgpack::object>* simxDisplayDialog(
        const char* titleText,
        const char* mainText,
        int dialogType,
        const char* inputText,
        const char* topic);
    std::vector<msgpack::object>* simxDisplayDialog(
        const char* titleText,
        const char* mainText,
        const char* dialogType,
        const char* inputText,
        const char* topic);
    std::vector<msgpack::object>* simxGetDialogResult(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxGetDialogInput(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxEndDialog(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxExecuteScriptString(
        const char* code,
        const char* topic);
    std::vector<msgpack::object>* simxGetCollectionHandle(
        const char* collectionName,
        const char* topic);
    std::vector<msgpack::object>* simxGetJointForce(
        int jointHandle,
        const char* topic);
    std::vector<msgpack::object>* simxGetJointMaxForce(
        int jointHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetJointForce(
        int jointHandle,
        float forceOrTorque,
        const char* topic);
    std::vector<msgpack::object>* simxSetJointMaxForce(
        int jointHandle,
        float forceOrTorque,
        const char* topic);
    std::vector<msgpack::object>* simxGetJointPosition(
        int jointHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetJointPosition(
        int jointHandle,
        float position,
        const char* topic);
    std::vector<msgpack::object>* simxGetJointTargetPosition(
        int jointHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetJointTargetPosition(
        int jointHandle,
        float targetPos,
        const char* topic);
    std::vector<msgpack::object>* simxGetJointTargetVelocity(
        int jointHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetJointTargetVelocity(
        int jointHandle,
        float targetPos,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectChild(
        int objectHandle,
        int index,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectParent(
        int objectHandle,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectParent(
        int objectHandle,
        int parentHandle,
        bool assembly,
        bool keepInPlace,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectsInTree(
        int treeBaseHandle,
        const char* objectType,
        int options,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectsInTree(
        const char* treeBaseHandle,
        const char* objectType,
        int options,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectName(
        int objectHandle,
        bool altName,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectFloatParameter(
        int objectHandle,
        int parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectFloatParameter(
        int objectHandle,
        const char* parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectIntParameter(
        int objectHandle,
        int parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectIntParameter(
        int objectHandle,
        const char* parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectStringParameter(
        int objectHandle,
        int parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectStringParameter(
        int objectHandle,
        const char* parameterID,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectFloatParameter(
        int objectHandle,
        int parameterID,
        float parameter,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectFloatParameter(
        int objectHandle,
        const char* parameterID,
        float parameter,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectIntParameter(
        int objectHandle,
        int parameterID,
        int parameter,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectIntParameter(
        int objectHandle,
        const char* parameterID,
        int parameter,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectStringParameter(
        int objectHandle,
        int parameterID,
        const char* parameter,
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectStringParameter(
        int objectHandle,
        const char* parameterID,
        const char* parameter,
        const char* topic);
    std::vector<msgpack::object>* simxGetSimulationTime(
        const char* topic);
    std::vector<msgpack::object>* simxGetSimulationTimeStep(
        const char* topic);
    std::vector<msgpack::object>* simxGetServerTimeInMs(
        const char* topic);
    std::vector<msgpack::object>* simxGetSimulationState(
        const char* topic);
    std::vector<msgpack::object>* simxEvaluateToInt(
        const char* str,
        const char* topic);
    std::vector<msgpack::object>* simxEvaluateToStr(
        const char* str,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjects(
        int objectType,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjects(
        const char* objectType,
        const char* topic);
    std::vector<msgpack::object>* simxCreateDummy(
        float size,
        const int* color,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectSelection(
        const char* topic);
    std::vector<msgpack::object>* simxSetObjectSelection(
        const int* selection_data,size_t selection_intCnt,
        const char* topic);
    std::vector<msgpack::object>* simxGetObjectVelocity(
        int handle,
        const char* topic);
    std::vector<msgpack::object>* simxLoadModelFromFile(
        const char* filename,
        const char* topic);
    std::vector<msgpack::object>* simxLoadModelFromBuffer(
        const char* buffer_data,size_t buffer_charCnt,
        const char* topic);
    std::vector<msgpack::object>* simxLoadScene(
        const char* filename,
        const char* topic);



    // -----------------------------------------------------------
    // Add your custom functions here (and in the cpp file), or even better,
    // add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    // and generate this file again.
    // Then add the server part of your custom functions at the
    // beginning of file lua/b0RemoteApiServer.lua
    // -----------------------------------------------------------
    
};
