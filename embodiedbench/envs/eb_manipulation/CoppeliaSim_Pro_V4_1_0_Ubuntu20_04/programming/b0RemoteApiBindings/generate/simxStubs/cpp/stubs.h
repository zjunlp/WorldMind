#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
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


#py for cmd in plugin.commands:
#py if cmd.generic and cmd.generateCode:
#py loopCnt=1
#py for p in cmd.params:
#py if p.ctype()=='int_eval':
#py loopCnt=2
#py endif
#py endfor
#py for k in range(loopCnt):
    std::vector<msgpack::object>* `cmd.name`(
#py theStringToWrite=''
#py itemCnt=len(cmd.params)
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='        int '+p.name
#py else:
#py theStringToWrite+='        const char* '+p.name
#py endif
#py elif p.htype()=='byte[]':
#py theStringToWrite+='        const char* '+p.name+'_data,size_t '+p.name+'_charCnt'
#py elif p.htype()=='int[]':
#py theStringToWrite+='        const int* '+p.name+'_data,size_t '+p.name+'_intCnt'
#py elif 'int[' in p.htype():
#py theStringToWrite+='        const int* '+p.name
#py elif p.htype()=='float[]':
#py theStringToWrite+='        const float* '+p.name+'_data,size_t '+p.name+'_floatCnt'
#py elif 'float[' in p.htype():
#py theStringToWrite+='        const float* '+p.name
#py elif p.htype()=='double[]':
#py theStringToWrite+='        const double* '+p.name+'_data,size_t '+p.name+'_doubleCnt'
#py elif 'double[' in p.htype():
#py theStringToWrite+='        const double* '+p.name
#py else:
#py theStringToWrite+='        '+p.htype()+' '+p.name
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=',\n'
#py endif
#py endfor
#py theStringToWrite+=');'
`theStringToWrite`
#py endfor
#py endif
#py endfor



    // -----------------------------------------------------------
    // Add your custom functions here (and in the cpp file), or even better,
    // add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    // and generate this file again.
    // Then add the server part of your custom functions at the
    // beginning of file lua/b0RemoteApiServer.lua
    // -----------------------------------------------------------
    
};
