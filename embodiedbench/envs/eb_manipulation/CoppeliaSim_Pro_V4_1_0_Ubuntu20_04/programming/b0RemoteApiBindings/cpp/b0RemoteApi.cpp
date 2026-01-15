// -------------------------------------------------------
// Add your custom functions at the bottom of the file
// and the server counterpart to lua/b0RemoteApiServer.lua
// -------------------------------------------------------

#include "b0RemoteApi.h"

b0RemoteApi::b0RemoteApi(const char* nodeName,const char* channelName,int inactivityToleranceInSec,bool setupSubscribersAsynchronously,int timeout)
{
    _channelName=channelName;
    _serviceCallTopic=_channelName+"SerX";
    _defaultPublisherTopic=_channelName+"SubX";
    _defaultSubscriberTopic=_channelName+"PubX";
    _allTopics.push_back(_serviceCallTopic);
    _allTopics.push_back(_defaultPublisherTopic);
    _allTopics.push_back(_defaultSubscriberTopic);
    _nextDefaultSubscriberHandle=2;
    _nextDedicatedPublisherHandle=500;
    _nextDedicatedSubscriberHandle=1000;

    if (b0_is_initialized()==0)
    {
        int arg1=1;
        const char* arg2="b0C";
        b0_init(&arg1,(char**)&arg2);
    }
    _node=b0_node_new(nodeName);
    srand((unsigned int)b0_node_hardware_time_usec(_node));
    const char* alp="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (size_t i=0;i<10;i++)
    {
        size_t p=size_t(61.9f*(float(rand())/RAND_MAX));
        _clientId+=alp[p];
    }
    _serviceClient=b0_service_client_new(_node,_serviceCallTopic.c_str());
    b0_service_client_set_option(_serviceClient,B0_SOCK_OPT_READTIMEOUT,timeout*1000);
    _defaultPublisher=b0_publisher_new(_node,_defaultPublisherTopic.c_str());
    _defaultSubscriber=b0_subscriber_new(_node,_defaultSubscriberTopic.c_str(),nullptr); // we will poll the socket
    std::cout << "\n  Running B0 Remote API client with channel name [" << channelName << "]" << std::endl;
    std::cout << "  make sure that: 1) the B0 resolver is running" << std::endl;
    std::cout << "                  2) CoppeliaSim is running the B0 Remote API server with the same channel name" << std::endl;
    std::cout << "  Initializing...\n" << std::endl;
    b0_node_init(_node);

    std::tuple<int> args(inactivityToleranceInSec);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    _handleFunction("inactivityTolerance",packedArgs.str(),_serviceCallTopic.c_str());
    _setupSubscribersAsynchronously=setupSubscribersAsynchronously;

    std::cout << "\n  Connected!\n" << std::endl;
}

b0RemoteApi::~b0RemoteApi()
{
    std::cout << "*************************************************************************************\n";
    std::cout << "** Leaving... if this is unexpected, you might have to adjust the timeout argument **\n";
    std::cout << "*************************************************************************************\n";
    _pongReceived=false;
    std::tuple<int> args1(0);
    std::stringstream packedArgs1;
    msgpack::pack(packedArgs1,args1);
    const char* pingTopic=simxDefaultSubscriber(boost::bind(&b0RemoteApi::_pingCallback,this,_1));
    _handleFunction("Ping",packedArgs1.str(),pingTopic);
    while (!_pongReceived)
        simxSpinOnce();

    std::tuple<std::string> args2(_clientId);
    std::stringstream packedArgs2;
    msgpack::pack(packedArgs2,args2);
    _handleFunction("DisconnectClient",packedArgs2.str(),_serviceCallTopic.c_str());

    for (std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.begin();it!=_allSubscribers.end();it++)
    {
        if (it->second.handle!=_defaultSubscriber)
        {
            b0_subscriber_cleanup(it->second.handle);
            b0_subscriber_delete(it->second.handle);
        }
    }

    for (std::map<std::string,b0_publisher*>::iterator it=_allDedicatedPublishers.begin();it!=_allDedicatedPublishers.end();it++)
    {
        b0_publisher_cleanup(it->second);
        b0_publisher_delete(it->second);
    }
    _tmpMsgPackObjects.clear();
    b0_node_cleanup(_node);

    b0_subscriber_delete(_defaultSubscriber);
    b0_publisher_delete(_defaultPublisher);
    b0_service_client_delete(_serviceClient);
    b0_node_delete(_node);
}

void b0RemoteApi::_pingCallback(std::vector<msgpack::object>* msg)
{
    _pongReceived=true;
}

void b0RemoteApi::simxSpin()
{
    while (true)
        simxSpinOnce();
}

void b0RemoteApi::simxSpinOnce()
{
    bool defaultSubscriberAlreadyProcessed=false;
    for (std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.begin();it!=_allSubscribers.end();it++)
    {
        std::string packedData;
        if ( (it->second.handle!=_defaultSubscriber)||(!defaultSubscriberAlreadyProcessed) )
        {
            defaultSubscriberAlreadyProcessed|=(it->second.handle==_defaultSubscriber);
            while (b0_subscriber_poll(it->second.handle,0))
            {
                packedData.clear();
                size_t s=0;
                char* data=(char*)b0_subscriber_read(it->second.handle,&s);
                if (data!=nullptr)
                {
                    packedData.assign(data,data+s);
                    b0_buffer_delete(data);
                }
                if (!it->second.dropMessages)
                    _handleReceivedMessage(packedData);
            }
            if ( it->second.dropMessages&&(packedData.size()>0) )
                _handleReceivedMessage(packedData);
        }
    }
}

void b0RemoteApi::_handleReceivedMessage(const std::string packedData)
{
    if (packedData.size()>0)
    {
        msgpack::unpacked msg;
        msgpack::unpack(msg,packedData.data(),packedData.size());
        msgpack::object obj(msg.get());
        if ( (obj.type==msgpack::type::ARRAY)&&(obj.via.array.size==2)&&( (obj.via.array.ptr[0].type==msgpack::type::STR)||(obj.via.array.ptr[0].type==msgpack::type::BIN) ) )
        {
            std::string topic(obj.via.array.ptr[0].as<std::string>());
            std::map<std::string,SHandleAndCb >::iterator it=_allSubscribers.find(topic);
            if (it!=_allSubscribers.end())
            {
                msgpack::object obj2=obj.via.array.ptr[1].as<msgpack::object>();
                if ( (obj2.type==msgpack::type::ARRAY)&&(obj2.via.array.ptr[0].type==msgpack::type::BOOLEAN) )
                {
                    std::vector<msgpack::object> vals;
                    obj2.convert(vals);
                    if (vals.size()<2)
                        vals.push_back(msgpack::object());
                    it->second.cb(&vals);
                }
            }
        }
    }
}

long b0RemoteApi::simxGetTimeInMs()
{
    return((long)b0_node_hardware_time_usec(_node)/1000);
}

void b0RemoteApi::simxSleep(int durationInMs)
{
#ifdef _WIN32
    Sleep(durationInMs);
#else
    usleep(durationInMs*1000);
#endif
}

const char* b0RemoteApi::simxDefaultPublisher()
{
    return(_defaultPublisherTopic.c_str());
}

const char* b0RemoteApi::simxCreatePublisher(bool dropMessages)
{
    std::string topic=_channelName+"Sub"+std::to_string(_nextDedicatedPublisherHandle++)+_clientId;
    _allTopics.push_back(topic);
    b0_publisher* pub=b0_publisher_new_ex(_node,topic.c_str(),false,true);
    //    pub->setConflate(true);
    b0_publisher_init(pub);
    _allDedicatedPublishers[topic]=pub;
    std::tuple<std::string,bool> args(topic,dropMessages);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    _handleFunction("createSubscriber",packedArgs.str(),_serviceCallTopic.c_str());
    return(_allTopics[_allTopics.size()-1].c_str());
}

const char* b0RemoteApi::simxDefaultSubscriber(CB_FUNC cb,int publishInterval)
{
    std::string topic=_channelName+"Pub"+std::to_string(_nextDefaultSubscriberHandle++)+_clientId;
    _allTopics.push_back(topic);
    SHandleAndCb dat;
    dat.handle=_defaultSubscriber;
    dat.cb=cb;
    dat.dropMessages=false;
    _allSubscribers[topic]=dat;
    std::tuple<std::string,int> args(topic,publishInterval);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    std::string channel=_serviceCallTopic;
    if (_setupSubscribersAsynchronously)
        channel=_defaultPublisherTopic;
    _handleFunction("setDefaultPublisherPubInterval",packedArgs.str(),channel.c_str());
    return(_allTopics[_allTopics.size()-1].c_str());
}

const char* b0RemoteApi::simxCreateSubscriber(CB_FUNC cb,int publishInterval,bool dropMessages)
{
    std::string topic=_channelName+"Pub"+std::to_string(_nextDedicatedSubscriberHandle++)+_clientId;
    _allTopics.push_back(topic);
    b0_subscriber* sub=b0_subscriber_new_ex(_node,topic.c_str(),nullptr,false,true);
    b0_subscriber_set_option(sub,B0_SOCK_OPT_CONFLATE,dropMessages);
    b0_subscriber_init(sub);
    SHandleAndCb dat;
    dat.handle=sub;
    dat.cb=cb;
    dat.dropMessages=dropMessages;
    _allSubscribers[topic]=dat;
    std::tuple<std::string,int> args(topic,publishInterval);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    std::string channel=_serviceCallTopic;
    if (_setupSubscribersAsynchronously)
        channel=_defaultPublisherTopic;
    _handleFunction("createPublisher",packedArgs.str(),channel.c_str());
    return(_allTopics[_allTopics.size()-1].c_str());
}

void b0RemoteApi::simxRemoveSubscriber(const char* topic)
{
    for (size_t i=0;i<_allTopics.size();i++)
    {
        if (_allTopics[i].compare(topic)==0)
        {
            _allTopics.erase(_allTopics.begin()+i);
            break;
        }
    }
    std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.find(topic);
    if (it!=_allSubscribers.end())
    {
        std::tuple<std::string> args(topic);
        std::stringstream packedArgs;
        msgpack::pack(packedArgs,args);
        std::string channel=_serviceCallTopic;
        if (_setupSubscribersAsynchronously)
            channel=_defaultPublisherTopic;
        if (it->second.handle==_defaultSubscriber)
            _handleFunction("stopDefaultPublisher",packedArgs.str(),channel.c_str());
        else
        {
            b0_subscriber_cleanup(it->second.handle);
            b0_subscriber_delete(it->second.handle);
            _handleFunction("stopPublisher",packedArgs.str(),channel.c_str());
        }
        _allSubscribers.erase(it);
    }
}

void b0RemoteApi::simxRemovePublisher(const char* topic)
{
    std::map<std::string,b0_publisher*>::iterator it=_allDedicatedPublishers.find(topic);
    if (it!=_allDedicatedPublishers.end())
    {
        for (size_t i=0;i<_allTopics.size();i++)
        {
            if (_allTopics[i].compare(topic)==0)
            {
                _allTopics.erase(_allTopics.begin()+i);
                break;
            }
        }
        b0_publisher_cleanup(it->second);
        b0_publisher_delete(it->second);
        std::tuple<std::string> args(topic);
        std::stringstream packedArgs;
        msgpack::pack(packedArgs,args);
        _handleFunction("stopSubscriber",packedArgs.str(),_serviceCallTopic.c_str());
        _allDedicatedPublishers.erase(it);
    }
}

const char* b0RemoteApi::simxServiceCall()
{
    return(_serviceCallTopic.c_str());
}

std::vector<msgpack::object>* b0RemoteApi::_handleFunction(const char* funcName,const std::string& packedArgs,const char* topic)
{
    _tmpMsgPackObjects.clear();

    if (topic==_serviceCallTopic)
    {
        std::tuple<std::string,std::string,std::string,int> header(funcName,_clientId,topic,0);
        std::stringstream packedHeader;
        msgpack::pack(packedHeader,header);
        std::string packedMsg;
        packedMsg+=char(-110); // array of 2
        packedMsg+=packedHeader.str();
        packedMsg+=packedArgs;
        std::string rep;
        size_t s=0;
        char* data=(char*)b0_service_client_call(_serviceClient,packedMsg.c_str(),packedMsg.size(),&s);
        if (data!=nullptr)
        {
            rep.assign(data,data+s);
            b0_buffer_delete(data);
        }
        msgpack::unpack(_tmpUnpackedMsg,rep.data(),rep.size());
        msgpack::object obj(_tmpUnpackedMsg.get());
        if ( (obj.type==msgpack::type::ARRAY)&&(obj.via.array.ptr[0].type==msgpack::type::BOOLEAN) )
        {
            obj.convert(_tmpMsgPackObjects);
            if (_tmpMsgPackObjects.size()<2)
                _tmpMsgPackObjects.push_back(msgpack::object());
            return(&_tmpMsgPackObjects);
        }
        return(nullptr);
    }
    else if (topic==_defaultPublisherTopic)
    {
        std::tuple<std::string,std::string,std::string,int> header(funcName,_clientId,topic,1);
        std::stringstream packedHeader;
        msgpack::pack(packedHeader,header);
        std::string packedMsg;
        packedMsg+=char(-110); // array of 2
        packedMsg+=packedHeader.str();
        packedMsg+=packedArgs;
        b0_publisher_publish(_defaultPublisher,packedMsg.c_str(),packedMsg.size());
        return(nullptr);
    }
    else
    {
        std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.find(topic);
        if (it!=_allSubscribers.end())
        {
            std::stringstream packedHeader;
            if (it->second.handle==_defaultSubscriber)
            {
                std::tuple<std::string,std::string,std::string,int> header(funcName,_clientId,topic,2);
                msgpack::pack(packedHeader,header);
            }
            else
            {
                std::tuple<std::string,std::string,std::string,int> header(funcName,_clientId,topic,4);
                msgpack::pack(packedHeader,header);
            }
            std::string packedMsg;
            packedMsg+=char(-110); // array of 2
            packedMsg+=packedHeader.str();
            packedMsg+=packedArgs;
            if (_setupSubscribersAsynchronously)
                b0_publisher_publish(_defaultPublisher,packedMsg.c_str(),packedMsg.size());
            else
            {
                std::string rep;
                size_t s=0;
                char* data=(char*)b0_service_client_call(_serviceClient,packedMsg.c_str(),packedMsg.size(),&s);
                if (data!=nullptr)
                {
                    rep.assign(data,data+s);
                    b0_buffer_delete(data);
                }
            }
            return(nullptr);
        }
        else
        {
            std::map<std::string,b0_publisher*>::iterator it=_allDedicatedPublishers.find(topic);
            if (it!=_allDedicatedPublishers.end())
            {
                std::stringstream packedHeader;
                std::tuple<std::string,std::string,std::string,int> header(funcName,_clientId,topic,3);
                msgpack::pack(packedHeader,header);
                std::string packedMsg;
                packedMsg+=char(-110); // array of 2
                packedMsg+=packedHeader.str();
                packedMsg+=packedArgs;
                b0_publisher_publish(it->second,packedMsg.c_str(),packedMsg.size());
                return(nullptr);
            }
        }
    }
    return(nullptr);
}

void b0RemoteApi::print(const std::vector<msgpack::object>* msg)
{
    if (msg->size()>0)
    {
        for (size_t i=0;i<msg->size();i++)
        {
            if (i>0)
                std::cout << ", ";
            std::cout << msg->at(i);
        }
        std::cout << std::endl;
    }
}

bool b0RemoteApi::hasValue(const std::vector<msgpack::object>* msg)
{
    return(msg->size()>0);
}

bool b0RemoteApi::readBool(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&(val->type==msgpack::type::BOOLEAN) )
    {
        if (success!=nullptr)
            success[0]=true;
        return(val->as<bool>());
    }
    if (success!=nullptr)
        success[0]=false;
    return(false);
}

int b0RemoteApi::readInt(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&( (val->type==msgpack::type::POSITIVE_INTEGER)||(val->type==msgpack::type::NEGATIVE_INTEGER)||(val->type==msgpack::type::FLOAT) ) )
    {
        if (success!=nullptr)
            success[0]=true;
        if (val->type==msgpack::type::FLOAT)
        {
            double v=val->as<double>();
            if (v<0.0)
                return((int)(v-0.5));
            return((int)(v+0.5));
        }
        return(val->as<int>());
    }
    if (success!=nullptr)
        success[0]=false;
    return(0);
}

float b0RemoteApi::readFloat(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    return((float)readDouble(msg,pos,success));
}

double b0RemoteApi::readDouble(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&( (val->type==msgpack::type::POSITIVE_INTEGER)||(val->type==msgpack::type::NEGATIVE_INTEGER)||(val->type==msgpack::type::FLOAT) ) )
    {
        if (success!=nullptr)
            success[0]=true;
        if (val->type==msgpack::type::FLOAT)
            return(val->as<double>());
        return((double)val->as<int>());
    }
    if (success!=nullptr)
        success[0]=false;
    return(0.0);
}

std::string b0RemoteApi::readString(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    return(readByteArray(msg,pos,success));
}

std::string b0RemoteApi::readByteArray(std::vector<msgpack::object>* msg,int pos,bool* success)
{
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&( (val->type==msgpack::type::STR)||(val->type==msgpack::type::BIN) ) )
    {
        if (success!=nullptr)
            success[0]=true;
        return(val->as<std::string>());
    }
    if (success!=nullptr)
        success[0]=false;
    return("");
}

bool b0RemoteApi::readIntArray(std::vector<msgpack::object>* msg,std::vector<int>& array,int pos)
{
    bool retVal=false;
    array.clear();
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&(val->type==msgpack::type::ARRAY) )
    {
        std::vector<msgpack::object> vals;
        val->convert(vals);
        for (size_t i=0;i<vals.size();i++)
        {
            if ( (vals[i].type==msgpack::type::POSITIVE_INTEGER)||(vals[i].type==msgpack::type::NEGATIVE_INTEGER)||(vals[i].type==msgpack::type::FLOAT) )
            {
                if (vals[i].type==msgpack::type::FLOAT)
                {
                    double v=vals[i].as<double>();
                    if (v<0.0)
                        array.push_back((int)(v-0.5));
                    else
                        array.push_back((int)(v+0.5));
                }
                else
                    array.push_back(vals[i].as<int>());
            }
            else
                array.push_back(0);
        }
        retVal=true;
    }
    return(retVal);
}

bool b0RemoteApi::readFloatArray(std::vector<msgpack::object>* msg,std::vector<float>& array,int pos)
{
    bool retVal=false;
    array.clear();
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&(val->type==msgpack::type::ARRAY) )
    {
        std::vector<msgpack::object> vals;
        val->convert(vals);
        for (size_t i=0;i<vals.size();i++)
        {
            if ( (vals[i].type==msgpack::type::POSITIVE_INTEGER)||(vals[i].type==msgpack::type::NEGATIVE_INTEGER)||(vals[i].type==msgpack::type::FLOAT) )
            {
                if (vals[i].type==msgpack::type::FLOAT)
                    array.push_back(vals[i].as<float>());
                else
                    array.push_back((float)vals[i].as<int>());
            }
            else
                array.push_back(0.0f);
        }
        retVal=true;
    }
    return(retVal);
}

bool b0RemoteApi::readDoubleArray(std::vector<msgpack::object>* msg,std::vector<double>& array,int pos)
{
    bool retVal=false;
    array.clear();
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&(val->type==msgpack::type::ARRAY) )
    {
        std::vector<msgpack::object> vals;
        val->convert(vals);
        for (size_t i=0;i<vals.size();i++)
        {
            if ( (vals[i].type==msgpack::type::POSITIVE_INTEGER)||(vals[i].type==msgpack::type::NEGATIVE_INTEGER)||(vals[i].type==msgpack::type::FLOAT) )
            {
                if (vals[i].type==msgpack::type::FLOAT)
                    array.push_back(vals[i].as<double>());
                else
                    array.push_back((double)vals[i].as<int>());
            }
            else
                array.push_back(0.0);
        }
        retVal=true;
    }
    return(retVal);
}

bool b0RemoteApi::readStringArray(std::vector<msgpack::object>* msg,std::vector<std::string>& array,int pos)
{
    bool retVal=false;
    array.clear();
    const msgpack::object* val=&msg->at(pos);
    if ( (val!=nullptr)&&(val->type==msgpack::type::ARRAY) )
    {
        std::vector<msgpack::object> vals;
        val->convert(vals);
        for (size_t i=0;i<vals.size();i++)
        {
            if ( (vals[i].type==msgpack::type::STR)||(vals[i].type==msgpack::type::BIN) )
                array.push_back(vals[i].as<std::string>());
            else
                array.push_back("");
        }
        retVal=true;
    }
    return(retVal);
}

void b0RemoteApi::simxSynchronous(bool enable)
{
    std::tuple<bool> args(enable);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    _handleFunction("Synchronous",packedArgs.str(),_serviceCallTopic.c_str());
}

void b0RemoteApi::simxSynchronousTrigger()
{
    std::tuple<int> args(0);
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    _handleFunction("SynchronousTrigger",packedArgs.str(),_defaultPublisherTopic.c_str());
}

void b0RemoteApi::simxGetSimulationStepDone(const char* topic)
{
    std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.find(topic);
    if (it!=_allSubscribers.end())
    {
        std::tuple<int> args(0);
        std::stringstream packedArgs;
        msgpack::pack(packedArgs,args);
        _handleFunction("GetSimulationStepDone",packedArgs.str(),topic);
    }
    else
        std::cout << "B0 Remote API error: invalid topic" << std::endl;
}

void b0RemoteApi::simxGetSimulationStepStarted(const char* topic)
{
    std::map<std::string,SHandleAndCb>::iterator it=_allSubscribers.find(topic);
    if (it!=_allSubscribers.end())
    {
        std::tuple<int> args(0);
        std::stringstream packedArgs;
        msgpack::pack(packedArgs,args);
        _handleFunction("GetSimulationStepStarted",packedArgs.str(),topic);
    }
    else
        std::cout << "B0 Remote API error: invalid topic" << std::endl;
}

std::vector<msgpack::object>* b0RemoteApi::simxCallScriptFunction(const char* funcAtObjName,int scriptType,const char* packedData,size_t packedDataSize,const char* topic)
{
    std::tuple<std::string,int,std::string> args(funcAtObjName,scriptType,std::string(packedData,packedData+packedDataSize));
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CallScriptFunction",packedArgs.str(),topic));
}

std::vector<msgpack::object>* b0RemoteApi::simxCallScriptFunction(const char* funcAtObjName,const char* scriptType,const char* packedData,size_t packedDataSize,const char* topic)
{
    std::tuple<std::string,std::string,std::string> args(funcAtObjName,scriptType,std::string(packedData,packedData+packedDataSize));
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CallScriptFunction",packedArgs.str(),topic));
}



std::vector<msgpack::object>* b0RemoteApi::simxGetObjectHandle(
    const char* objectName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        objectName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectHandle",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddStatusbarMessage(
    const char* msg,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        msg
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddStatusbarMessage",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectPosition(
    int objectHandle,
    int relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectPosition(
    int objectHandle,
    const char* relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectOrientation(
    int objectHandle,
    int relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectOrientation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectOrientation(
    int objectHandle,
    const char* relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectOrientation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectQuaternion(
    int objectHandle,
    int relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectQuaternion",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectQuaternion(
    int objectHandle,
    const char* relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectQuaternion",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectPose(
    int objectHandle,
    int relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectPose",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectPose(
    int objectHandle,
    const char* relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectPose",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectMatrix(
    int objectHandle,
    int relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectMatrix",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectMatrix(
    int objectHandle,
    const char* relObjHandle,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        relObjHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectMatrix",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectPosition(
    int objectHandle,
    int relObjHandle,
    const float* position,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(position,position+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectPosition(
    int objectHandle,
    const char* relObjHandle,
    const float* position,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(position,position+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectOrientation(
    int objectHandle,
    int relObjHandle,
    const float* euler,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(euler,euler+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectOrientation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectOrientation(
    int objectHandle,
    const char* relObjHandle,
    const float* euler,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(euler,euler+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectOrientation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectQuaternion(
    int objectHandle,
    int relObjHandle,
    const float* quat,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(quat,quat+4)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectQuaternion",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectQuaternion(
    int objectHandle,
    const char* relObjHandle,
    const float* quat,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(quat,quat+4)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectQuaternion",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectPose(
    int objectHandle,
    int relObjHandle,
    const float* pose,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(pose,pose+7)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectPose",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectPose(
    int objectHandle,
    const char* relObjHandle,
    const float* pose,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(pose,pose+7)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectPose",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectMatrix(
    int objectHandle,
    int relObjHandle,
    const float* matr,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(matr,matr+12)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectMatrix",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectMatrix(
    int objectHandle,
    const char* relObjHandle,
    const float* matr,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::vector<float>
    > args(
        objectHandle,
        relObjHandle,
        std::vector<float>(matr,matr+12)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectMatrix",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxClearFloatSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ClearFloatSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxClearIntegerSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ClearIntegerSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxClearStringSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ClearStringSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetFloatSignal(
    const char* sigName,
    float sigValue,
    const char* topic)
{
    std::tuple<
        std::string,
        float
    > args(
        sigName,
        sigValue
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetFloatSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetIntSignal(
    const char* sigName,
    int sigValue,
    const char* topic)
{
    std::tuple<
        std::string,
        int
    > args(
        sigName,
        sigValue
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetIntSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetStringSignal(
    const char* sigName,
    const char* sigValue_data,size_t sigValue_charCnt,
    const char* topic)
{
    std::tuple<
        std::string,
        std::string
    > args(
        sigName,
        std::string(sigValue_data,sigValue_data+sigValue_charCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetStringSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetFloatSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetFloatSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetIntSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetIntSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetStringSignal(
    const char* sigName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        sigName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetStringSignal",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAuxiliaryConsoleClose(
    int consoleHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        consoleHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AuxiliaryConsoleClose",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAuxiliaryConsolePrint(
    int consoleHandle,
    const char* text,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        consoleHandle,
        text
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AuxiliaryConsolePrint",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAuxiliaryConsoleShow(
    int consoleHandle,
    bool showState,
    const char* topic)
{
    std::tuple<
        int,
        bool
    > args(
        consoleHandle,
        showState
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AuxiliaryConsoleShow",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAuxiliaryConsoleOpen(
    const char* title,
    int maxLines,
    int mode,
    const int* position,
    const int* size,
    const int* textColor,
    const int* backgroundColor,
    const char* topic)
{
    std::tuple<
        std::string,
        int,
        int,
        std::vector<int>,
        std::vector<int>,
        std::vector<int>,
        std::vector<int>
    > args(
        title,
        maxLines,
        mode,
        std::vector<int>(position,position+2),
        std::vector<int>(size,size+2),
        std::vector<int>(textColor,textColor+3),
        std::vector<int>(backgroundColor,backgroundColor+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AuxiliaryConsoleOpen",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxStartSimulation(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("StartSimulation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxStopSimulation(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("StopSimulation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxPauseSimulation(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("PauseSimulation",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetVisionSensorImage(
    int objectHandle,
    bool greyScale,
    const char* topic)
{
    std::tuple<
        int,
        bool
    > args(
        objectHandle,
        greyScale
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetVisionSensorImage",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetVisionSensorImage(
    int objectHandle,
    bool greyScale,
    const char* img_data,size_t img_charCnt,
    const char* topic)
{
    std::tuple<
        int,
        bool,
        std::string
    > args(
        objectHandle,
        greyScale,
        std::string(img_data,img_data+img_charCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetVisionSensorImage",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetVisionSensorDepthBuffer(
    int objectHandle,
    bool toMeters,
    bool asByteArray,
    const char* topic)
{
    std::tuple<
        int,
        bool,
        bool
    > args(
        objectHandle,
        toMeters,
        asByteArray
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetVisionSensorDepthBuffer",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddDrawingObject_points(
    int size,
    const int* color,
    const float* coords_data,size_t coords_floatCnt,
    const char* topic)
{
    std::tuple<
        int,
        std::vector<int>,
        std::vector<float>
    > args(
        size,
        std::vector<int>(color,color+3),
        std::vector<float>(coords_data,coords_data+coords_floatCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddDrawingObject_points",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddDrawingObject_spheres(
    float size,
    const int* color,
    const float* coords_data,size_t coords_floatCnt,
    const char* topic)
{
    std::tuple<
        float,
        std::vector<int>,
        std::vector<float>
    > args(
        size,
        std::vector<int>(color,color+3),
        std::vector<float>(coords_data,coords_data+coords_floatCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddDrawingObject_spheres",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddDrawingObject_cubes(
    float size,
    const int* color,
    const float* coords_data,size_t coords_floatCnt,
    const char* topic)
{
    std::tuple<
        float,
        std::vector<int>,
        std::vector<float>
    > args(
        size,
        std::vector<int>(color,color+3),
        std::vector<float>(coords_data,coords_data+coords_floatCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddDrawingObject_cubes",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddDrawingObject_segments(
    int lineSize,
    const int* color,
    const float* segments_data,size_t segments_floatCnt,
    const char* topic)
{
    std::tuple<
        int,
        std::vector<int>,
        std::vector<float>
    > args(
        lineSize,
        std::vector<int>(color,color+3),
        std::vector<float>(segments_data,segments_data+segments_floatCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddDrawingObject_segments",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxAddDrawingObject_triangles(
    const int* color,
    const float* triangles_data,size_t triangles_floatCnt,
    const char* topic)
{
    std::tuple<
        std::vector<int>,
        std::vector<float>
    > args(
        std::vector<int>(color,color+3),
        std::vector<float>(triangles_data,triangles_data+triangles_floatCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("AddDrawingObject_triangles",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxRemoveDrawingObject(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("RemoveDrawingObject",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetCollisionHandle(
    const char* nameOfObject,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        nameOfObject
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetCollisionHandle",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetDistanceHandle(
    const char* nameOfObject,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        nameOfObject
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetDistanceHandle",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxReadCollision(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ReadCollision",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxReadDistance(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ReadDistance",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckCollision(
    int entity1,
    int entity2,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        entity1,
        entity2
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckCollision",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckCollision(
    int entity1,
    const char* entity2,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        entity1,
        entity2
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckCollision",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckDistance(
    int entity1,
    int entity2,
    float threshold,
    const char* topic)
{
    std::tuple<
        int,
        int,
        float
    > args(
        entity1,
        entity2,
        threshold
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckDistance",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckDistance(
    int entity1,
    const char* entity2,
    float threshold,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        float
    > args(
        entity1,
        entity2,
        threshold
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckDistance",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxReadProximitySensor(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ReadProximitySensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckProximitySensor(
    int handle,
    int entity,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        handle,
        entity
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckProximitySensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckProximitySensor(
    int handle,
    const char* entity,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        handle,
        entity
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckProximitySensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxReadForceSensor(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ReadForceSensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxBreakForceSensor(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("BreakForceSensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxReadVisionSensor(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ReadVisionSensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckVisionSensor(
    int handle,
    int entity,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        handle,
        entity
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckVisionSensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCheckVisionSensor(
    int handle,
    const char* entity,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        handle,
        entity
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CheckVisionSensor",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCopyPasteObjects(
    const int* objectHandles_data,size_t objectHandles_intCnt,
    int options,
    const char* topic)
{
    std::tuple<
        std::vector<int>,
        int
    > args(
        std::vector<int>(objectHandles_data,objectHandles_data+objectHandles_intCnt),
        options
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CopyPasteObjects",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxRemoveObjects(
    const int* objectHandles_data,size_t objectHandles_intCnt,
    int options,
    const char* topic)
{
    std::tuple<
        std::vector<int>,
        int
    > args(
        std::vector<int>(objectHandles_data,objectHandles_data+objectHandles_intCnt),
        options
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("RemoveObjects",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCloseScene(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CloseScene",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetStringParameter(
    int paramId,
    const char* paramVal,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetStringParameter(
    const char* paramId,
    const char* paramVal,
    const char* topic)
{
    std::tuple<
        std::string,
        std::string
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetFloatParameter(
    int paramId,
    float paramVal,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetFloatParameter(
    const char* paramId,
    float paramVal,
    const char* topic)
{
    std::tuple<
        std::string,
        float
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetArrayParameter(
    int paramId,
    const float* paramVal,
    const char* topic)
{
    std::tuple<
        int,
        std::vector<float>
    > args(
        paramId,
        std::vector<float>(paramVal,paramVal+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetArrayParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetArrayParameter(
    const char* paramId,
    const float* paramVal,
    const char* topic)
{
    std::tuple<
        std::string,
        std::vector<float>
    > args(
        paramId,
        std::vector<float>(paramVal,paramVal+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetArrayParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetIntParameter(
    int paramId,
    int paramVal,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetIntParameter(
    const char* paramId,
    int paramVal,
    const char* topic)
{
    std::tuple<
        std::string,
        int
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetBoolParameter(
    int paramId,
    bool paramVal,
    const char* topic)
{
    std::tuple<
        int,
        bool
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetBoolParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetBoolParameter(
    const char* paramId,
    bool paramVal,
    const char* topic)
{
    std::tuple<
        std::string,
        bool
    > args(
        paramId,
        paramVal
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetBoolParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetStringParameter(
    int paramId,
    const char* topic)
{
    std::tuple<
        int
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetStringParameter(
    const char* paramId,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetFloatParameter(
    int paramId,
    const char* topic)
{
    std::tuple<
        int
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetFloatParameter(
    const char* paramId,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetArrayParameter(
    int paramId,
    const char* topic)
{
    std::tuple<
        int
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetArrayParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetArrayParameter(
    const char* paramId,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetArrayParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetIntParameter(
    int paramId,
    const char* topic)
{
    std::tuple<
        int
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetIntParameter(
    const char* paramId,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetBoolParameter(
    int paramId,
    const char* topic)
{
    std::tuple<
        int
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetBoolParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetBoolParameter(
    const char* paramId,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        paramId
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetBoolParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxDisplayDialog(
    const char* titleText,
    const char* mainText,
    int dialogType,
    const char* inputText,
    const char* topic)
{
    std::tuple<
        std::string,
        std::string,
        int,
        std::string
    > args(
        titleText,
        mainText,
        dialogType,
        inputText
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("DisplayDialog",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxDisplayDialog(
    const char* titleText,
    const char* mainText,
    const char* dialogType,
    const char* inputText,
    const char* topic)
{
    std::tuple<
        std::string,
        std::string,
        std::string,
        std::string
    > args(
        titleText,
        mainText,
        dialogType,
        inputText
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("DisplayDialog",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetDialogResult(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetDialogResult",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetDialogInput(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetDialogInput",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxEndDialog(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("EndDialog",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxExecuteScriptString(
    const char* code,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        code
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("ExecuteScriptString",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetCollectionHandle(
    const char* collectionName,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        collectionName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetCollectionHandle",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetJointForce(
    int jointHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        jointHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetJointForce",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetJointMaxForce(
    int jointHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        jointHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetJointMaxForce",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetJointForce(
    int jointHandle,
    float forceOrTorque,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        jointHandle,
        forceOrTorque
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetJointForce",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetJointMaxForce(
    int jointHandle,
    float forceOrTorque,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        jointHandle,
        forceOrTorque
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetJointMaxForce",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetJointPosition(
    int jointHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        jointHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetJointPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetJointPosition(
    int jointHandle,
    float position,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        jointHandle,
        position
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetJointPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetJointTargetPosition(
    int jointHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        jointHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetJointTargetPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetJointTargetPosition(
    int jointHandle,
    float targetPos,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        jointHandle,
        targetPos
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetJointTargetPosition",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetJointTargetVelocity(
    int jointHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        jointHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetJointTargetVelocity",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetJointTargetVelocity(
    int jointHandle,
    float targetPos,
    const char* topic)
{
    std::tuple<
        int,
        float
    > args(
        jointHandle,
        targetPos
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetJointTargetVelocity",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectChild(
    int objectHandle,
    int index,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        index
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectChild",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectParent(
    int objectHandle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        objectHandle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectParent",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectParent(
    int objectHandle,
    int parentHandle,
    bool assembly,
    bool keepInPlace,
    const char* topic)
{
    std::tuple<
        int,
        int,
        bool,
        bool
    > args(
        objectHandle,
        parentHandle,
        assembly,
        keepInPlace
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectParent",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectsInTree(
    int treeBaseHandle,
    const char* objectType,
    int options,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        int
    > args(
        treeBaseHandle,
        objectType,
        options
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectsInTree",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectsInTree(
    const char* treeBaseHandle,
    const char* objectType,
    int options,
    const char* topic)
{
    std::tuple<
        std::string,
        std::string,
        int
    > args(
        treeBaseHandle,
        objectType,
        options
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectsInTree",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectName(
    int objectHandle,
    bool altName,
    const char* topic)
{
    std::tuple<
        int,
        bool
    > args(
        objectHandle,
        altName
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectName",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectFloatParameter(
    int objectHandle,
    int parameterID,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectFloatParameter(
    int objectHandle,
    const char* parameterID,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectIntParameter(
    int objectHandle,
    int parameterID,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectIntParameter(
    int objectHandle,
    const char* parameterID,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectStringParameter(
    int objectHandle,
    int parameterID,
    const char* topic)
{
    std::tuple<
        int,
        int
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectStringParameter(
    int objectHandle,
    const char* parameterID,
    const char* topic)
{
    std::tuple<
        int,
        std::string
    > args(
        objectHandle,
        parameterID
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectFloatParameter(
    int objectHandle,
    int parameterID,
    float parameter,
    const char* topic)
{
    std::tuple<
        int,
        int,
        float
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectFloatParameter(
    int objectHandle,
    const char* parameterID,
    float parameter,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        float
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectFloatParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectIntParameter(
    int objectHandle,
    int parameterID,
    int parameter,
    const char* topic)
{
    std::tuple<
        int,
        int,
        int
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectIntParameter(
    int objectHandle,
    const char* parameterID,
    int parameter,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        int
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectIntParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectStringParameter(
    int objectHandle,
    int parameterID,
    const char* parameter,
    const char* topic)
{
    std::tuple<
        int,
        int,
        std::string
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectStringParameter(
    int objectHandle,
    const char* parameterID,
    const char* parameter,
    const char* topic)
{
    std::tuple<
        int,
        std::string,
        std::string
    > args(
        objectHandle,
        parameterID,
        parameter
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectStringParameter",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetSimulationTime(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetSimulationTime",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetSimulationTimeStep(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetSimulationTimeStep",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetServerTimeInMs(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetServerTimeInMs",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetSimulationState(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetSimulationState",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxEvaluateToInt(
    const char* str,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        str
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("EvaluateToInt",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxEvaluateToStr(
    const char* str,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        str
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("EvaluateToStr",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjects(
    int objectType,
    const char* topic)
{
    std::tuple<
        int
    > args(
        objectType
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjects",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjects(
    const char* objectType,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        objectType
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjects",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxCreateDummy(
    float size,
    const int* color,
    const char* topic)
{
    std::tuple<
        float,
        std::vector<int>
    > args(
        size,
        std::vector<int>(color,color+3)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("CreateDummy",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectSelection(
    const char* topic)
{
    std::tuple<
        int
    > args(
        0
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectSelection",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxSetObjectSelection(
    const int* selection_data,size_t selection_intCnt,
    const char* topic)
{
    std::tuple<
        std::vector<int>
    > args(
        std::vector<int>(selection_data,selection_data+selection_intCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("SetObjectSelection",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxGetObjectVelocity(
    int handle,
    const char* topic)
{
    std::tuple<
        int
    > args(
        handle
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("GetObjectVelocity",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxLoadModelFromFile(
    const char* filename,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        filename
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("LoadModelFromFile",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxLoadModelFromBuffer(
    const char* buffer_data,size_t buffer_charCnt,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        std::string(buffer_data,buffer_data+buffer_charCnt)
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("LoadModelFromBuffer",packedArgs.str(),topic));
}
std::vector<msgpack::object>* b0RemoteApi::simxLoadScene(
    const char* filename,
    const char* topic)
{
    std::tuple<
        std::string
    > args(
        filename
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("LoadScene",packedArgs.str(),topic));
}

// -----------------------------------------------------------
// Add your custom functions here (and in the *.h file), or even better,
// add them to b0RemoteApiBindings/generate/simxFunctions.xml,
// and generate this file again.
// Then add the server part of your custom functions at the
// beginning of file lua/b0RemoteApiServer.lua
// -----------------------------------------------------------
