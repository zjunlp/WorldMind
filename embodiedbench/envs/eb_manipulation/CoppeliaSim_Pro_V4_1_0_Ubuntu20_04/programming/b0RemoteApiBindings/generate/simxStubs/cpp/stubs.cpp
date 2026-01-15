#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
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



#py for cmd in plugin.commands:
#py if cmd.generic and cmd.generateCode:
#py loopCnt=1
#py for p in cmd.params:
#py if p.ctype()=='int_eval':
#py loopCnt=2
#py endif
#py endfor
#py for k in range(loopCnt):
std::vector<msgpack::object>* b0RemoteApi::`cmd.name`(
#py theStringToWrite=''
#py itemCnt=len(cmd.params)
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='    int '+p.name
#py else:
#py theStringToWrite+='    const char* '+p.name
#py endif
#py elif p.htype()=='byte[]':
#py theStringToWrite+='    const char* '+p.name+'_data,size_t '+p.name+'_charCnt'
#py elif p.htype()=='int[]':
#py theStringToWrite+='    const int* '+p.name+'_data,size_t '+p.name+'_intCnt'
#py elif 'int[' in p.htype():
#py theStringToWrite+='    const int* '+p.name
#py elif p.htype()=='float[]':
#py theStringToWrite+='    const float* '+p.name+'_data,size_t '+p.name+'_floatCnt'
#py elif 'float[' in p.htype():
#py theStringToWrite+='    const float* '+p.name
#py elif p.htype()=='double[]':
#py theStringToWrite+='    const double* '+p.name+'_data,size_t '+p.name+'_doubleCnt'
#py elif 'double[' in p.htype():
#py theStringToWrite+='    const double* '+p.name
#py else:
#py theStringToWrite+='    '+p.htype()+' '+p.name
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=',\n'
#py endif
#py endfor
#py theStringToWrite+=')\n{\n    std::tuple<'
`theStringToWrite`
#py theStringToWrite=''
#py if len(cmd.params)==1:
#py theStringToWrite+='        int'
#py else:
#py itemCnt=len(cmd.params)-1
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.ctype()=='string':
#py theStringToWrite+='        std::string'
#py elif p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='        int'
#py else:
#py theStringToWrite+='        std::string'
#py endif
#py elif p.htype()=='byte[]':
#py theStringToWrite+='        std::string'
#py elif 'int[' in p.htype():
#py theStringToWrite+='        std::vector<int>'
#py elif 'float[' in p.htype():
#py theStringToWrite+='        std::vector<float>'
#py elif 'double[' in p.htype():
#py theStringToWrite+='        std::vector<double>'
#py else:
#py theStringToWrite+='        '+p.htype()
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=','
#py theStringToWrite+='\n'
#py else:
#py break
#py endif
#py endfor
#py endif
`theStringToWrite`
    > args(
#py theStringToWrite=''
#py if len(cmd.params)==1:
#py theStringToWrite+='        0'
#py else:
#py itemCnt=len(cmd.params)-1
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.ctype()=='string':
#py theStringToWrite+='        '+p.name
#py elif p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='        '+p.name
#py else:
#py theStringToWrite+='        '+p.name
#py endif
#py elif p.htype()=='byte[]':
#py theStringToWrite+='        std::string('+p.name+'_data,'+p.name+'_data+'+p.name+'_charCnt)'
#py elif p.htype()=='int[]':
#py theStringToWrite+='        std::vector<int>('+p.name+'_data,'+p.name+'_data+'+p.name+'_intCnt)'
#py elif p.htype()=='int[2]':
#py theStringToWrite+='        std::vector<int>('+p.name+','+p.name+'+2)'
#py elif p.htype()=='int[3]':
#py theStringToWrite+='        std::vector<int>('+p.name+','+p.name+'+3)'
#py elif p.htype()=='float[]':
#py theStringToWrite+='        std::vector<float>('+p.name+'_data,'+p.name+'_data+'+p.name+'_floatCnt)'
#py elif p.htype()=='float[2]':
#py theStringToWrite+='        std::vector<float>('+p.name+','+p.name+'+2)'
#py elif p.htype()=='float[3]':
#py theStringToWrite+='        std::vector<float>('+p.name+','+p.name+'+3)'
#py elif p.htype()=='float[4]':
#py theStringToWrite+='        std::vector<float>('+p.name+','+p.name+'+4)'
#py elif p.htype()=='float[7]':
#py theStringToWrite+='        std::vector<float>('+p.name+','+p.name+'+7)'
#py elif p.htype()=='float[12]':
#py theStringToWrite+='        std::vector<float>('+p.name+','+p.name+'+12)'
#py elif p.htype()=='double[]':
#py theStringToWrite+='        std::vector<double>('+p.name+'_data,'+p.name+'_data+'+p.name+'_doubleCnt)'
#py elif p.htype()=='double[2]':
#py theStringToWrite+='        std::vector<double>('+p.name+','+p.name+'+2)'
#py elif p.htype()=='double[3]':
#py theStringToWrite+='        std::vector<double>('+p.name+','+p.name+'+3)'
#py elif p.htype()=='double[4]':
#py theStringToWrite+='        std::vector<double>('+p.name+','+p.name+'+4)'
#py elif p.htype()=='double[7]':
#py theStringToWrite+='        std::vector<double>('+p.name+','+p.name+'+7)'
#py elif p.htype()=='double[12]':
#py theStringToWrite+='        std::vector<double>('+p.name+','+p.name+'+12)'
#py else:
#py theStringToWrite+='        '+p.name
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=',\n'
#py else:
#py break
#py endif
#py endfor
#py endif
`theStringToWrite`
    );
    std::stringstream packedArgs;
    msgpack::pack(packedArgs,args);
    return(_handleFunction("`cmd.name[4:]`",packedArgs.str(),topic));
}
#py endfor
#py endif
#py endfor

// -----------------------------------------------------------
// Add your custom functions here (and in the *.h file), or even better,
// add them to b0RemoteApiBindings/generate/simxFunctions.xml,
// and generate this file again.
// Then add the server part of your custom functions at the
// beginning of file lua/b0RemoteApiServer.lua
// -----------------------------------------------------------
