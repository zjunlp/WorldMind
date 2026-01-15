#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
% -------------------------------------------------------
% Add your custom functions at the bottom of the file
% and the server counterpart to lua/b0RemoteApiServer.lua
% -------------------------------------------------------

classdef b0RemoteApi < handle

    properties
        libName;
        hFile;
        pongReceived;
        channelName;
        serviceCallTopic;
        defaultPublisherTopic;
        defaultSubscriberTopic;
        nextDefaultSubscriberHandle;
        nextDedicatedPublisherHandle;
        nextDedicatedSubscriberHandle;
        node;
        clientId;
        serviceClient;
        defaultPublisher;
        defaultSubscriber;
        allSubscribers;
        allDedicatedPublishers;
        nodeName;
        inactivityToleranceInSec;
        timeout;
        setupSubscribersAsynchronously;
    end
    
    methods
        function cleanUp(obj)
            disp('*************************************************************************************');
            disp('** Leaving... if this is unexpected, you might have to adjust the timeout argument **');
            disp('*************************************************************************************');
            obj.pongReceived=false;
            obj.handleFunction('Ping',{0},obj.simxDefaultSubscriber(@obj.pingCallback));
            while not(obj.pongReceived)
                obj.simxSpinOnce();
            end
            obj.handleFunction('DisconnectClient',{obj.clientId},obj.serviceCallTopic);
            
            allKeys=keys(obj.allSubscribers);
            for key=allKeys
                value=obj.allSubscribers(key{1});
                if value('handle')~=obj.defaultSubscriber
                    calllib(obj.libName,'b0_subscriber_delete',value('handle'));
                end
            end
            allKeys=keys(obj.allDedicatedPublishers);
            for key=allKeys
                value=obj.allDedicatedPublishers(key{1});
                calllib(obj.libName,'b0_publisher_delete',value);
            end
%            calllib(obj.libName,'b0_node_delete',obj.node);
            unloadlibrary(obj.libName);
        end
        
        function pingCallback(obj,msg)
            obj.pongReceived=true;
        end
        
        function obj = b0RemoteApi(nodeName,channelName,inactivityToleranceInSec,setupSubscribersAsynchronously,timeout,hfile)
            addpath('./msgpack-matlab/');
            obj.libName = 'b0';
            obj.nodeName='b0RemoteApi_matlabClient';
            obj.channelName='b0RemoteApi';
            obj.inactivityToleranceInSec=60;
            obj.setupSubscribersAsynchronously=false;
            obj.timeout=3
            if nargin>=1
                obj.nodeName=nodeName;
            end
            if nargin>=2
                obj.channelName=channelName;
            end
            if nargin>=3
                obj.inactivityToleranceInSec=inactivityToleranceInSec;
            end
            if nargin>=4
                obj.setupSubscribersAsynchronously=setupSubscribersAsynchronously;
            end
            if nargin>=5
                obj.timeout=timeout;
            end
            obj.serviceCallTopic=strcat(obj.channelName,'SerX');
            obj.defaultPublisherTopic=strcat(obj.channelName,'SubX');
            obj.defaultSubscriberTopic=strcat(obj.channelName,'PubX');
            obj.nextDefaultSubscriberHandle=2;
            obj.nextDedicatedPublisherHandle=500;
            obj.nextDedicatedSubscriberHandle=1000;
            if ~libisloaded(obj.libName)
                if nargin>=6
                    obj.hFile = hfile;
                    loadlibrary(obj.libName,obj.hFile);
                else
                    loadlibrary(obj.libName,@b0RemoteApiProto);
                end
            end
            initialized=calllib(obj.libName,'b0_is_initialized');
            if ~initialized
                arg1=libpointer('int32Ptr',int32(1));
                arg2=libpointer('stringPtrPtr',{'b0Matlab'});
                calllib(obj.libName,'b0_init',arg1,arg2);
            end
            obj.node = calllib(obj.libName,'b0_node_new',libpointer('int8Ptr',[uint8(obj.nodeName) 0]));
            chars = ['a':'z' 'A':'Z' '0':'9'];
            n = randi(numel(chars),[1 10]);
            obj.clientId= chars(n);
            

            tmp = libpointer('int8Ptr',[uint8(obj.serviceCallTopic) 0]);
            obj.serviceClient = calllib(obj.libName,'b0_service_client_new_ex',obj.node,tmp,1,1);
            calllib(obj.libName,'b0_service_client_set_option',obj.serviceClient,3,obj.timeout*1000);

            tmp = libpointer('int8Ptr',[uint8(obj.defaultPublisherTopic) 0]);
            obj.defaultPublisher = calllib(obj.libName,'b0_publisher_new_ex',obj.node,tmp,1,1);

            tmp = libpointer('int8Ptr',[uint8(obj.defaultSubscriberTopic) 0]);
            obj.defaultSubscriber = calllib(obj.libName,'b0_subscriber_new_ex',obj.node,tmp,[],1,1); % We will poll the socket
            
            
            disp(char(10));
            disp(strcat('  Running B0 Remote API client with channel name [',obj.channelName,']'));
            disp('  make sure that: 1) the B0 resolver is running');
            disp('                  2) CoppeliaSim is running the B0 Remote API server with the same channel name');
            disp('  Initializing...');
            disp(char(10));
            try
                calllib(obj.libName,'b0_node_init',obj.node);
            catch me
                obj.cleanUp();
                rethrow(me);
            end
            obj.handleFunction('inactivityTolerance',{obj.inactivityToleranceInSec},obj.serviceCallTopic);
            disp(char(10));
            disp('  Connected!');
            disp(char(10));
            obj.allSubscribers=containers.Map;
            obj.allDedicatedPublishers=containers.Map;
        end

        function delete(obj)
            obj.cleanUp();
        end

        function topic = simxDefaultPublisher(obj)
            topic = obj.defaultPublisherTopic;
        end

        function topic = simxCreatePublisher(obj,dropMessages)
            if not(exist('dropMessages'))
                dropMessages=false;
            end
            topic=strcat(obj.channelName,'Sub',num2str(obj.nextDedicatedPublisherHandle),obj.clientId);
            obj.nextDedicatedPublisherHandle=obj.nextDedicatedPublisherHandle+1;
            tmp = libpointer('int8Ptr',[uint8(topic) 0]);
            pub = calllib(obj.libName,'b0_publisher_new_ex',obj.node,tmp,0,1);
            calllib(obj.libName,'b0_publisher_init',pub);
            obj.allDedicatedPublishers(topic)=pub;
            obj.handleFunction('createSubscriber',{topic,dropMessages},obj.serviceCallTopic);
        end

        function topic = simxDefaultSubscriber(obj,cb,publishInterval)
            if not(exist('publishInterval'))
                publishInterval=1;
            end
            topic=strcat(obj.channelName,'Pub',num2str(obj.nextDefaultSubscriberHandle),obj.clientId);
            obj.nextDefaultSubscriberHandle=obj.nextDefaultSubscriberHandle+1;
            theMap=containers.Map;
            theMap('handle')=obj.defaultSubscriber;
            theMap('cb')=cb;
            theMap('dropMessages')=false;
            obj.allSubscribers(topic)=theMap;
            channel=obj.serviceCallTopic;
            if obj.setupSubscribersAsynchronously
                channel=obj.defaultPublisherTopic;
            end
            obj.handleFunction('setDefaultPublisherPubInterval',{topic,publishInterval},channel);
        end
            
        function topic = simxCreateSubscriber(obj,cb,publishInterval,dropMessages)
            if not(exist('publishInterval'))
                publishInterval=1;
            end
            if not(exist('dropMessages'))
                dropMessages=false;
            end
            topic=strcat(obj.channelName,'Pub',num2str(obj.nextDedicatedSubscriberHandle),obj.clientId);
            obj.nextDedicatedSubscriberHandle=obj.nextDedicatedSubscriberHandle+1;
            tmp = libpointer('int8Ptr',[uint8(topic) 0]);
            sub = calllib(obj.libName,'b0_subscriber_new_ex',obj.node,tmp,[],0,1); % We will poll the socket
            if dropMessages
                calllib(obj.libName,'b0_subscriber_set_option',sub,6,1); % enable conflate
            else
                calllib(obj.libName,'b0_subscriber_set_option',sub,6,0); % disable conflate
            end
            calllib(obj.libName,'b0_subscriber_init',sub);
            theMap=containers.Map;
            theMap('handle')=sub;
            theMap('cb')=cb;
            theMap('dropMessages')=dropMessages;
            obj.allSubscribers(topic)=theMap;
            channel=obj.serviceCallTopic;
            if obj.setupSubscribersAsynchronously
                channel=obj.defaultPublisherTopic;
            end
            obj.handleFunction('createPublisher',{topic,publishInterval},channel);
        end
  
        function simxRemoveSubscriber(obj,topic)
            if isKey(obj.allSubscribers,topic)
                value=obj.allSubscribers(topic);
                channel=obj.serviceCallTopic;
                if obj.setupSubscribersAsynchronously
                    channel=obj.defaultPublisherTopic;
                end
                if value('handle')==obj.defaultSubscriber
                    obj.handleFunction('stopDefaultPublisher',{topic},channel);
                else
                    calllib(obj.libName,'b0_subscriber_delete',value('handle'));
                    obj.handleFunction('stopPublisher',{topic},channel);
                end
                remove(obj.allSubscribers,topic)
            end
        end
        
        function simxRemovePublisher(obj,topic)
            if isKey(obj.allDedicatedPublishers,topic)
                value=obj.allDedicatedPublishers(topic);
                calllib(obj.libName,'b0_publisher_delete',value);
                obj.handleFunction('stopSubscriber',{topic},obj.serviceCallTopic);
                remove(obj.allDedicatedPublishers,topic)
            end
        end
        
        function topic = simxServiceCall(obj)
            topic = obj.serviceCallTopic;
        end
        
        function simxSpin(obj)
            while true
                obj.simxSpinOnce();
            end
        end
        
        function simxSpinOnce(obj)
            defaultSubscriberAlreadyProcessed=false;
            allKeys=keys(obj.allSubscribers);
            for key=allKeys
                value=obj.allSubscribers(key{1});
                retData=[];
                if (value('handle')~=obj.defaultSubscriber) || not(defaultSubscriberAlreadyProcessed)
                    defaultSubscriberAlreadyProcessed=defaultSubscriberAlreadyProcessed || (value('handle')==obj.defaultSubscriber);
                    while calllib(obj.libName,'b0_subscriber_poll',value('handle'),0)>0
                        if not(isempty(retData))
                            calllib(obj.libName,'b0_buffer_delete',retData);
                            retData=[];
                        end
                        retData = libpointer('uint8PtrPtr');
                        retSize = libpointer('uint64Ptr',uint64(0));
                        [retData subClient retSize]=calllib(obj.libName,'b0_subscriber_read',value('handle'),retSize);
                        retData.setdatatype('uint8Ptr',1,retSize);
                        if not(value('dropMessages'))
                            obj.handleReceivedMessage(retData.value);
                            calllib(obj.libName,'b0_buffer_delete',retData);
                            retData=[];
                        end
                    end
                    if value('dropMessages') && not(isempty(retData))
                        obj.handleReceivedMessage(retData.value);
                        calllib(obj.libName,'b0_buffer_delete',retData);
                        retData=[];
                    end
                end
            end
        end
        
        function handleReceivedMessage(obj,data)
            msg = parsemsgpack(data);
            kk=msg(1);
            k=kk{1};
            if isKey(obj.allSubscribers,k)
                value=obj.allSubscribers(k);
                cbMsg=msg(2);
                if length(cbMsg)==1
                    cbMsg=[cbMsg,[]];
                end
                cb=value('cb');
                cb(cbMsg{1});
            end
        end
        
        function ret = handleFunction(obj,funcName,reqArgs,topic)
            if strcmp(topic,obj.serviceCallTopic)
                packedData = dumpmsgpack({{funcName,obj.clientId,topic,0},reqArgs});
            
                retData = libpointer('uint8PtrPtr');
                retSize = libpointer('uint64Ptr',uint64(0));
                [retData servClient packedData retSize]= calllib(obj.libName,'b0_service_client_call',obj.serviceClient,packedData,length(packedData),retSize);
                if retSize > 0
                    retData.setdatatype('uint8Ptr',1,retSize);
                    returnedData = retData.value;
                    ret = parsemsgpack(returnedData);
                    % Following line new on 16.01.2020:
                    calllib(obj.libName,'b0_buffer_delete',retData);
                    if length(ret)<2
                        ret=[ret,[]];
                    end
                else 
                    ret=[];
                end
            else 
                if strcmp(topic,obj.defaultPublisherTopic)
                    packedData = dumpmsgpack({{funcName,obj.clientId,topic,1},reqArgs});
                    calllib(obj.libName,'b0_publisher_publish',obj.defaultPublisher,packedData,length(packedData));
                    ret=[];
                else 
                    if isKey(obj.allSubscribers,topic)
                        val=obj.allSubscribers(topic);
                        packedData=[];
                        if val('handle')==obj.defaultSubscriber
                            packedData = dumpmsgpack({{funcName,obj.clientId,topic,2},reqArgs});
                        else
                            packedData = dumpmsgpack({{funcName,obj.clientId,topic,4},reqArgs});
                        end
                        if obj.setupSubscribersAsynchronously
                            calllib(obj.libName,'b0_publisher_publish',obj.defaultPublisher,packedData,length(packedData));
                        else
                            retData = libpointer('uint8PtrPtr');
                            retSize = libpointer('uint64Ptr',uint64(0));
                            [retData servClient packedData retSize]= calllib(obj.libName,'b0_service_client_call',obj.serviceClient,packedData,length(packedData),retSize);
                            % Following 3 new on 16.01.2020:
                            if retSize > 0
                                calllib(obj.libName,'b0_buffer_delete',retData);
                            end
                        end
                        ret=[];
                    else 
                        if isKey(obj.allDedicatedPublishers,topic)
                            packedData = dumpmsgpack({{funcName,obj.clientId,topic,3},reqArgs});
                            calllib(obj.libName,'b0_publisher_publish',obj.allDedicatedPublishers,packedData,length(packedData));
                        else
                            disp('B0 Remote API error: invalid topic');
                        end
                        ret=[];
                    end
                end
            end
        end

        function ret = simxGetTimeInMs(obj)
            ret = calllib(obj.libName,'b0_node_hardware_time_usec',obj.node)/1000;
        end
        
        function simxSleep(obj,durationInMs)
            startT=obj.simxGetTimeInMs();
            while obj.simxGetTimeInMs()<startT+durationInMs
            end
        end
        
        function simxSynchronous(obj,enable)
            args = {enable};
            obj.handleFunction('Synchronous',args,obj.serviceCallTopic);
        end
        
        function simxSynchronousTrigger(obj)
            args = {0};
            obj.handleFunction('SynchronousTrigger',args,obj.defaultPublisherTopic);
        end
        
        function simxGetSimulationStepDone(obj,topic)
            if isKey(obj.allSubscribers,topic)
                reqArgs = {0};
                obj.handleFunction('GetSimulationStepDone',reqArgs,topic);
            else
                disp('B0 Remote API error: invalid topic');
            end
        end
        
        function simxGetSimulationStepStarted(obj,topic)
            if isKey(obj.allSubscribers,topic)
                reqArgs = {0};
                obj.handleFunction('GetSimulationStepStarted',reqArgs,topic);
            else
                disp('B0 Remote API error: invalid topic');
            end
        end

        function ret = simxCallScriptFunction(obj,funcAtObjName,scriptType,arg,topic)
            packedData = dumpmsgpack(arg);
            args = {funcAtObjName,scriptType,packedData};
            ret = obj.handleFunction('CallScriptFunction',args,topic);
        end
    
#py for cmd in plugin.commands:
#py if cmd.generic and cmd.generateCode:
#py theStringToWrite='        function ret = '+cmd.name+'(obj,'
#py itemCnt=len(cmd.params)
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py theStringToWrite+=p.name
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=','
#py endif
#py endfor
`theStringToWrite`)
#py if len(cmd.params)==1:
#py theStringToWrite='            args = {0'
#py else:
#py itemCnt=len(cmd.params)-1
#py itemIndex=-1
#py theStringToWrite='            args = {'
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py theStringToWrite+=p.name
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=','
#py else:
#py break
#py endif
#py endfor
#py endif
#py theStringToWrite+='};'
`theStringToWrite`
            ret = obj.handleFunction('`cmd.name[4:]`',args,topic);
        end
#py endif
#py endfor

        % -----------------------------------------------------------
        % Add your custom functions here, or even better,
        % add them to b0RemoteApiBindings/generate/simxFunctions.xml,
        % and generate this file again.
        % Then add the server part of your custom functions at the
        % beginning of file lua/b0RemoteApiServer.lua
        % -----------------------------------------------------------
        
    end
end
