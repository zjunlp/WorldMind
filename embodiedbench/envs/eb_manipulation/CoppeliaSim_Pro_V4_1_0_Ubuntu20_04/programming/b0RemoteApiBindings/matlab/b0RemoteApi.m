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
    
        function ret = simxGetObjectHandle(obj,objectName,topic)
            args = {objectName};
            ret = obj.handleFunction('GetObjectHandle',args,topic);
        end
        function ret = simxAddStatusbarMessage(obj,msg,topic)
            args = {msg};
            ret = obj.handleFunction('AddStatusbarMessage',args,topic);
        end
        function ret = simxGetObjectPosition(obj,objectHandle,relObjHandle,topic)
            args = {objectHandle,relObjHandle};
            ret = obj.handleFunction('GetObjectPosition',args,topic);
        end
        function ret = simxGetObjectOrientation(obj,objectHandle,relObjHandle,topic)
            args = {objectHandle,relObjHandle};
            ret = obj.handleFunction('GetObjectOrientation',args,topic);
        end
        function ret = simxGetObjectQuaternion(obj,objectHandle,relObjHandle,topic)
            args = {objectHandle,relObjHandle};
            ret = obj.handleFunction('GetObjectQuaternion',args,topic);
        end
        function ret = simxGetObjectPose(obj,objectHandle,relObjHandle,topic)
            args = {objectHandle,relObjHandle};
            ret = obj.handleFunction('GetObjectPose',args,topic);
        end
        function ret = simxGetObjectMatrix(obj,objectHandle,relObjHandle,topic)
            args = {objectHandle,relObjHandle};
            ret = obj.handleFunction('GetObjectMatrix',args,topic);
        end
        function ret = simxSetObjectPosition(obj,objectHandle,relObjHandle,position,topic)
            args = {objectHandle,relObjHandle,position};
            ret = obj.handleFunction('SetObjectPosition',args,topic);
        end
        function ret = simxSetObjectOrientation(obj,objectHandle,relObjHandle,euler,topic)
            args = {objectHandle,relObjHandle,euler};
            ret = obj.handleFunction('SetObjectOrientation',args,topic);
        end
        function ret = simxSetObjectQuaternion(obj,objectHandle,relObjHandle,quat,topic)
            args = {objectHandle,relObjHandle,quat};
            ret = obj.handleFunction('SetObjectQuaternion',args,topic);
        end
        function ret = simxSetObjectPose(obj,objectHandle,relObjHandle,pose,topic)
            args = {objectHandle,relObjHandle,pose};
            ret = obj.handleFunction('SetObjectPose',args,topic);
        end
        function ret = simxSetObjectMatrix(obj,objectHandle,relObjHandle,matr,topic)
            args = {objectHandle,relObjHandle,matr};
            ret = obj.handleFunction('SetObjectMatrix',args,topic);
        end
        function ret = simxClearFloatSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('ClearFloatSignal',args,topic);
        end
        function ret = simxClearIntegerSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('ClearIntegerSignal',args,topic);
        end
        function ret = simxClearStringSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('ClearStringSignal',args,topic);
        end
        function ret = simxSetFloatSignal(obj,sigName,sigValue,topic)
            args = {sigName,sigValue};
            ret = obj.handleFunction('SetFloatSignal',args,topic);
        end
        function ret = simxSetIntSignal(obj,sigName,sigValue,topic)
            args = {sigName,sigValue};
            ret = obj.handleFunction('SetIntSignal',args,topic);
        end
        function ret = simxSetStringSignal(obj,sigName,sigValue,topic)
            args = {sigName,sigValue};
            ret = obj.handleFunction('SetStringSignal',args,topic);
        end
        function ret = simxGetFloatSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('GetFloatSignal',args,topic);
        end
        function ret = simxGetIntSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('GetIntSignal',args,topic);
        end
        function ret = simxGetStringSignal(obj,sigName,topic)
            args = {sigName};
            ret = obj.handleFunction('GetStringSignal',args,topic);
        end
        function ret = simxAuxiliaryConsoleClose(obj,consoleHandle,topic)
            args = {consoleHandle};
            ret = obj.handleFunction('AuxiliaryConsoleClose',args,topic);
        end
        function ret = simxAuxiliaryConsolePrint(obj,consoleHandle,text,topic)
            args = {consoleHandle,text};
            ret = obj.handleFunction('AuxiliaryConsolePrint',args,topic);
        end
        function ret = simxAuxiliaryConsoleShow(obj,consoleHandle,showState,topic)
            args = {consoleHandle,showState};
            ret = obj.handleFunction('AuxiliaryConsoleShow',args,topic);
        end
        function ret = simxAuxiliaryConsoleOpen(obj,title,maxLines,mode,position,size,textColor,backgroundColor,topic)
            args = {title,maxLines,mode,position,size,textColor,backgroundColor};
            ret = obj.handleFunction('AuxiliaryConsoleOpen',args,topic);
        end
        function ret = simxStartSimulation(obj,topic)
            args = {0};
            ret = obj.handleFunction('StartSimulation',args,topic);
        end
        function ret = simxStopSimulation(obj,topic)
            args = {0};
            ret = obj.handleFunction('StopSimulation',args,topic);
        end
        function ret = simxPauseSimulation(obj,topic)
            args = {0};
            ret = obj.handleFunction('PauseSimulation',args,topic);
        end
        function ret = simxGetVisionSensorImage(obj,objectHandle,greyScale,topic)
            args = {objectHandle,greyScale};
            ret = obj.handleFunction('GetVisionSensorImage',args,topic);
        end
        function ret = simxSetVisionSensorImage(obj,objectHandle,greyScale,img,topic)
            args = {objectHandle,greyScale,img};
            ret = obj.handleFunction('SetVisionSensorImage',args,topic);
        end
        function ret = simxGetVisionSensorDepthBuffer(obj,objectHandle,toMeters,asByteArray,topic)
            args = {objectHandle,toMeters,asByteArray};
            ret = obj.handleFunction('GetVisionSensorDepthBuffer',args,topic);
        end
        function ret = simxAddDrawingObject_points(obj,size,color,coords,topic)
            args = {size,color,coords};
            ret = obj.handleFunction('AddDrawingObject_points',args,topic);
        end
        function ret = simxAddDrawingObject_spheres(obj,size,color,coords,topic)
            args = {size,color,coords};
            ret = obj.handleFunction('AddDrawingObject_spheres',args,topic);
        end
        function ret = simxAddDrawingObject_cubes(obj,size,color,coords,topic)
            args = {size,color,coords};
            ret = obj.handleFunction('AddDrawingObject_cubes',args,topic);
        end
        function ret = simxAddDrawingObject_segments(obj,lineSize,color,segments,topic)
            args = {lineSize,color,segments};
            ret = obj.handleFunction('AddDrawingObject_segments',args,topic);
        end
        function ret = simxAddDrawingObject_triangles(obj,color,triangles,topic)
            args = {color,triangles};
            ret = obj.handleFunction('AddDrawingObject_triangles',args,topic);
        end
        function ret = simxRemoveDrawingObject(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('RemoveDrawingObject',args,topic);
        end
        function ret = simxGetCollisionHandle(obj,nameOfObject,topic)
            args = {nameOfObject};
            ret = obj.handleFunction('GetCollisionHandle',args,topic);
        end
        function ret = simxGetDistanceHandle(obj,nameOfObject,topic)
            args = {nameOfObject};
            ret = obj.handleFunction('GetDistanceHandle',args,topic);
        end
        function ret = simxReadCollision(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('ReadCollision',args,topic);
        end
        function ret = simxReadDistance(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('ReadDistance',args,topic);
        end
        function ret = simxCheckCollision(obj,entity1,entity2,topic)
            args = {entity1,entity2};
            ret = obj.handleFunction('CheckCollision',args,topic);
        end
        function ret = simxCheckDistance(obj,entity1,entity2,threshold,topic)
            args = {entity1,entity2,threshold};
            ret = obj.handleFunction('CheckDistance',args,topic);
        end
        function ret = simxReadProximitySensor(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('ReadProximitySensor',args,topic);
        end
        function ret = simxCheckProximitySensor(obj,handle,entity,topic)
            args = {handle,entity};
            ret = obj.handleFunction('CheckProximitySensor',args,topic);
        end
        function ret = simxReadForceSensor(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('ReadForceSensor',args,topic);
        end
        function ret = simxBreakForceSensor(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('BreakForceSensor',args,topic);
        end
        function ret = simxReadVisionSensor(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('ReadVisionSensor',args,topic);
        end
        function ret = simxCheckVisionSensor(obj,handle,entity,topic)
            args = {handle,entity};
            ret = obj.handleFunction('CheckVisionSensor',args,topic);
        end
        function ret = simxCopyPasteObjects(obj,objectHandles,options,topic)
            args = {objectHandles,options};
            ret = obj.handleFunction('CopyPasteObjects',args,topic);
        end
        function ret = simxRemoveObjects(obj,objectHandles,options,topic)
            args = {objectHandles,options};
            ret = obj.handleFunction('RemoveObjects',args,topic);
        end
        function ret = simxCloseScene(obj,topic)
            args = {0};
            ret = obj.handleFunction('CloseScene',args,topic);
        end
        function ret = simxSetStringParameter(obj,paramId,paramVal,topic)
            args = {paramId,paramVal};
            ret = obj.handleFunction('SetStringParameter',args,topic);
        end
        function ret = simxSetFloatParameter(obj,paramId,paramVal,topic)
            args = {paramId,paramVal};
            ret = obj.handleFunction('SetFloatParameter',args,topic);
        end
        function ret = simxSetArrayParameter(obj,paramId,paramVal,topic)
            args = {paramId,paramVal};
            ret = obj.handleFunction('SetArrayParameter',args,topic);
        end
        function ret = simxSetIntParameter(obj,paramId,paramVal,topic)
            args = {paramId,paramVal};
            ret = obj.handleFunction('SetIntParameter',args,topic);
        end
        function ret = simxSetBoolParameter(obj,paramId,paramVal,topic)
            args = {paramId,paramVal};
            ret = obj.handleFunction('SetBoolParameter',args,topic);
        end
        function ret = simxGetStringParameter(obj,paramId,topic)
            args = {paramId};
            ret = obj.handleFunction('GetStringParameter',args,topic);
        end
        function ret = simxGetFloatParameter(obj,paramId,topic)
            args = {paramId};
            ret = obj.handleFunction('GetFloatParameter',args,topic);
        end
        function ret = simxGetArrayParameter(obj,paramId,topic)
            args = {paramId};
            ret = obj.handleFunction('GetArrayParameter',args,topic);
        end
        function ret = simxGetIntParameter(obj,paramId,topic)
            args = {paramId};
            ret = obj.handleFunction('GetIntParameter',args,topic);
        end
        function ret = simxGetBoolParameter(obj,paramId,topic)
            args = {paramId};
            ret = obj.handleFunction('GetBoolParameter',args,topic);
        end
        function ret = simxDisplayDialog(obj,titleText,mainText,dialogType,inputText,topic)
            args = {titleText,mainText,dialogType,inputText};
            ret = obj.handleFunction('DisplayDialog',args,topic);
        end
        function ret = simxGetDialogResult(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('GetDialogResult',args,topic);
        end
        function ret = simxGetDialogInput(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('GetDialogInput',args,topic);
        end
        function ret = simxEndDialog(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('EndDialog',args,topic);
        end
        function ret = simxExecuteScriptString(obj,code,topic)
            args = {code};
            ret = obj.handleFunction('ExecuteScriptString',args,topic);
        end
        function ret = simxGetCollectionHandle(obj,collectionName,topic)
            args = {collectionName};
            ret = obj.handleFunction('GetCollectionHandle',args,topic);
        end
        function ret = simxGetJointForce(obj,jointHandle,topic)
            args = {jointHandle};
            ret = obj.handleFunction('GetJointForce',args,topic);
        end
        function ret = simxGetJointMaxForce(obj,jointHandle,topic)
            args = {jointHandle};
            ret = obj.handleFunction('GetJointMaxForce',args,topic);
        end
        function ret = simxSetJointForce(obj,jointHandle,forceOrTorque,topic)
            args = {jointHandle,forceOrTorque};
            ret = obj.handleFunction('SetJointForce',args,topic);
        end
        function ret = simxSetJointMaxForce(obj,jointHandle,forceOrTorque,topic)
            args = {jointHandle,forceOrTorque};
            ret = obj.handleFunction('SetJointMaxForce',args,topic);
        end
        function ret = simxGetJointPosition(obj,jointHandle,topic)
            args = {jointHandle};
            ret = obj.handleFunction('GetJointPosition',args,topic);
        end
        function ret = simxSetJointPosition(obj,jointHandle,position,topic)
            args = {jointHandle,position};
            ret = obj.handleFunction('SetJointPosition',args,topic);
        end
        function ret = simxGetJointTargetPosition(obj,jointHandle,topic)
            args = {jointHandle};
            ret = obj.handleFunction('GetJointTargetPosition',args,topic);
        end
        function ret = simxSetJointTargetPosition(obj,jointHandle,targetPos,topic)
            args = {jointHandle,targetPos};
            ret = obj.handleFunction('SetJointTargetPosition',args,topic);
        end
        function ret = simxGetJointTargetVelocity(obj,jointHandle,topic)
            args = {jointHandle};
            ret = obj.handleFunction('GetJointTargetVelocity',args,topic);
        end
        function ret = simxSetJointTargetVelocity(obj,jointHandle,targetPos,topic)
            args = {jointHandle,targetPos};
            ret = obj.handleFunction('SetJointTargetVelocity',args,topic);
        end
        function ret = simxGetObjectChild(obj,objectHandle,index,topic)
            args = {objectHandle,index};
            ret = obj.handleFunction('GetObjectChild',args,topic);
        end
        function ret = simxGetObjectParent(obj,objectHandle,topic)
            args = {objectHandle};
            ret = obj.handleFunction('GetObjectParent',args,topic);
        end
        function ret = simxSetObjectParent(obj,objectHandle,parentHandle,assembly,keepInPlace,topic)
            args = {objectHandle,parentHandle,assembly,keepInPlace};
            ret = obj.handleFunction('SetObjectParent',args,topic);
        end
        function ret = simxGetObjectsInTree(obj,treeBaseHandle,objectType,options,topic)
            args = {treeBaseHandle,objectType,options};
            ret = obj.handleFunction('GetObjectsInTree',args,topic);
        end
        function ret = simxGetObjectName(obj,objectHandle,altName,topic)
            args = {objectHandle,altName};
            ret = obj.handleFunction('GetObjectName',args,topic);
        end
        function ret = simxGetObjectFloatParameter(obj,objectHandle,parameterID,topic)
            args = {objectHandle,parameterID};
            ret = obj.handleFunction('GetObjectFloatParameter',args,topic);
        end
        function ret = simxGetObjectIntParameter(obj,objectHandle,parameterID,topic)
            args = {objectHandle,parameterID};
            ret = obj.handleFunction('GetObjectIntParameter',args,topic);
        end
        function ret = simxGetObjectStringParameter(obj,objectHandle,parameterID,topic)
            args = {objectHandle,parameterID};
            ret = obj.handleFunction('GetObjectStringParameter',args,topic);
        end
        function ret = simxSetObjectFloatParameter(obj,objectHandle,parameterID,parameter,topic)
            args = {objectHandle,parameterID,parameter};
            ret = obj.handleFunction('SetObjectFloatParameter',args,topic);
        end
        function ret = simxSetObjectIntParameter(obj,objectHandle,parameterID,parameter,topic)
            args = {objectHandle,parameterID,parameter};
            ret = obj.handleFunction('SetObjectIntParameter',args,topic);
        end
        function ret = simxSetObjectStringParameter(obj,objectHandle,parameterID,parameter,topic)
            args = {objectHandle,parameterID,parameter};
            ret = obj.handleFunction('SetObjectStringParameter',args,topic);
        end
        function ret = simxGetSimulationTime(obj,topic)
            args = {0};
            ret = obj.handleFunction('GetSimulationTime',args,topic);
        end
        function ret = simxGetSimulationTimeStep(obj,topic)
            args = {0};
            ret = obj.handleFunction('GetSimulationTimeStep',args,topic);
        end
        function ret = simxGetServerTimeInMs(obj,topic)
            args = {0};
            ret = obj.handleFunction('GetServerTimeInMs',args,topic);
        end
        function ret = simxGetSimulationState(obj,topic)
            args = {0};
            ret = obj.handleFunction('GetSimulationState',args,topic);
        end
        function ret = simxEvaluateToInt(obj,str,topic)
            args = {str};
            ret = obj.handleFunction('EvaluateToInt',args,topic);
        end
        function ret = simxEvaluateToStr(obj,str,topic)
            args = {str};
            ret = obj.handleFunction('EvaluateToStr',args,topic);
        end
        function ret = simxGetObjects(obj,objectType,topic)
            args = {objectType};
            ret = obj.handleFunction('GetObjects',args,topic);
        end
        function ret = simxCreateDummy(obj,size,color,topic)
            args = {size,color};
            ret = obj.handleFunction('CreateDummy',args,topic);
        end
        function ret = simxGetObjectSelection(obj,topic)
            args = {0};
            ret = obj.handleFunction('GetObjectSelection',args,topic);
        end
        function ret = simxSetObjectSelection(obj,selection,topic)
            args = {selection};
            ret = obj.handleFunction('SetObjectSelection',args,topic);
        end
        function ret = simxGetObjectVelocity(obj,handle,topic)
            args = {handle};
            ret = obj.handleFunction('GetObjectVelocity',args,topic);
        end
        function ret = simxLoadModelFromFile(obj,filename,topic)
            args = {filename};
            ret = obj.handleFunction('LoadModelFromFile',args,topic);
        end
        function ret = simxLoadModelFromBuffer(obj,buffer,topic)
            args = {buffer};
            ret = obj.handleFunction('LoadModelFromBuffer',args,topic);
        end
        function ret = simxLoadScene(obj,filename,topic)
            args = {filename};
            ret = obj.handleFunction('LoadScene',args,topic);
        end

        % -----------------------------------------------------------
        % Add your custom functions here, or even better,
        % add them to b0RemoteApiBindings/generate/simxFunctions.xml,
        % and generate this file again.
        % Then add the server part of your custom functions at the
        % beginning of file lua/b0RemoteApiServer.lua
        % -----------------------------------------------------------
        
    end
end
