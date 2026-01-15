-- -------------------------------------------------------
-- Add your custom functions at the bottom of the file
-- and the server counterpart to lua/b0RemoteApiServer.lua
-- -------------------------------------------------------
    
require 'b0Lua'
b0.messagePack=require('messagePack-lua/MessagePack')

_printToConsole=print
function print(...)
    _printToConsole(_getAsString(...))
end

function _getAsString(...)
    local a={...}
    local t=''
    if #a==1 and type(a[1])=='string' then
        t=string.format('%s', a[1])
    else
        for i=1,#a,1 do
            if i~=1 then
                t=t..','
            end
            if type(a[i])=='table' then
                t=t.._tableToString(a[i],{},99)
            else
                t=t.._anyToString(a[i],{},99)
            end
        end
    end
    if #a==0 then
        t='nil'
    end
    return(t)
end

function _tableToString(tt,visitedTables,maxLevel,indent)
	indent = indent or 0
    maxLevel=maxLevel-1
	if type(tt) == 'table' then
        if maxLevel<=0 then
            return tostring(tt)
        else
            if  visitedTables[tt] then
                return tostring(tt)..' (already visited)'
            else
                visitedTables[tt]=true
                local sb = {}
                if _isArray(tt) then
                    table.insert(sb, '{')
                    for i = 1, #tt do
                        table.insert(sb, _anyToString(tt[i], visitedTables,maxLevel, indent))
                        if i < #tt then table.insert(sb, ', ') end
                    end
                    table.insert(sb, '}')
                else
                    table.insert(sb, '{\n')
                    -- Print the map content ordered according to type, then key:
                    local a = {}
                    for n in pairs(tt) do table.insert(a, n) end
                    table.sort(a)
                    local tp={'boolean','number','string','function','userdata','thread','table'}
                    for j=1,#tp,1 do
                        for i,n in ipairs(a) do
                            if type(tt[n])==tp[j] then
                                table.insert(sb, string.rep(' ', indent+4))
                                table.insert(sb, tostring(n))
                                table.insert(sb, '=')
                                table.insert(sb, _anyToString(tt[n], visitedTables,maxLevel, indent+4))
                                table.insert(sb, ',\n')
                            end
                        end                
                    end
                    table.insert(sb, string.rep(' ', indent))
                    table.insert(sb, '}')
                end
                visitedTables[tt]=false -- siblings pointing onto a same table should still be explored!
                return table.concat(sb)
            end
        end
    else
        return _anyToString(tt, visitedTables,maxLevel, indent)
    end
end

function _anyToString(x, visitedTables,maxLevel,tblindent)
    local tblindent = tblindent or 0
    if 'nil' == type(x) then
        return tostring(nil)
    elseif 'table' == type(x) then
        return _tableToString(x, visitedTables,maxLevel, tblindent)
    elseif 'string' == type(x) then
        return _getShortString(x)
    else
        return tostring(x)
    end
end

function _getShortString(x)
    if type(x)=='string' then
        if string.find(x,"\0") then
            return "[buffer string]"
        else
            local a,b=string.gsub(x,"[%a%d%p%s]", "@")
            if b~=#x then
                return "[string containing special chars]"
            else
                if #x>160 then
                    return "[long string]"
                else
                    return string.format('%s', x)
                end
            end
        end
    end
    return "[not a string]"
end

function _isArray(t)
    local i = 0
    for _ in pairs(t) do
        i = i + 1
        if t[i] == nil then return false end
    end
    return true
end


function b0RemoteApi(nodeName,channelName,inactivityToleranceInSec,setupSubscribersAsynchronously,timeout)
    local self={}
    
    if nodeName==nil then nodeName='b0RemoteApi_luaClient' end
    if channelName==nil then channelName='b0RemoteApi' end
    if inactivityToleranceInSec==nil then inactivityToleranceInSec=60 end
    if setupSubscribersAsynchronously==nil then setupSubscribersAsynchronously=false end
    if timeout==nil then timeout=3 end
    
    local _channelName=channelName
    local _serviceCallTopic=channelName..'SerX'
    local _defaultPublisherTopic=channelName..'SubX'
    local _defaultSubscriberTopic=channelName..'PubX'
    local _nextDefaultSubscriberHandle=2
    local _nextDedicatedPublisherHandle=500
    local _nextDedicatedSubscriberHandle=1000
    b0.init()
    local _node=b0.node_new(nodeName)
    math.randomseed(b0.node_hardware_time_usec(_node))
    local _clientId=''
    for i=1,10,1 do
        local r=math.random(62)
        _clientId=_clientId..string.sub('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',r,r)
    end
    local _serviceClient=b0.service_client_new_ex(_node,_serviceCallTopic,1,1)
    b0.service_client_set_option(_serviceClient,b0.SOCK_OPT_READTIMEOUT,timeout*1000)
    local _defaultPublisher=b0.publisher_new_ex(_node,_defaultPublisherTopic,1,1)
    local _defaultSubscriber=b0.subscriber_new_ex(_node,_defaultSubscriberTopic,1,1) -- we will poll the socket
    local _allSubscribers={}
    local _allDedicatedPublishers={}
    local _setupSubscribersAsynchronously=setupSubscribersAsynchronously
    local _pongReceived=false

    function _pingCallback(msg)
        _pongReceived=true
    end
        
    function self.delete()
        print('*************************************************************************************')
        print('** Leaving... if this is unexpected, you might have to adjust the timeout argument **')
        print('*************************************************************************************')
        _pongReceived=false
        _handleFunction('Ping',{0},self.simxDefaultSubscriber(_pingCallback))
        while not _pongReceived do
            self.simxSpinOnce()
        end
        _handleFunction('DisconnectClient',{_clientId},_serviceCallTopic)
        for key,value in pairs(_allSubscribers) do 
            if value.handle~=_defaultSubscriber then
                b0.subscriber_delete(value.handle)
            end
        end
        for key,value in pairs(_allDedicatedPublishers) do
            b0.publisher_delete(value)
        end
--        b0.node_delete(_node)
    end
    
    function _handleFunction(funcName,reqArgs,topic)
        if topic==_serviceCallTopic then
            local packedData=b0.messagePack.pack({{funcName,_clientId,topic,0},reqArgs})
            local repl=b0.messagePack.unpack(b0.service_client_call(_serviceClient,packedData))
            return repl
        elseif topic==_defaultPublisherTopic then
            local packedData=b0.messagePack.pack({{funcName,_clientId,topic,1},reqArgs})
            b0.publisher_publish(_defaultPublisher,packedData)
        elseif _allSubscribers[topic] then
            if _allSubscribers[topic].handle==_defaultSubscriber then
                local packedData=b0.messagePack.pack({{funcName,_clientId,topic,2},reqArgs})
                if _setupSubscribersAsynchronously then
                    b0.publisher_publisher(_defaultPublisher,packedData)
                else
                    b0.service_client_call(_serviceClient,packedData)
                end
            else
                local packedData=b0.messagePack.pack({{funcName,_clientId,topic,4},reqArgs})
                if _setupSubscribersAsynchronously then
                    b0.publisher_publish(_defaultPublisher,packedData)
                else
                    b0.service_client_call(_serviceClient,packedData)
                end
            end
        elseif _allDedicatedPublishers[topic] then
            local packedData=b0.messagePack.pack({{funcName,_clientId,topic,3},reqArgs})
            b0.publisher_publish(_allDedicatedPublishers[topic],packedData)
        else
            print('B0 Remote API error: invalid topic')
        end
    end
    
    function self.simxDefaultPublisher()
        return _defaultPublisherTopic
    end

    function self.simxCreatePublisher(dropMessages)
        if dropMessages==nil then dropMessages=false end
        local topic=_channelName..'Sub'..tostring(_nextDedicatedPublisherHandle).._clientId
        _nextDedicatedPublisherHandle=_nextDedicatedPublisherHandle+1
        local pub=b0.publisher_new_ex(_node,topic,0,1)
        b0.publisher_init(pub)
        _allDedicatedPublishers[topic]=pub
        _handleFunction('createSubscriber',{topic,dropMessages},_serviceCallTopic)
        return topic
    end

    function self.simxDefaultSubscriber(cb,publishInterval)
        if publishInterval==nil then publishInterval=1 end
        local topic=_channelName..'Pub'..tostring(_nextDefaultSubscriberHandle).._clientId
        _nextDefaultSubscriberHandle=_nextDefaultSubscriberHandle+1
        _allSubscribers[topic]={}
        _allSubscribers[topic].handle=_defaultSubscriber
        _allSubscribers[topic].cb=cb
        _allSubscribers[topic].dropMessages=false
        local channel=_serviceCallTopic
        if _setupSubscribersAsynchronously then
            channel=_defaultPublisherTopic
        end
        _handleFunction('setDefaultPublisherPubInterval',{topic,publishInterval},channel)
        return topic
    end
        
    function self.simxCreateSubscriber(cb,publishInterval,dropMessages)
        if publishInterval==nil then publishInterval=1 end
        if dropMessages==nil then dropMessages=false end
        local topic=_channelName..'Pub'..tostring(_nextDedicatedSubscriberHandle).._clientId
        _nextDedicatedSubscriberHandle=_nextDedicatedSubscriberHandle+1
        local subb=b0.subscriber_new_ex(_node,topic,0,1)
        if dropMessages then
            b0.subscriber_set_option(subb,b0.SOCK_OPT_CONFLATE,1)
        else
            b0.subscriber_set_option(subb,b0.SOCK_OPT_CONFLATE,0)
        end
        b0.subscriber_init(subb)
        _allSubscribers[topic]={}
        _allSubscribers[topic].handle=subb
        _allSubscribers[topic].cb=cb
        _allSubscribers[topic].dropMessages=dropMessages
        local channel=_serviceCallTopic
        if _setupSubscribersAsynchronously then
            channel=_defaultPublisherTopic
        end
        _handleFunction('createPublisher',{topic,publishInterval},channel)
        return topic
    end
  
    function self.simxRemoveSubscriber(topic)
        val=_allSubscribers[topic]
        if val then
            local channel=_serviceCallTopic
            if _setupSubscribersAsynchronously then
                channel=_defaultPublisherTopic
            end
            if val.handle==_defaultSubscriber then
                _handleFunction('stopDefaultPublisher',{topic},channel)
            else
                b0.subscriber_delete(val.handle)
                _handleFunction('stopPublisher',{topic},channel)
            end
            _allSubscribers[topic]=nil
        end
    end
    
    function self.simxRemovePublisher(topic)
        val=_allDedicatedPublishers[topic]
        if val then
            b0.publisher_delete(val)
            _handleFunction('stopSubscriber',{topic},_serviceCallTopic)
            _allDedicatedPublishers[topic]=nil
        end
    end
    
    function self.simxServiceCall()
        return _serviceCallTopic
    end

    function _handleReceivedMessage(msg)
        msg=b0.messagePack.unpack(msg)
        if _allSubscribers[msg[1]] then
            local cbMsg=msg[2]
            _allSubscribers[msg[1]].cb(cbMsg)
        end
    end
        
    function self.simxSpinOnce()
        local defaultSubscriberAlreadyProcessed=false
        for key,value in pairs(_allSubscribers) do
            local readData=nil
            if (value.handle~=_defaultSubscriber) or (not defaultSubscriberAlreadyProcessed) then
                defaultSubscriberAlreadyProcessed=defaultSubscriberAlreadyProcessed or (value.handle==_defaultSubscriber)
                while b0.subscriber_poll(value.handle,0)>0 do
                    readData=b0.subscriber_read(value.handle)
                    if not value.dropMessages then
                        _handleReceivedMessage(readData)
                    end
                end
                if value.dropMessages and readData then
                    _handleReceivedMessage(readData)
                end
            end
        end
    end
                    
    function self.simxSpin()
        while true do
            self.simxSpinOnce()
        end
    end

    function self.simxSynchronous(enable)
        local reqArgs = {enable}
        local funcName = 'Synchronous'
        _handleFunction(funcName,reqArgs,_serviceCallTopic)
    end
        
    function self.simxSynchronousTrigger()
        local reqArgs = {0}
        local funcName = 'SynchronousTrigger'
        _handleFunction(funcName,reqArgs,_defaultPublisherTopic)
    end
        
    function self.simxGetSimulationStepDone(topic)
        if _allSubscribers[topic] then
            local reqArgs = {0}
            local funcName = 'GetSimulationStepDone'
            _handleFunction(funcName,reqArgs,topic)
        else
            print('B0 Remote API error: invalid topic')
        end
    end
        
    function self.simxGetSimulationStepStarted(topic)
        if _allSubscribers[topic] then
            local reqArgs = {0}
            local funcName = 'GetSimulationStepStarted'
            _handleFunction(funcName,reqArgs,topic)
        else
            print('B0 Remote API error: invalid topic')
        end
    end
    
    function self.simxGetTimeInMs()
        return b0.node_hardware_time_usec(_node)/1000    
    end
    
    function self.simxSleep(durationInMs)
        local st=self.simxGetTimeInMs()
        while self.simxGetTimeInMs()-st<durationInMs do end
    end
    
    print('\n  Running B0 Remote API client with channel name ['..channelName..']')
    print('  make sure that: 1) the B0 resolver is running')
    print('                  2) CoppeliaSim is running the B0 Remote API server with the same channel name')
    print('  Initializing...\n')
    b0.node_init(_node)
    
    _handleFunction('inactivityTolerance',{inactivityToleranceInSec},_serviceCallTopic)
    print('\n  Connected!\n')

    function self.simxCallScriptFunction(funcAtObjName,scriptType,arg,topic)
        local packedArg=b0.messagePack.pack(arg)
        local reqArgs = {funcAtObjName,scriptType,packedArg}
        local funcName = 'CallScriptFunction'
        return _handleFunction(funcName,reqArgs,topic)
    end
    
    function self.simxGetObjectHandle(objectName,topic)
        local reqArgs = {objectName}
        return _handleFunction("GetObjectHandle",reqArgs,topic)
    end
    function self.simxAddStatusbarMessage(msg,topic)
        local reqArgs = {msg}
        return _handleFunction("AddStatusbarMessage",reqArgs,topic)
    end
    function self.simxGetObjectPosition(objectHandle,relObjHandle,topic)
        local reqArgs = {objectHandle,relObjHandle}
        return _handleFunction("GetObjectPosition",reqArgs,topic)
    end
    function self.simxGetObjectOrientation(objectHandle,relObjHandle,topic)
        local reqArgs = {objectHandle,relObjHandle}
        return _handleFunction("GetObjectOrientation",reqArgs,topic)
    end
    function self.simxGetObjectQuaternion(objectHandle,relObjHandle,topic)
        local reqArgs = {objectHandle,relObjHandle}
        return _handleFunction("GetObjectQuaternion",reqArgs,topic)
    end
    function self.simxGetObjectPose(objectHandle,relObjHandle,topic)
        local reqArgs = {objectHandle,relObjHandle}
        return _handleFunction("GetObjectPose",reqArgs,topic)
    end
    function self.simxGetObjectMatrix(objectHandle,relObjHandle,topic)
        local reqArgs = {objectHandle,relObjHandle}
        return _handleFunction("GetObjectMatrix",reqArgs,topic)
    end
    function self.simxSetObjectPosition(objectHandle,relObjHandle,position,topic)
        local reqArgs = {objectHandle,relObjHandle,position}
        return _handleFunction("SetObjectPosition",reqArgs,topic)
    end
    function self.simxSetObjectOrientation(objectHandle,relObjHandle,euler,topic)
        local reqArgs = {objectHandle,relObjHandle,euler}
        return _handleFunction("SetObjectOrientation",reqArgs,topic)
    end
    function self.simxSetObjectQuaternion(objectHandle,relObjHandle,quat,topic)
        local reqArgs = {objectHandle,relObjHandle,quat}
        return _handleFunction("SetObjectQuaternion",reqArgs,topic)
    end
    function self.simxSetObjectPose(objectHandle,relObjHandle,pose,topic)
        local reqArgs = {objectHandle,relObjHandle,pose}
        return _handleFunction("SetObjectPose",reqArgs,topic)
    end
    function self.simxSetObjectMatrix(objectHandle,relObjHandle,matr,topic)
        local reqArgs = {objectHandle,relObjHandle,matr}
        return _handleFunction("SetObjectMatrix",reqArgs,topic)
    end
    function self.simxClearFloatSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("ClearFloatSignal",reqArgs,topic)
    end
    function self.simxClearIntegerSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("ClearIntegerSignal",reqArgs,topic)
    end
    function self.simxClearStringSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("ClearStringSignal",reqArgs,topic)
    end
    function self.simxSetFloatSignal(sigName,sigValue,topic)
        local reqArgs = {sigName,sigValue}
        return _handleFunction("SetFloatSignal",reqArgs,topic)
    end
    function self.simxSetIntSignal(sigName,sigValue,topic)
        local reqArgs = {sigName,sigValue}
        return _handleFunction("SetIntSignal",reqArgs,topic)
    end
    function self.simxSetStringSignal(sigName,sigValue,topic)
        local reqArgs = {sigName,sigValue}
        return _handleFunction("SetStringSignal",reqArgs,topic)
    end
    function self.simxGetFloatSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("GetFloatSignal",reqArgs,topic)
    end
    function self.simxGetIntSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("GetIntSignal",reqArgs,topic)
    end
    function self.simxGetStringSignal(sigName,topic)
        local reqArgs = {sigName}
        return _handleFunction("GetStringSignal",reqArgs,topic)
    end
    function self.simxAuxiliaryConsoleClose(consoleHandle,topic)
        local reqArgs = {consoleHandle}
        return _handleFunction("AuxiliaryConsoleClose",reqArgs,topic)
    end
    function self.simxAuxiliaryConsolePrint(consoleHandle,text,topic)
        local reqArgs = {consoleHandle,text}
        return _handleFunction("AuxiliaryConsolePrint",reqArgs,topic)
    end
    function self.simxAuxiliaryConsoleShow(consoleHandle,showState,topic)
        local reqArgs = {consoleHandle,showState}
        return _handleFunction("AuxiliaryConsoleShow",reqArgs,topic)
    end
    function self.simxAuxiliaryConsoleOpen(title,maxLines,mode,position,size,textColor,backgroundColor,topic)
        local reqArgs = {title,maxLines,mode,position,size,textColor,backgroundColor}
        return _handleFunction("AuxiliaryConsoleOpen",reqArgs,topic)
    end
    function self.simxStartSimulation(topic)
        local reqArgs = {0}
        return _handleFunction("StartSimulation",reqArgs,topic)
    end
    function self.simxStopSimulation(topic)
        local reqArgs = {0}
        return _handleFunction("StopSimulation",reqArgs,topic)
    end
    function self.simxPauseSimulation(topic)
        local reqArgs = {0}
        return _handleFunction("PauseSimulation",reqArgs,topic)
    end
    function self.simxGetVisionSensorImage(objectHandle,greyScale,topic)
        local reqArgs = {objectHandle,greyScale}
        return _handleFunction("GetVisionSensorImage",reqArgs,topic)
    end
    function self.simxSetVisionSensorImage(objectHandle,greyScale,img,topic)
        local reqArgs = {objectHandle,greyScale,img}
        return _handleFunction("SetVisionSensorImage",reqArgs,topic)
    end
    function self.simxGetVisionSensorDepthBuffer(objectHandle,toMeters,asByteArray,topic)
        local reqArgs = {objectHandle,toMeters,asByteArray}
        return _handleFunction("GetVisionSensorDepthBuffer",reqArgs,topic)
    end
    function self.simxAddDrawingObject_points(size,color,coords,topic)
        local reqArgs = {size,color,coords}
        return _handleFunction("AddDrawingObject_points",reqArgs,topic)
    end
    function self.simxAddDrawingObject_spheres(size,color,coords,topic)
        local reqArgs = {size,color,coords}
        return _handleFunction("AddDrawingObject_spheres",reqArgs,topic)
    end
    function self.simxAddDrawingObject_cubes(size,color,coords,topic)
        local reqArgs = {size,color,coords}
        return _handleFunction("AddDrawingObject_cubes",reqArgs,topic)
    end
    function self.simxAddDrawingObject_segments(lineSize,color,segments,topic)
        local reqArgs = {lineSize,color,segments}
        return _handleFunction("AddDrawingObject_segments",reqArgs,topic)
    end
    function self.simxAddDrawingObject_triangles(color,triangles,topic)
        local reqArgs = {color,triangles}
        return _handleFunction("AddDrawingObject_triangles",reqArgs,topic)
    end
    function self.simxRemoveDrawingObject(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("RemoveDrawingObject",reqArgs,topic)
    end
    function self.simxGetCollisionHandle(nameOfObject,topic)
        local reqArgs = {nameOfObject}
        return _handleFunction("GetCollisionHandle",reqArgs,topic)
    end
    function self.simxGetDistanceHandle(nameOfObject,topic)
        local reqArgs = {nameOfObject}
        return _handleFunction("GetDistanceHandle",reqArgs,topic)
    end
    function self.simxReadCollision(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("ReadCollision",reqArgs,topic)
    end
    function self.simxReadDistance(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("ReadDistance",reqArgs,topic)
    end
    function self.simxCheckCollision(entity1,entity2,topic)
        local reqArgs = {entity1,entity2}
        return _handleFunction("CheckCollision",reqArgs,topic)
    end
    function self.simxCheckDistance(entity1,entity2,threshold,topic)
        local reqArgs = {entity1,entity2,threshold}
        return _handleFunction("CheckDistance",reqArgs,topic)
    end
    function self.simxReadProximitySensor(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("ReadProximitySensor",reqArgs,topic)
    end
    function self.simxCheckProximitySensor(handle,entity,topic)
        local reqArgs = {handle,entity}
        return _handleFunction("CheckProximitySensor",reqArgs,topic)
    end
    function self.simxReadForceSensor(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("ReadForceSensor",reqArgs,topic)
    end
    function self.simxBreakForceSensor(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("BreakForceSensor",reqArgs,topic)
    end
    function self.simxReadVisionSensor(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("ReadVisionSensor",reqArgs,topic)
    end
    function self.simxCheckVisionSensor(handle,entity,topic)
        local reqArgs = {handle,entity}
        return _handleFunction("CheckVisionSensor",reqArgs,topic)
    end
    function self.simxCopyPasteObjects(objectHandles,options,topic)
        local reqArgs = {objectHandles,options}
        return _handleFunction("CopyPasteObjects",reqArgs,topic)
    end
    function self.simxRemoveObjects(objectHandles,options,topic)
        local reqArgs = {objectHandles,options}
        return _handleFunction("RemoveObjects",reqArgs,topic)
    end
    function self.simxCloseScene(topic)
        local reqArgs = {0}
        return _handleFunction("CloseScene",reqArgs,topic)
    end
    function self.simxSetStringParameter(paramId,paramVal,topic)
        local reqArgs = {paramId,paramVal}
        return _handleFunction("SetStringParameter",reqArgs,topic)
    end
    function self.simxSetFloatParameter(paramId,paramVal,topic)
        local reqArgs = {paramId,paramVal}
        return _handleFunction("SetFloatParameter",reqArgs,topic)
    end
    function self.simxSetArrayParameter(paramId,paramVal,topic)
        local reqArgs = {paramId,paramVal}
        return _handleFunction("SetArrayParameter",reqArgs,topic)
    end
    function self.simxSetIntParameter(paramId,paramVal,topic)
        local reqArgs = {paramId,paramVal}
        return _handleFunction("SetIntParameter",reqArgs,topic)
    end
    function self.simxSetBoolParameter(paramId,paramVal,topic)
        local reqArgs = {paramId,paramVal}
        return _handleFunction("SetBoolParameter",reqArgs,topic)
    end
    function self.simxGetStringParameter(paramId,topic)
        local reqArgs = {paramId}
        return _handleFunction("GetStringParameter",reqArgs,topic)
    end
    function self.simxGetFloatParameter(paramId,topic)
        local reqArgs = {paramId}
        return _handleFunction("GetFloatParameter",reqArgs,topic)
    end
    function self.simxGetArrayParameter(paramId,topic)
        local reqArgs = {paramId}
        return _handleFunction("GetArrayParameter",reqArgs,topic)
    end
    function self.simxGetIntParameter(paramId,topic)
        local reqArgs = {paramId}
        return _handleFunction("GetIntParameter",reqArgs,topic)
    end
    function self.simxGetBoolParameter(paramId,topic)
        local reqArgs = {paramId}
        return _handleFunction("GetBoolParameter",reqArgs,topic)
    end
    function self.simxDisplayDialog(titleText,mainText,dialogType,inputText,topic)
        local reqArgs = {titleText,mainText,dialogType,inputText}
        return _handleFunction("DisplayDialog",reqArgs,topic)
    end
    function self.simxGetDialogResult(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("GetDialogResult",reqArgs,topic)
    end
    function self.simxGetDialogInput(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("GetDialogInput",reqArgs,topic)
    end
    function self.simxEndDialog(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("EndDialog",reqArgs,topic)
    end
    function self.simxExecuteScriptString(code,topic)
        local reqArgs = {code}
        return _handleFunction("ExecuteScriptString",reqArgs,topic)
    end
    function self.simxGetCollectionHandle(collectionName,topic)
        local reqArgs = {collectionName}
        return _handleFunction("GetCollectionHandle",reqArgs,topic)
    end
    function self.simxGetJointForce(jointHandle,topic)
        local reqArgs = {jointHandle}
        return _handleFunction("GetJointForce",reqArgs,topic)
    end
    function self.simxGetJointMaxForce(jointHandle,topic)
        local reqArgs = {jointHandle}
        return _handleFunction("GetJointMaxForce",reqArgs,topic)
    end
    function self.simxSetJointForce(jointHandle,forceOrTorque,topic)
        local reqArgs = {jointHandle,forceOrTorque}
        return _handleFunction("SetJointForce",reqArgs,topic)
    end
    function self.simxSetJointMaxForce(jointHandle,forceOrTorque,topic)
        local reqArgs = {jointHandle,forceOrTorque}
        return _handleFunction("SetJointMaxForce",reqArgs,topic)
    end
    function self.simxGetJointPosition(jointHandle,topic)
        local reqArgs = {jointHandle}
        return _handleFunction("GetJointPosition",reqArgs,topic)
    end
    function self.simxSetJointPosition(jointHandle,position,topic)
        local reqArgs = {jointHandle,position}
        return _handleFunction("SetJointPosition",reqArgs,topic)
    end
    function self.simxGetJointTargetPosition(jointHandle,topic)
        local reqArgs = {jointHandle}
        return _handleFunction("GetJointTargetPosition",reqArgs,topic)
    end
    function self.simxSetJointTargetPosition(jointHandle,targetPos,topic)
        local reqArgs = {jointHandle,targetPos}
        return _handleFunction("SetJointTargetPosition",reqArgs,topic)
    end
    function self.simxGetJointTargetVelocity(jointHandle,topic)
        local reqArgs = {jointHandle}
        return _handleFunction("GetJointTargetVelocity",reqArgs,topic)
    end
    function self.simxSetJointTargetVelocity(jointHandle,targetPos,topic)
        local reqArgs = {jointHandle,targetPos}
        return _handleFunction("SetJointTargetVelocity",reqArgs,topic)
    end
    function self.simxGetObjectChild(objectHandle,index,topic)
        local reqArgs = {objectHandle,index}
        return _handleFunction("GetObjectChild",reqArgs,topic)
    end
    function self.simxGetObjectParent(objectHandle,topic)
        local reqArgs = {objectHandle}
        return _handleFunction("GetObjectParent",reqArgs,topic)
    end
    function self.simxSetObjectParent(objectHandle,parentHandle,assembly,keepInPlace,topic)
        local reqArgs = {objectHandle,parentHandle,assembly,keepInPlace}
        return _handleFunction("SetObjectParent",reqArgs,topic)
    end
    function self.simxGetObjectsInTree(treeBaseHandle,objectType,options,topic)
        local reqArgs = {treeBaseHandle,objectType,options}
        return _handleFunction("GetObjectsInTree",reqArgs,topic)
    end
    function self.simxGetObjectName(objectHandle,altName,topic)
        local reqArgs = {objectHandle,altName}
        return _handleFunction("GetObjectName",reqArgs,topic)
    end
    function self.simxGetObjectFloatParameter(objectHandle,parameterID,topic)
        local reqArgs = {objectHandle,parameterID}
        return _handleFunction("GetObjectFloatParameter",reqArgs,topic)
    end
    function self.simxGetObjectIntParameter(objectHandle,parameterID,topic)
        local reqArgs = {objectHandle,parameterID}
        return _handleFunction("GetObjectIntParameter",reqArgs,topic)
    end
    function self.simxGetObjectStringParameter(objectHandle,parameterID,topic)
        local reqArgs = {objectHandle,parameterID}
        return _handleFunction("GetObjectStringParameter",reqArgs,topic)
    end
    function self.simxSetObjectFloatParameter(objectHandle,parameterID,parameter,topic)
        local reqArgs = {objectHandle,parameterID,parameter}
        return _handleFunction("SetObjectFloatParameter",reqArgs,topic)
    end
    function self.simxSetObjectIntParameter(objectHandle,parameterID,parameter,topic)
        local reqArgs = {objectHandle,parameterID,parameter}
        return _handleFunction("SetObjectIntParameter",reqArgs,topic)
    end
    function self.simxSetObjectStringParameter(objectHandle,parameterID,parameter,topic)
        local reqArgs = {objectHandle,parameterID,parameter}
        return _handleFunction("SetObjectStringParameter",reqArgs,topic)
    end
    function self.simxGetSimulationTime(topic)
        local reqArgs = {0}
        return _handleFunction("GetSimulationTime",reqArgs,topic)
    end
    function self.simxGetSimulationTimeStep(topic)
        local reqArgs = {0}
        return _handleFunction("GetSimulationTimeStep",reqArgs,topic)
    end
    function self.simxGetServerTimeInMs(topic)
        local reqArgs = {0}
        return _handleFunction("GetServerTimeInMs",reqArgs,topic)
    end
    function self.simxGetSimulationState(topic)
        local reqArgs = {0}
        return _handleFunction("GetSimulationState",reqArgs,topic)
    end
    function self.simxEvaluateToInt(str,topic)
        local reqArgs = {str}
        return _handleFunction("EvaluateToInt",reqArgs,topic)
    end
    function self.simxEvaluateToStr(str,topic)
        local reqArgs = {str}
        return _handleFunction("EvaluateToStr",reqArgs,topic)
    end
    function self.simxGetObjects(objectType,topic)
        local reqArgs = {objectType}
        return _handleFunction("GetObjects",reqArgs,topic)
    end
    function self.simxCreateDummy(size,color,topic)
        local reqArgs = {size,color}
        return _handleFunction("CreateDummy",reqArgs,topic)
    end
    function self.simxGetObjectSelection(topic)
        local reqArgs = {0}
        return _handleFunction("GetObjectSelection",reqArgs,topic)
    end
    function self.simxSetObjectSelection(selection,topic)
        local reqArgs = {selection}
        return _handleFunction("SetObjectSelection",reqArgs,topic)
    end
    function self.simxGetObjectVelocity(handle,topic)
        local reqArgs = {handle}
        return _handleFunction("GetObjectVelocity",reqArgs,topic)
    end
    function self.simxLoadModelFromFile(filename,topic)
        local reqArgs = {filename}
        return _handleFunction("LoadModelFromFile",reqArgs,topic)
    end
    function self.simxLoadModelFromBuffer(buffer,topic)
        local reqArgs = {buffer}
        return _handleFunction("LoadModelFromBuffer",reqArgs,topic)
    end
    function self.simxLoadScene(filename,topic)
        local reqArgs = {filename}
        return _handleFunction("LoadScene",reqArgs,topic)
    end

    -- -----------------------------------------------------------
    -- Add your custom functions here, or even better,
    -- add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    -- and generate this file again.
    -- Then add the server part of your custom functions at the
    -- beginning of file lua/b0RemoteApiServer.lua
    -- -----------------------------------------------------------
    
    return self
end
