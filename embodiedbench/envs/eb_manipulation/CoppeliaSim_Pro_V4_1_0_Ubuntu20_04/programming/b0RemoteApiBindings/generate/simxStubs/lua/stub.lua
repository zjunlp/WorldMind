#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
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
    
#py for cmd in plugin.commands:
#py if cmd.generic and cmd.generateCode:
#py theStringToWrite='    function self.'+cmd.name+'('
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
#py theStringToWrite='        local reqArgs = {0'
#py else:
#py itemCnt=len(cmd.params)-1
#py itemIndex=-1
#py theStringToWrite='        local reqArgs = {'
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
#py theStringToWrite+='}'
`theStringToWrite`
        return _handleFunction("`cmd.name[4:]`",reqArgs,topic)
    end
#py endif
#py endfor

    -- -----------------------------------------------------------
    -- Add your custom functions here, or even better,
    -- add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    -- and generate this file again.
    -- Then add the server part of your custom functions at the
    -- beginning of file lua/b0RemoteApiServer.lua
    -- -----------------------------------------------------------
    
    return self
end
