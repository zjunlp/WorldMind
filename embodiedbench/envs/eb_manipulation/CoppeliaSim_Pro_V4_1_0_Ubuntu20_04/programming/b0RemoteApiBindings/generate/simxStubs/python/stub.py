#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
# -------------------------------------------------------
# Add your custom functions at the bottom of the file
# and the server counterpart to lua/b0RemoteApiServer.lua
# -------------------------------------------------------

import b0
import msgpack
import random
import string
import time

class RemoteApiClient:
    def __init__(self,nodeName='b0RemoteApi_pythonClient',channelName='b0RemoteApi',inactivityToleranceInSec=60,setupSubscribersAsynchronously=False,timeout=3):
        self._channelName=channelName
        self._serviceCallTopic=channelName+'SerX'
        self._defaultPublisherTopic=channelName+'SubX'
        self._defaultSubscriberTopic=channelName+'PubX'
        self._nextDefaultSubscriberHandle=2
        self._nextDedicatedPublisherHandle=500
        self._nextDedicatedSubscriberHandle=1000
        b0.init()
        self._node=b0.Node(nodeName)
        self._clientId=''.join(random.choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(10))
        self._serviceClient=b0.ServiceClient(self._node,self._serviceCallTopic)
        self._serviceClient.set_option(3,timeout*1000) #read timeout
        self._defaultPublisher=b0.Publisher(self._node,self._defaultPublisherTopic)
        self._defaultSubscriber=b0.Subscriber(self._node,self._defaultSubscriberTopic,None) # we will poll the socket
        print('\n  Running B0 Remote API client with channel name ['+channelName+']')
        print('  make sure that: 1) the B0 resolver is running')
        print('                  2) CoppeliaSim is running the B0 Remote API server with the same channel name')
        print('  Initializing...\n')
        self._node.init()
        self._handleFunction('inactivityTolerance',[inactivityToleranceInSec],self._serviceCallTopic)
        print('\n  Connected!\n')
        self._allSubscribers={}
        self._allDedicatedPublishers={}
        self._setupSubscribersAsynchronously=setupSubscribersAsynchronously
  
    def __enter__(self):
        return self
    
    def __exit__(self,*err):
        print('*************************************************************************************')
        print('** Leaving... if this is unexpected, you might have to adjust the timeout argument **')
        print('*************************************************************************************')
        self._pongReceived=False
        self._handleFunction('Ping',[0],self.simxDefaultSubscriber(self._pingCallback))
        while not self._pongReceived:
            self.simxSpinOnce();
        self._handleFunction('DisconnectClient',[self._clientId],self._serviceCallTopic)
        for key, value in self._allSubscribers.items():
            if value['handle']!=self._defaultSubscriber:
                value['handle'].cleanup()
        for key, value in self._allDedicatedPublishers.items():
            value.cleanup()
        self._node.cleanup()
        
    def _pingCallback(self,msg):
        self._pongReceived=True
        
    def _handleReceivedMessage(self,msg):
        msg=msgpack.unpackb(msg,raw=True)
        msg[0]=msg[0].decode('ascii')
        if msg[0] in self._allSubscribers:
            cbMsg=msg[1]
            if len(cbMsg)==1:
                cbMsg.append(None)
            self._allSubscribers[msg[0]]['cb'](cbMsg)
            
    def _handleFunction(self,funcName,reqArgs,topic):
        if topic==self._serviceCallTopic:
            packedData=msgpack.packb([[funcName,self._clientId,topic,0],reqArgs])
            rep = msgpack.unpackb(self._serviceClient.call(packedData),raw=True)
            if len(rep)==1:
                rep.append(None)
            return rep
        elif topic==self._defaultPublisherTopic:
            packedData=msgpack.packb([[funcName,self._clientId,topic,1],reqArgs])
            self._defaultPublisher.publish(packedData)
        elif topic in self._allSubscribers:
            if self._allSubscribers[topic]['handle']==self._defaultSubscriber:
                packedData=msgpack.packb([[funcName,self._clientId,topic,2],reqArgs])
                if self._setupSubscribersAsynchronously:
                    self._defaultPublisher.publish(packedData)
                else:
                    self._serviceClient.call(packedData)
            else:
                packedData=msgpack.packb([[funcName,self._clientId,topic,4],reqArgs])
                if self._setupSubscribersAsynchronously:
                    self._defaultPublisher.publish(packedData)
                else:
                    self._serviceClient.call(packedData)
        elif topic in self._allDedicatedPublishers:
            packedData=msgpack.packb([[funcName,self._clientId,topic,3],reqArgs])
            self._allDedicatedPublishers[topic].publish(packedData)
        else:
            print('B0 Remote API error: invalid topic')
        
    def simxDefaultPublisher(self):
        return self._defaultPublisherTopic

    def simxCreatePublisher(self,dropMessages=False):
        topic=self._channelName+'Sub'+str(self._nextDedicatedPublisherHandle)+self._clientId
        self._nextDedicatedPublisherHandle=self._nextDedicatedPublisherHandle+1
        pub=b0.Publisher(self._node,topic,0,1)
        pub.init()
        self._allDedicatedPublishers[topic]=pub
        self._handleFunction('createSubscriber',[topic,dropMessages],self._serviceCallTopic)
        return topic

    def simxDefaultSubscriber(self,cb,publishInterval=1):
        topic=self._channelName+'Pub'+str(self._nextDefaultSubscriberHandle)+self._clientId
        self._nextDefaultSubscriberHandle=self._nextDefaultSubscriberHandle+1
        self._allSubscribers[topic]={}
        self._allSubscribers[topic]['handle']=self._defaultSubscriber
        self._allSubscribers[topic]['cb']=cb
        self._allSubscribers[topic]['dropMessages']=False
        channel=self._serviceCallTopic
        if self._setupSubscribersAsynchronously:
            channel=self._defaultPublisherTopic
        self._handleFunction('setDefaultPublisherPubInterval',[topic,publishInterval],channel)
        return topic
        
    def simxCreateSubscriber(self,cb,publishInterval=1,dropMessages=False):
        topic=self._channelName+'Pub'+str(self._nextDedicatedSubscriberHandle)+self._clientId
        self._nextDedicatedSubscriberHandle=self._nextDedicatedSubscriberHandle+1
        sub=b0.Subscriber(self._node,topic,None,0,1)
        if dropMessages:
            sub.set_option(6,1) #conflate option enabled
        else:
            sub.set_option(6,0) #conflate option disabled
        sub.init()
        self._allSubscribers[topic]={}
        self._allSubscribers[topic]['handle']=sub
        self._allSubscribers[topic]['cb']=cb
        self._allSubscribers[topic]['dropMessages']=dropMessages
        channel=self._serviceCallTopic
        if self._setupSubscribersAsynchronously:
            channel=self._defaultPublisherTopic
        self._handleFunction('createPublisher',[topic,publishInterval],channel)
        return topic
  
    def simxRemoveSubscriber(self,topic):
        if topic in self._allSubscribers:
            value=self._allSubscribers[topic]
            channel=self._serviceCallTopic
            if self._setupSubscribersAsynchronously:
                channel=self._defaultPublisherTopic
            if value['handle']==self._defaultSubscriber:
                self._handleFunction('stopDefaultPublisher',[topic],channel)
            else:
                value['handle'].cleanup()
                self._handleFunction('stopPublisher',[topic],channel)
            del self._allSubscribers[topic]

    def simxRemovePublisher(self,topic):
        if topic in self._allDedicatedPublishers:
            value=self._allDedicatedPublishers[topic]
            value.cleanup()
            self._handleFunction('stopSubscriber',[topic],self._serviceCallTopic)
            del self._allDedicatedPublishers[topic]
        
    def simxServiceCall(self):
        return self._serviceCallTopic
        
    def simxSpin(self):
        while True:
            self.simxSpinOnce()
        
    def simxSpinOnce(self):
        defaultSubscriberAlreadyProcessed=False
        for key, value in self._allSubscribers.items():
            readData=None
            if (value['handle']!=self._defaultSubscriber) or (not defaultSubscriberAlreadyProcessed):
                defaultSubscriberAlreadyProcessed=defaultSubscriberAlreadyProcessed or (value['handle']==self._defaultSubscriber)
                while value['handle'].poll(0):
                    readData=value['handle'].read()
                    if not value['dropMessages']:
                        self._handleReceivedMessage(readData)
                if value['dropMessages'] and (readData is not None):
                    self._handleReceivedMessage(readData)
                    
    def simxGetTimeInMs(self):
        return self._node.hardware_time_usec()/1000;    

    def simxSleep(self,durationInMs):
        time.sleep(durationInMs)
        
    def simxSynchronous(self,enable):
        reqArgs = [enable]
        funcName = 'Synchronous'
        self._handleFunction(funcName,reqArgs,self._serviceCallTopic)
        
    def simxSynchronousTrigger(self):
        reqArgs = [0]
        funcName = 'SynchronousTrigger'
        self._handleFunction(funcName,reqArgs,self._defaultPublisherTopic)
        
    def simxGetSimulationStepDone(self,topic):
        if topic in self._allSubscribers:
            reqArgs = [0]
            funcName = 'GetSimulationStepDone'
            self._handleFunction(funcName,reqArgs,topic)
        else:
            print('B0 Remote API error: invalid topic')
        
    def simxGetSimulationStepStarted(self,topic):
        if topic in self._allSubscribers:
            reqArgs = [0]
            funcName = 'GetSimulationStepStarted'
            self._handleFunction(funcName,reqArgs,topic)
        else:
            print('B0 Remote API error: invalid topic')
    
    def simxCallScriptFunction(self,funcAtObjName,scriptType,arg,topic):
        packedArg=msgpack.packb(arg)
        reqArgs = [funcAtObjName,scriptType,packedArg]
        funcName = 'CallScriptFunction'
        return self._handleFunction(funcName,reqArgs,topic)

        
#py for cmd in plugin.commands:
#py if cmd.generic and cmd.generateCode:
#py theStringToWrite='    def '+cmd.name+'(self,'
#py itemCnt=len(cmd.params)
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py theStringToWrite+=p.name
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=','
#py endif
#py endfor
`theStringToWrite`):
#py if len(cmd.params)==1:
#py theStringToWrite='        reqArgs = [0'
#py else:
#py itemCnt=len(cmd.params)-1
#py itemIndex=-1
#py theStringToWrite='        reqArgs = ['
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
#py theStringToWrite+=']'
`theStringToWrite`
        return self._handleFunction('`cmd.name[4:]`',reqArgs,topic)
#py endif
#py endfor

    # -----------------------------------------------------------
    # Add your custom functions here, or even better,
    # add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    # and generate this file again.
    # Then add the server part of your custom functions at the
    # beginning of file lua/b0RemoteApiServer.lua
    # -----------------------------------------------------------
