#py from parse import parse
#py import model
#py plugin = parse(pycpp.params['xml_file'])
// -------------------------------------------------------
// Add your custom functions at the bottom of the file
// and the server counterpart to lua/b0RemoteApiServer.lua
// -------------------------------------------------------

package coppelia;

import org.msgpack.core.MessagePack;
import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessageUnpacker;
import org.msgpack.value.*;

import java.util.Random;
import java.util.HashMap;
import java.util.Map;
import java.util.function.*;

import java.io.IOException;

public class b0RemoteApi
{
    class SHandleAndCb
    {
        long handle;
        boolean dropMessages;
        Consumer<MessageUnpacker> cb;
    }
    class SHandle
    {
        long handle;
    }

    static{
        System.loadLibrary("b0");
    }
    public static void print(final MessageUnpacker msg) throws IOException 
    {
        int cnt=0;
        while (msg.hasNext())
        {
            if (cnt>0)
                System.out.printf(", ");
            System.out.print(msg.unpackValue());
            cnt=cnt+1;
        }
        if (cnt>0)
            System.out.printf("\n");
    }
    
    public static boolean hasValue(final MessageUnpacker msg) throws IOException 
    {
        return msg.hasNext();
    }
    public static Value readValue(final MessageUnpacker msg) throws IOException
    {
        return readValue(msg,0);
    }
    public static Value readValue(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        while (valuesToDiscard>0)
        {
            msg.unpackValue();
            valuesToDiscard=valuesToDiscard-1;
        }
        return msg.unpackValue();
    }
    public static boolean readBool(final MessageUnpacker msg) throws IOException
    {
        return readBool(msg,0);
    }
    public static boolean readBool(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        return val.asBooleanValue().getBoolean();
    }
    public static int readInt(final MessageUnpacker msg) throws IOException
    {
        return readInt(msg,0);
    }
    public static int readInt(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isIntegerValue())
            return val.asNumberValue().toInt();
        throw new IOException("not an int");
    }
    public static float readFloat(final MessageUnpacker msg) throws IOException
    {
        return readFloat(msg,0);
    }
    public static float readFloat(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        return (float)readDouble(msg,valuesToDiscard);
    }
    public static double readDouble(final MessageUnpacker msg) throws IOException
    {
        return readDouble(msg,0);
    }
    public static double readDouble(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        return val.asNumberValue().toDouble();
    }
    public static String readString(final MessageUnpacker msg) throws IOException
    {
        return readString(msg,0);
    }
    public static String readString(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        return val.asStringValue().asString();
    }
    public static byte[] readByteArray(final MessageUnpacker msg) throws IOException
    {
        return readByteArray(msg,0);
    }
    public static byte[] readByteArray(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isRawValue())
            return val.asRawValue().asByteArray();
        throw new IOException("not a byte array");
    }
    public static int[] readIntArray(final MessageUnpacker msg) throws IOException
    {
        return readIntArray(msg,0);
    }
    public static int[] readIntArray(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isArrayValue())
        {
            ArrayValue arr=val.asArrayValue();
            int s=arr.size();
            int[] retVal=new int[s];
            for (int i=0;i<s;i=i+1)
            {
                Value v=arr.get(i);
                if (v.isNumberValue())
                    retVal[i]=v.asNumberValue().toInt();
                else
                    retVal[i]=0;
            }
            return retVal;
        }
        throw new IOException("not an array");
    }
    public static float[] readFloatArray(final MessageUnpacker msg) throws IOException
    {
        return readFloatArray(msg,0);
    }
    public static float[] readFloatArray(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isArrayValue())
        {
            ArrayValue arr=val.asArrayValue();
            int s=arr.size();
            float[] retVal=new float[s];
            for (int i=0;i<s;i=i+1)
            {
                Value v=arr.get(i);
                if (v.isNumberValue())
                    retVal[i]=v.asNumberValue().toFloat();
                else
                    retVal[i]=0.0f;
            }
            return retVal;
        }
        throw new IOException("not an array");
    }
    public static double[] readDoubleArray(final MessageUnpacker msg) throws IOException
    {
        return readDoubleArray(msg,0);
    }
    public static double[] readDoubleArray(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isArrayValue())
        {
            ArrayValue arr=val.asArrayValue();
            int s=arr.size();
            double[] retVal=new double[s];
            for (int i=0;i<s;i=i+1)
            {
                Value v=arr.get(i);
                if (v.isNumberValue())
                    retVal[i]=v.asNumberValue().toDouble();
                else
                    retVal[i]=0.0;
            }
            return retVal;
        }
        throw new IOException("not an array");
    }
    public static String[] readStringArray(final MessageUnpacker msg) throws IOException
    {
        return readStringArray(msg,0);
    }
    public static String[] readStringArray(final MessageUnpacker msg,int valuesToDiscard) throws IOException
    {
        Value val=readValue(msg,valuesToDiscard);
        if (val.isArrayValue())
        {
            ArrayValue arr=val.asArrayValue();
            int s=arr.size();
            String[] retVal=new String[s];
            for (int i=0;i<s;i=i+1)
            {
                Value v=arr.get(i);
                if (v.isStringValue())
                    retVal[i]=v.asStringValue().asString();
                else
                    retVal[i]=new String("");
            }
            return retVal;
        }
        throw new IOException("not an array");
    }
    
    public b0RemoteApi() throws IOException
    {
        _b0RemoteApi("b0RemoteApi_c++Client","b0RemoteApi",60,false,3);
    }
    public b0RemoteApi(final String nodeName) throws IOException
    {
        _b0RemoteApi(nodeName,"b0RemoteApi",60,false,3);
    }
    public b0RemoteApi(final String nodeName,final String channelName) throws IOException
    {
        _b0RemoteApi(nodeName,channelName,60,false,3);
    }
    public b0RemoteApi(final String nodeName,final String channelName,int inactivityToleranceInSec) throws IOException
    {
        _b0RemoteApi(nodeName,channelName,inactivityToleranceInSec,false,3);
    }
    public b0RemoteApi(final String nodeName,final String channelName,int inactivityToleranceInSec,boolean setupSubscribersAsynchronously) throws IOException
    {
        _b0RemoteApi(nodeName,channelName,inactivityToleranceInSec,setupSubscribersAsynchronously,3);
    }

    public b0RemoteApi(final String nodeName,final String channelName,int inactivityToleranceInSec,boolean setupSubscribersAsynchronously,int timeout) throws IOException
    {
        _b0RemoteApi(nodeName,channelName,inactivityToleranceInSec,setupSubscribersAsynchronously,timeout);
    }
    
    private void _b0RemoteApi(final String nodeName,final String channelName,int inactivityToleranceInSec,boolean setupSubscribersAsynchronously,int timeout) throws IOException
    {
        _channelName=channelName;
        _serviceCallTopic=_channelName.concat("SerX");
        _defaultPublisherTopic=_channelName.concat("SubX");
        _defaultSubscriberTopic=_channelName.concat("PubX");
        _nextDefaultSubscriberHandle=2;
        _nextDedicatedPublisherHandle=500;
        _nextDedicatedSubscriberHandle=1000;
        b0Init();
        _node=b0NodeNew(nodeName);
        Random rand = new Random(System.currentTimeMillis());
        String alp=new String("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        _clientId="";
        for (int i=0;i<10;i=i+1)
        {
            char c=alp.charAt(rand.nextInt(62));
            _clientId=_clientId.concat(String.valueOf(c));
        }
        _serviceClient=b0ServiceClientNewEx(_node,_serviceCallTopic,1,1);
        b0ServiceClientSetOption(_serviceClient,B0_SOCK_OPT_READTIMEOUT,timeout*1000);
        _defaultPublisher=b0PublisherNewEx(_node,_defaultPublisherTopic,1,1);
        _defaultSubscriber=b0SubscriberNewEx(_node,_defaultSubscriberTopic,1,1);
        System.out.println("");
        System.out.println("Running B0 Remote API client with channel name ["+_channelName+"]");
        System.out.println("  make sure that: 1) the B0 resolver is running");
        System.out.println("                  2) CoppeliaSim is running the B0 Remote API server with the same channel name");
        System.out.println("  Initializing...");
        System.out.println("");
        b0NodeInit(_node);
        
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packInt(inactivityToleranceInSec);
        _handleFunction("inactivityTolerance",args,_serviceCallTopic);
        _setupSubscribersAsynchronously=setupSubscribersAsynchronously;
        _allSubscribers=new HashMap<String,SHandleAndCb>();
        _allDedicatedPublishers=new HashMap<String,SHandle>();

        System.out.println("");
        System.out.println("  Connected!");
        System.out.println("");
    }
    
    public void delete() throws IOException
    {
        System.out.println("*************************************************************************************");
        System.out.println("** Leaving... if this is unexpected, you might have to adjust the timeout argument **");
        System.out.println("*************************************************************************************");
        _pongReceived=false;
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packInt(0);
        String pingTopic=simxDefaultSubscriber(this::_pingCallback);
        _handleFunction("Ping",args,pingTopic);
        
        while (!_pongReceived)
            simxSpinOnce();

        args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packString(_clientId);
        _handleFunction("DisconnectClient",args,_serviceCallTopic);

        for (String key:_allSubscribers.keySet())
        {
            SHandleAndCb val=_allSubscribers.get(key);
            if (val.handle!=_defaultSubscriber)
                b0SubscriberDelete(val.handle);
        }

        for (String key:_allDedicatedPublishers.keySet())
        {
            SHandle val=_allDedicatedPublishers.get(key);
            b0PublisherDelete(val.handle);
        }
        
//        b0NodeDelete(_node);
    }

    private void _pingCallback(final MessageUnpacker msg) 
    {
        _pongReceived=true;
    }

    
    private byte[] _concatPackers(final MessageBufferPacker header,final MessageBufferPacker data) throws IOException
    {
        MessageBufferPacker msg=MessagePack.newDefaultBufferPacker();
        msg.packArrayHeader(2);
        byte[] packedData=new byte[msg.toByteArray().length+header.toByteArray().length+data.toByteArray().length];
        System.arraycopy(msg.toByteArray(),0,packedData,0,msg.toByteArray().length);
        System.arraycopy(header.toByteArray(),0,packedData,msg.toByteArray().length,header.toByteArray().length);
        System.arraycopy(data.toByteArray(),0,packedData,msg.toByteArray().length+header.toByteArray().length,data.toByteArray().length);
        return packedData;
    }
    
    private MessageUnpacker _handleFunction(final String funcName,final MessageBufferPacker packedArgs,final String topic) throws IOException
    {
        if (topic.equals(_serviceCallTopic))
        {
            MessageBufferPacker header=MessagePack.newDefaultBufferPacker();
            header.packArrayHeader(4).packString(funcName).packString(_clientId).packString(topic).packInt(0);
            byte[] rep=b0ServiceClientCall(_serviceClient,_concatPackers(header,packedArgs));
            MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(rep);
            int s=unpacker.unpackArrayHeader();
            if (s>=2)
//                return MessagePack.newDefaultUnpacker(rep);
                return unpacker;
            MessageBufferPacker tmp=MessagePack.newDefaultBufferPacker();
//            tmp.packArrayHeader(2).packBoolean(unpacker.unpackBoolean()).packNil();
            tmp.packBoolean(unpacker.unpackBoolean()).packNil();
            return MessagePack.newDefaultUnpacker(tmp.toByteArray());
        }
        else if (topic.equals(_defaultPublisherTopic))
        {
            MessageBufferPacker header=MessagePack.newDefaultBufferPacker();
            header.packArrayHeader(4).packString(funcName).packString(_clientId).packString(topic).packInt(1);
            b0PublisherPublish(_defaultPublisher,_concatPackers(header,packedArgs));
        }
        else
        {
            if (_allSubscribers.containsKey(topic))
            {
                SHandleAndCb val=_allSubscribers.get(topic);
                MessageBufferPacker header=MessagePack.newDefaultBufferPacker();
                if (val.handle==_defaultSubscriber)
                    header.packArrayHeader(4).packString(funcName).packString(_clientId).packString(topic).packInt(2);
                else
                    header.packArrayHeader(4).packString(funcName).packString(_clientId).packString(topic).packInt(4);
                if (_setupSubscribersAsynchronously)
                    b0PublisherPublish(_defaultPublisher,_concatPackers(header,packedArgs));
                else
                    b0ServiceClientCall(_serviceClient,_concatPackers(header,packedArgs));
            }
            else
            {
                if (_allDedicatedPublishers.containsKey(topic))
                {
                    SHandle val=_allDedicatedPublishers.get(topic);
                    MessageBufferPacker header=MessagePack.newDefaultBufferPacker();
                    header.packArrayHeader(4).packString(funcName).packString(_clientId).packString(topic).packInt(3);
                    b0PublisherPublish(val.handle,_concatPackers(header,packedArgs));
                }
            }
        }
        return null;
    }
    
    public void simxSpin() throws IOException
    {
        while (true)
            simxSpinOnce();
    }

    public void simxSpinOnce() throws IOException
    {
        boolean defaultSubscriberAlreadyProcessed=false;
        for (String key:_allSubscribers.keySet())
        {
            byte[] packedData=null;
            SHandleAndCb val=_allSubscribers.get(key);
            if ( (val.handle!=_defaultSubscriber)||(!defaultSubscriberAlreadyProcessed) )
            {
                defaultSubscriberAlreadyProcessed=defaultSubscriberAlreadyProcessed|(val.handle==_defaultSubscriber);
                while (b0SubscriberPoll(val.handle,0)>0)
                {
                    packedData=b0SubscriberRead(val.handle);
                    if (!val.dropMessages)
                        _handleReceivedMessage(packedData);
                }
                if ( val.dropMessages&&(packedData!=null) )
                    _handleReceivedMessage(packedData);
            }
        }
    }

    private void _handleReceivedMessage(final byte[] packedData) throws IOException
    {
        if (packedData.length>0)
        {
            MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(packedData);
            int s=unpacker.unpackArrayHeader();
            if (s==2)
            {
                String topic=unpacker.unpackString();
                if (_allSubscribers.containsKey(topic))
                {
                    unpacker.unpackArrayHeader();
                    SHandleAndCb val=_allSubscribers.get(topic);
                    val.cb.accept(unpacker);
                    
                    
 /*                   
                    MessageUnpacker msg=null;
                    s=unpacker.unpackArrayHeader();
                    if (s>=2)
                    {
                        msg=MessagePack.newDefaultUnpacker(packedData);
                        msg.unpackArrayHeader();
                        msg.unpackString();
                    }
                    else
                    {
                        MessageBufferPacker tmp=MessagePack.newDefaultBufferPacker();
                        tmp.packArrayHeader(2).packBoolean(unpacker.unpackBoolean()).packNil();                        
                        msg=MessagePack.newDefaultUnpacker(tmp.toByteArray());
                    }
                    SHandleAndCb val=_allSubscribers.get(topic);
                    val.cb.accept(msg);
                    */
                }
            }
        }
    }

    public long simxGetTimeInMs()
    {
        return b0NodeHardwareTimeUsec(_node)/1000;
    }

    public void simxSleep(int durationInMs) throws InterruptedException
    {
        Thread.sleep(durationInMs);
    }
    
    public String simxDefaultPublisher()
    {
        return _defaultPublisherTopic;
    }

    public String simxCreatePublisher() throws IOException
    {
        return simxCreatePublisher(false);
    }
    
    public String simxCreatePublisher(boolean dropMessages) throws IOException
    {
        String topic=_channelName+"Sub"+_nextDedicatedPublisherHandle+_clientId;
        _nextDedicatedPublisherHandle=_nextDedicatedPublisherHandle+1;
        long pub=b0PublisherNewEx(_node,topic,0,1);
        b0PublisherInit(pub);
        SHandle dat=new SHandle();
        dat.handle=pub;
        _allDedicatedPublishers.put(topic,dat);
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(2).packString(topic).packBoolean(dropMessages);
        _handleFunction("createSubscriber",args,_serviceCallTopic);
        return topic;
    }

    public String simxDefaultSubscriber(final Consumer<MessageUnpacker> cb) throws IOException
    {
        return simxDefaultSubscriber(cb,1);
    }
    public String simxDefaultSubscriber(final Consumer<MessageUnpacker> cb,int publishInterval) throws IOException
    {
        String topic=_channelName+"Pub"+_nextDefaultSubscriberHandle+_clientId;
        _nextDefaultSubscriberHandle=_nextDefaultSubscriberHandle+1;
        SHandleAndCb dat=new SHandleAndCb();
        dat.handle=_defaultSubscriber;
        dat.cb=cb;
        dat.dropMessages=false;
        _allSubscribers.put(topic,dat);
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(2).packString(topic).packInt(publishInterval);
        String channel=_serviceCallTopic;
        if (_setupSubscribersAsynchronously)
            channel=_defaultPublisherTopic;
        _handleFunction("setDefaultPublisherPubInterval",args,channel);
        return topic;
    }

    public String simxCreateSubscriber(final Consumer<MessageUnpacker> cb) throws IOException
    {
        return simxCreateSubscriber(cb,1);
    }
    public String simxCreateSubscriber(final Consumer<MessageUnpacker> cb,int publishInterval) throws IOException
    {
        return simxCreateSubscriber(cb,publishInterval,false);
    }
    
    public String simxCreateSubscriber(final Consumer<MessageUnpacker> cb,int publishInterval,boolean dropMessages) throws IOException
    {
        String topic=_channelName+"Pub"+_nextDedicatedSubscriberHandle+_clientId;
        _nextDedicatedSubscriberHandle=_nextDedicatedSubscriberHandle+1;
        long sub=b0SubscriberNewEx(_node,topic,0,1);
        if (dropMessages)
            b0SubscriberSetOption(sub,B0_SOCK_OPT_CONFLATE,1);
        else
            b0SubscriberSetOption(sub,B0_SOCK_OPT_CONFLATE,0);
        b0SubscriberInit(sub);
        SHandleAndCb dat=new SHandleAndCb();
        dat.handle=sub;
        dat.cb=cb;
        dat.dropMessages=dropMessages;
        _allSubscribers.put(topic,dat);
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(2).packString(topic).packInt(publishInterval);
        String channel=_serviceCallTopic;
        if (_setupSubscribersAsynchronously)
            channel=_defaultPublisherTopic;
        _handleFunction("createPublisher",args,channel);
        return topic;
    }
    
    public void simxRemoveSubscriber(final String topic) throws IOException
    {
        if (_allSubscribers.containsKey(topic))
        {
            SHandleAndCb val=_allSubscribers.get(topic);
            MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
            args.packArrayHeader(1).packString(topic);
            String channel=_serviceCallTopic;
            if (_setupSubscribersAsynchronously)
                channel=_defaultPublisherTopic;
            if (val.handle==_defaultSubscriber)
                _handleFunction("stopDefaultPublisher",args,channel);
            else
            {
                b0SubscriberDelete(val.handle);
                _handleFunction("stopPublisher",args,channel);
            }
            _allSubscribers.remove(topic);
        }
    }
    
    public void simxRemovePublisher(final String topic) throws IOException
    {
        if (_allDedicatedPublishers.containsKey(topic))
        {
            SHandle val=_allDedicatedPublishers.get(topic);
            b0PublisherDelete(val.handle);            
            MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
            args.packArrayHeader(1).packString(topic);
            _handleFunction("stopSubscriber",args,_serviceCallTopic);
            _allDedicatedPublishers.remove(topic);
        }
    }
    
    public String simxServiceCall()
    {
        return _serviceCallTopic;
    }
    

    private static final int B0_SOCK_OPT_READTIMEOUT = 3;
    private static final int B0_SOCK_OPT_CONFLATE = 6;
    
    private native int b0Init();
    
    private native long b0NodeNew(final String name);
    private native void b0NodeDelete(long node);
    private native void b0NodeInit(long node);
    private native long b0NodeTimeUsec(long node);
    private native long b0NodeHardwareTimeUsec(long node);

    private native long b0PublisherNewEx(long node,final String topicName,int managed,int notifyGraph);
    private native void b0PublisherDelete(long pub);
    private native void b0PublisherInit(long pub);
    private native void b0PublisherPublish(long pub,byte[] data);
    
    private native long b0SubscriberNewEx(long node,final String topicName,int managed,int notifyGraph);
    private native void b0SubscriberDelete(long sub);
    private native void b0SubscriberInit(long sub);
    private native int b0SubscriberPoll(long sub,long timeout);
    private native byte[] b0SubscriberRead(long sub);
    private native void b0SubscriberSetOption(long sub,long option,long value);
    
    private native long b0ServiceClientNewEx(long node,final String serviceName,int managed,int notifyGraph);
    private native void b0ServiceClientDelete(long cli);
    private native byte[] b0ServiceClientCall(long cli,byte[] data);
    private native void b0ServiceClientSetOption(long cli,long option,long value);
    
    String _serviceCallTopic;
    String _defaultPublisherTopic;
    String _defaultSubscriberTopic;
    int _nextDefaultSubscriberHandle;
    int _nextDedicatedPublisherHandle;
    int _nextDedicatedSubscriberHandle;
    boolean _pongReceived;
    boolean _setupSubscribersAsynchronously;
    String _channelName;
    long _node;
    String _clientId;
    long _serviceClient;
    long _defaultPublisher;
    long _defaultSubscriber;
    HashMap<String,SHandleAndCb> _allSubscribers;
    HashMap<String,SHandle> _allDedicatedPublishers;
    
    public void simxSynchronous(boolean enable) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packBoolean(enable);
        _handleFunction("Synchronous",args,_serviceCallTopic);
    }
    
    public void simxSynchronousTrigger() throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packInt(0);
        _handleFunction("SynchronousTrigger",args,_defaultPublisherTopic);
    }
    public void simxGetSimulationStepDone(final String topic) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packInt(0);
        _handleFunction("GetSimulationStepDone",args,topic);
    }
    
    public void simxGetSimulationStepStarted(final String topic) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(1).packInt(0);
        _handleFunction("GetSimulationStepStarted",args,topic);
    }

    public MessageUnpacker simxCallScriptFunction(final String funcAtObjName,int scriptType,byte[] packedArg,final String topic) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(3);
        args.packString(funcAtObjName);
        args.packInt(scriptType);
        args.packBinaryHeader(packedArg.length);
        args.writePayload(packedArg);
        return _handleFunction("CallScriptFunction",args,topic);
    }
    
    public MessageUnpacker simxCallScriptFunction(final String funcAtObjName,final String scriptType,byte[] packedArg,final String topic) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
        args.packArrayHeader(3);
        args.packString(funcAtObjName);
        args.packString(scriptType);
        args.packBinaryHeader(packedArg.length);
        args.writePayload(packedArg);
        return _handleFunction("CallScriptFunction",args,topic);
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
#py theStringToWrite='    public MessageUnpacker '+cmd.name+'(\n'
#py itemCnt=len(cmd.params)
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='        int '+p.name
#py else:
#py theStringToWrite+='        final String '+p.name
#py endif
#py elif p.ctype()=='bool':
#py theStringToWrite+='        boolean '+p.name
#py else:
#py theStringToWrite+='        '+p.htype()+' '+p.name
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+=',\n'
#py endif
#py endfor
`theStringToWrite`) throws IOException
    {
        MessageBufferPacker args=MessagePack.newDefaultBufferPacker();
#py if len(cmd.params)==1:
#py theStringToWrite='        args.packArrayHeader(1).packInt(0);\n'
#py else:
#py itemCnt=len(cmd.params)-1
#py theStringToWrite='        args.packArrayHeader('+str(itemCnt)+');\n'
#py itemIndex=-1
#py for p in cmd.params:
#py itemIndex=itemIndex+1
#py if p.htype()=='bool':
#py theStringToWrite+='        args.packBoolean('+p.name+');'
#py elif p.htype()=='int':
#py theStringToWrite+='        args.packInt('+p.name+');'
#py elif p.htype()=='float':
#py theStringToWrite+='        args.packFloat('+p.name+');'
#py elif p.htype()=='double':
#py theStringToWrite+='        args.packDouble('+p.name+');'
#py elif p.ctype()=='string':
#py theStringToWrite+='        args.packString('+p.name+');'
#py elif p.ctype()=='int_eval':
#py if k==0:
#py theStringToWrite+='        args.packInt('+p.name+');'
#py else:
#py theStringToWrite+='        args.packString('+p.name+');'
#py endif
#py elif p.ctype()=='byte[]':
#py theStringToWrite+='        args.packBinaryHeader('+p.name+'.length);\n'
#py theStringToWrite+='        args.writePayload('+p.name+');'
#py elif 'int[' in p.ctype():
#py valCntStr=''
#py if p.ctype()=='int[]':
#py valCntStr=p.name+'.length'
#py else:
#py valCntStr=p.ctype()[4:-1]
#py endif
#py theStringToWrite+='        args.packArrayHeader('+valCntStr+');\n'
#py theStringToWrite+='        for (int i=0;i<'+valCntStr+';i=i+1)\n'
#py theStringToWrite+='            args.packInt('+p.name+'[i]);'
#py elif 'float[' in p.ctype():
#py valCntStr=''
#py if p.ctype()=='float[]':
#py valCntStr=p.name+'.length'
#py else:
#py valCntStr=p.ctype()[6:-1]
#py endif
#py theStringToWrite+='        args.packArrayHeader('+valCntStr+');\n'
#py theStringToWrite+='        for (int i=0;i<'+valCntStr+';i=i+1)\n'
#py theStringToWrite+='            args.packFloat('+p.name+'[i]);'
#py elif 'double[' in p.ctype():
#py valCntStr=''
#py if p.ctype()=='double[]':
#py valCntStr=p.name+'.length'
#py else:
#py valCntStr=p.ctype()[7:-1]
#py endif
#py theStringToWrite+='        args.packArrayHeader('+valCntStr+');\n'
#py theStringToWrite+='        for (int i=0;i<'+valCntStr+';i=i+1)\n'
#py theStringToWrite+='            args.packDouble('+p.name+'[i]);'
#py endif
#py if (itemCnt>1) and itemIndex<itemCnt-1:
#py theStringToWrite+='\n'
#py else:
#py break
#py endif
#py endfor
#py endif
`theStringToWrite`
        return _handleFunction("`cmd.name[4:]`",args,topic);
    }
#py endfor
#py endif
#py endfor

    // -----------------------------------------------------------
    // Add your custom functions here, or even better,
    // add them to b0RemoteApiBindings/generate/simxFunctions.xml,
    // and generate this file again.
    // Then add the server part of your custom functions at the
    // beginning of file lua/b0RemoteApiServer.lua
    // -----------------------------------------------------------
}
