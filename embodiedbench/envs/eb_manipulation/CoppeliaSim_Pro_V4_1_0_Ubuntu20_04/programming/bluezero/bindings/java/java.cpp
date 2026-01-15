extern "C"
{
#include "java.h"
#include <b0/bindings/c.h>

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0Init(JNIEnv *env, jobject obj)
{
    int argc = 1;
    char *argv[1] = {"b0java"};
    return b0_init(&argc, argv);
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0NodeNew(JNIEnv *env, jobject obj, jstring name)
{
    const char *_name=env->GetStringUTFChars(name,0);
    b0_node* node=b0_node_new(_name);
    env->ReleaseStringUTFChars(name,_name);
    return((jlong)node);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0NodeDelete(JNIEnv *env, jobject obj, jlong node)
{
    b0_node_delete((b0_node*)node);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0NodeInit(JNIEnv *env, jobject obj, jlong node)
{
    b0_node_init((b0_node*)node);
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0NodeTimeUsec(JNIEnv *env, jobject obj, jlong node)
{
    return(b0_node_time_usec((b0_node*)node));
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0NodeHardwareTimeUsec(JNIEnv *env, jobject obj, jlong node)
{
    return(b0_node_hardware_time_usec((b0_node*)node));
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0PublisherNewEx(JNIEnv *env, jobject obj, jlong node, jstring topicName, jint managed, jint notifyGraph)
{
    const char *_topicName = env->GetStringUTFChars(topicName, 0);
    b0_publisher* pub = b0_publisher_new_ex((b0_node*)node,_topicName,
    managed, notifyGraph);
    env->ReleaseStringUTFChars(topicName, _topicName);
    return((jlong)pub);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherDelete(JNIEnv *env, jobject obj, jlong pub)
{
    b0_publisher_delete((b0_publisher*)pub);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherInit(JNIEnv *env, jobject obj, jlong pub)
{
    b0_publisher_init((b0_publisher*)pub);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherPublish(JNIEnv *env, jobject obj, jlong pub, jbyteArray data)
{
    jbyte* bufferPtr=env->GetByteArrayElements(data,0);
    jsize l=env->GetArrayLength(data);
    b0_publisher_publish((b0_publisher*)pub, bufferPtr,l);
    env->ReleaseByteArrayElements(data,bufferPtr,0);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherSetOption(JNIEnv *env, jobject obj, jlong pub, jlong option, jlong value)
{
    b0_publisher_set_option((b0_publisher*)pub, option, value);
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0SubscriberNewEx(JNIEnv *env, jobject obj, jlong node, jstring topicName, jint managed, jint notifyGraph)
{
    const char *_topicName = env->GetStringUTFChars(topicName, 0);
    b0_subscriber* sub = b0_subscriber_new_ex((b0_node*)node,
    _topicName, 0, managed, notifyGraph);
    env->ReleaseStringUTFChars(topicName, _topicName);
    return((jlong)sub);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberDelete(JNIEnv *env, jobject obj, jlong sub)
{
    b0_subscriber_delete((b0_subscriber*)sub);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberInit(JNIEnv *env, jobject obj, jlong sub)
{
    b0_subscriber_init((b0_subscriber*)sub);
}

JNIEXPORT jint JNICALL Java_coppelia_b0RemoteApi_b0SubscriberPoll(JNIEnv *env, jobject obj, jlong sub, jlong timeout)
{
    return(b0_subscriber_poll((b0_subscriber*)sub,(long)timeout));
}

JNIEXPORT jbyteArray JNICALL Java_coppelia_b0RemoteApi_b0SubscriberRead(JNIEnv *env, jobject obj, jlong sub)
{
    size_t l;
    void* data=b0_subscriber_read((b0_subscriber*)sub,&l);
    jbyteArray jarray=env->NewByteArray(l);
    env->SetByteArrayRegion(jarray,0,l,(jbyte*)data);
    b0_buffer_delete(data);
    return(jarray);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberSetOption(JNIEnv *env, jobject obj, jlong sub, jlong option, jlong value)
{
    b0_subscriber_set_option((b0_subscriber*)sub, option, value);
}

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientNewEx(JNIEnv *env, jobject obj, jlong node, jstring serviceName, jint managed, jint notifyGraph)
{
    const char *_serviceName = env->GetStringUTFChars(serviceName, 0);
    b0_service_client* cli =b0_service_client_new_ex((b0_node*)node,
    _serviceName, managed, notifyGraph);
    env->ReleaseStringUTFChars(serviceName, _serviceName);
    return((jlong)cli);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientDelete(JNIEnv *env, jobject
obj, jlong cli)
{
    b0_service_client_delete((b0_service_client*)cli);
}

JNIEXPORT jbyteArray JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientCall(JNIEnv *env, jobject obj, jlong cli, jbyteArray data)
{
    jbyte* inBufferPtr = env->GetByteArrayElements(data, 0);
    jsize inL = env->GetArrayLength(data);
    size_t outL;
    void* outData =
    b0_service_client_call((b0_service_client*)cli,inBufferPtr,inL,&outL);
    jbyteArray jarray = env->NewByteArray(outL);
    env->SetByteArrayRegion(jarray, 0, outL, (jbyte*)outData);
    b0_buffer_delete(outData);
    env->ReleaseByteArrayElements(data, inBufferPtr, 0);
    return(jarray);
}

JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientSetOption(JNIEnv *env, jobject obj, jlong cli, jlong option, jlong value)
{
    b0_service_client_set_option((b0_service_client*)cli, option, value);
}

} // extern "C"
