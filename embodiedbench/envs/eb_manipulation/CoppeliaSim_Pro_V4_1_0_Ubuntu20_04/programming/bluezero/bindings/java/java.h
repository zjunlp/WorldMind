#include <jni.h>

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0Init(JNIEnv *env, jobject obj);
JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0NodeNew(JNIEnv *env, jobject obj, jstring name);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0NodeDelete(JNIEnv *env, jobject obj, jlong node);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0NodeInit(JNIEnv *env, jobject obj, jlong node);
JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0NodeTimeUsec(JNIEnv *env, jobject obj, jlong node);

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0PublisherNewEx(JNIEnv *env, jobject obj,
jlong node, jstring topicName, jint managed, jint notifyGraph);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherDelete(JNIEnv *env, jobject obj,
jlong pub);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherInit(JNIEnv *env, jobject obj, jlong pub);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherPublish(JNIEnv *env, jobject obj, jlong pub, jbyteArray data);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0PublisherSetOption(JNIEnv *env, jobject obj, jlong pub, jlong option, jlong value);

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0SubscriberNewEx(JNIEnv *env, jobject obj, jlong node, jstring topicName, jint managed, jint notifyGraph);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberDelete(JNIEnv *env, jobject obj, jlong sub);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberInit(JNIEnv *env, jobject obj, jlong sub);
JNIEXPORT jint JNICALL Java_coppelia_b0RemoteApi_b0SubscriberPoll(JNIEnv *env, jobject obj, jlong sub, jlong timeout);
JNIEXPORT jbyteArray JNICALL Java_coppelia_b0RemoteApi_b0SubscriberRead(JNIEnv *env, jobject obj, jlong sub);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0SubscriberSetOption(JNIEnv *env, jobject obj, jlong sub, jlong option, jlong value);

JNIEXPORT jlong JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientNewEx(JNIEnv *env, jobject obj, jlong node, jstring serviceName, jint managed, jint notifyGraph);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientDelete(JNIEnv *env, jobject obj, jlong cli);
JNIEXPORT jbyteArray JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientCall(JNIEnv *env, jobject obj, jlong cli, jbyteArray data);
JNIEXPORT void JNICALL Java_coppelia_b0RemoteApi_b0ServiceClientSetOption(JNIEnv *env, jobject obj, jlong cli, jlong option, jlong value);

