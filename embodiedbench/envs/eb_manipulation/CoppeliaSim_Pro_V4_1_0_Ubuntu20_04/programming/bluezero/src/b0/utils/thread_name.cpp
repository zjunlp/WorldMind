#include <b0/utils/thread_name.h>

#ifdef _WIN32

#ifndef __MINGW64__

#include <windows.h>

const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType;     // Must be 0x1000.
    LPCSTR szName;    // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags;    // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

void dbg_set_thread_name(uint32_t dwThreadID, const char* threadName)
{
    // DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
    }
}

void dbg_set_thread_name(const char *threadName)
{
    dbg_set_thread_name(GetCurrentThreadId(), threadName);
}

#if 0
void set_thread_name(std::thread *thread, const char *threadName)
{
    DWORD threadId = ::GetThreadId(static_cast<HANDLE>(thread->native_handle()));
    set_thread_name(threadId, threadName);
}
#endif

#endif // !__MINGW64__

#include <string>
#include <map>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

static std::map<std::string, std::string> thread_name;

void set_thread_name(const char *threadName)
{
#ifndef __MINGW64__
    dbg_set_thread_name(threadName);
#endif
    std::string this_thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    thread_name[this_thread_id] = std::string(threadName);
}

std::string get_thread_name()
{
    std::string this_thread_id = boost::lexical_cast<std::string>(boost::this_thread::get_id());
    auto i = thread_name.find(this_thread_id);
    return i == thread_name.end() ? "" : i->second;
}

#else // _WIN32

#include <pthread.h>

#if 0
void set_thread_name(std::thread *thread, const char *threadName)
{
    auto handle = thread->native_handle();
    pthread_setname_np(handle, threadName);
}
#endif

void set_thread_name(const char *threadName)
{
#if defined(HAVE_PTHREAD_SETNAME_3)
    pthread_setname_np(pthread_self(), threadName, (void*)0);
#elif defined(HAVE_PTHREAD_SETNAME_2)
    pthread_setname_np(pthread_self(), threadName);
#elif defined(HAVE_PTHREAD_SETNAME_1)
    pthread_setname_np(threadName);
#else
#error "pthread_setname_np() not available"
#endif
}

std::string get_thread_name()
{
    char buf[64];
    pthread_getname_np(pthread_self(), &buf[0], 64);
    return std::string(&buf[0]);
}

#endif // _WIN32

