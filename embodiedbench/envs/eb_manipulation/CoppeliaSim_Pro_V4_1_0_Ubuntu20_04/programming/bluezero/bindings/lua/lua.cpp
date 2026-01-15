extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

#define B0_INIT_COMMAND "b0.init"
int B0_INIT_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
    int argc = 1;
    char *argv[1] = {"b0lua"};
    b0_init(&argc, argv);
	return(retValCnt);
}

#define B0_NODE_NEW_COMMAND "b0.node_new"
int B0_NODE_NEW_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_isstring(L, 1))
	{
		const char* nodeName = lua_tostring(L, 1);
		b0_node* node = b0_node_new(nodeName);
		if (node != NULL)
		{
			lua_pushlightuserdata(L, node);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

#define B0_NODE_DELETE_COMMAND "b0.node_delete"
int B0_NODE_DELETE_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		b0_node_delete(ptr);
	}
	return(retValCnt);
}

#define B0_NODE_INIT_COMMAND "b0.node_init"
int B0_NODE_INIT_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		b0_node_init(ptr);
	}
	return(retValCnt);
}

#define B0_NODE_SPIN_ONCE_COMMAND "b0.node_spin_once"
int B0_NODE_SPIN_ONCE_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		b0_node_spin_once(ptr);
	}
	return(retValCnt);
}

#define B0_NODE_TIME_USEC_COMMAND "b0.node_time_usec"
int B0_NODE_TIME_USEC_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		long long t = b0_node_time_usec(ptr);
		lua_pushnumber(L, (double)t);
		retValCnt = 1;
	}
	return(retValCnt);
}

#define B0_NODE_HARDWARE_TIME_USEC_COMMAND "b0.node_hardware_time_usec"
int B0_NODE_HARDWARE_TIME_USEC_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		long long t = b0_node_hardware_time_usec(ptr);
		lua_pushnumber(L, (double)t);
		retValCnt = 1;
	}
	return(retValCnt);
}

#define B0_SERVICE_CLIENT_NEW_EX_COMMAND "b0.service_client_new_ex"
int B0_SERVICE_CLIENT_NEW_EX_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 4) && lua_islightuserdata(L, 1) && lua_isstring(L, 2) && lua_isnumber(L, 3) && lua_isnumber(L, 4))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		const char* serviceName = lua_tostring(L, 2);
		int managed = (int)lua_tointeger(L, 3);
		int notify_graph = (int)lua_tointeger(L, 4);
		b0_service_client* sc = b0_service_client_new_ex(ptr, serviceName, managed, notify_graph);
		if (sc != NULL)
		{
			lua_pushlightuserdata(L, sc);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

#define B0_SERVICE_CLIENT_DELETE_COMMAND "b0.service_client_delete"
int B0_SERVICE_CLIENT_DELETE_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_service_client* ptr = (b0_service_client*)lua_touserdata(L, 1);
		b0_service_client_delete(ptr);
	}
	return(retValCnt);
}

#define B0_SERVICE_CLIENT_CALL_COMMAND "b0.service_client_call"
int B0_SERVICE_CLIENT_CALL_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 2) && lua_islightuserdata(L, 1) && lua_isstring(L, 2))
	{
		b0_service_client* ptr = (b0_service_client*)lua_touserdata(L, 1);
		size_t dataInSize;
		const char* dataIn = lua_tolstring(L, 2, &dataInSize);
		size_t dataOutSize;
		char* dataOut = (char*)b0_service_client_call(ptr, dataIn, dataInSize, &dataOutSize);
		if (dataOut != NULL)
		{
			lua_pushlstring(L, dataOut, dataOutSize);
			b0_buffer_delete(dataOut);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

#define B0_PUBLISHER_NEW_EX_COMMAND "b0.publisher_new_ex"
int B0_PUBLISHER_NEW_EX_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 4) && lua_islightuserdata(L, 1) && lua_isstring(L, 2) && lua_isnumber(L, 3) && lua_isnumber(L, 4))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		const char* pubName = lua_tostring(L, 2);
		int managed = (int)lua_tointeger(L, 3);
		int notify_graph = (int)lua_tointeger(L, 4);
		b0_publisher* pub = b0_publisher_new_ex(ptr, pubName, managed, notify_graph);
		if (pub != NULL)
		{
			lua_pushlightuserdata(L, pub);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

#define B0_PUBLISHER_DELETE_COMMAND "b0.publisher_delete"
int B0_PUBLISHER_DELETE_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_publisher* ptr = (b0_publisher*)lua_touserdata(L, 1);
		b0_publisher_delete(ptr);
	}
	return(retValCnt);
}

#define B0_PUBLISHER_INIT_COMMAND "b0.publisher_init"
int B0_PUBLISHER_INIT_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_publisher* ptr = (b0_publisher*)lua_touserdata(L, 1);
		b0_publisher_init(ptr);
	}
	return(retValCnt);
}

#define B0_PUBLISHER_PUBLISH_COMMAND "b0.publisher_publish"
int B0_PUBLISHER_PUBLISH_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 2) && lua_islightuserdata(L, 1) && lua_isstring(L, 2))
	{
		b0_publisher* ptr = (b0_publisher*)lua_touserdata(L, 1);
		size_t inDataSize;
		const char* inData = lua_tolstring(L, 2, &inDataSize);
		b0_publisher_publish(ptr, inData, inDataSize);
	}
	return(retValCnt);
}

#define B0_SUBSCRIBER_NEW_EX_COMMAND "b0.subscriber_new_ex"
int B0_SUBSCRIBER_NEW_EX_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 4) && lua_islightuserdata(L, 1) && lua_isstring(L, 2) && lua_isnumber(L, 3) && lua_isnumber(L, 4))
	{
		b0_node* ptr = (b0_node*)lua_touserdata(L, 1);
		const char* subName = lua_tostring(L, 2);
		int managed = (int)lua_tointeger(L, 3);
		int notify_graph = (int)lua_tointeger(L, 4);
		b0_subscriber* sub = b0_subscriber_new_ex(ptr, subName, NULL, managed, notify_graph);
		if (sub != NULL)
		{
			lua_pushlightuserdata(L, sub);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

#define B0_SUBSCRIBER_DELETE_COMMAND "b0.subscriber_delete"
int B0_SUBSCRIBER_DELETE_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_subscriber* ptr = (b0_subscriber*)lua_touserdata(L, 1);
		b0_subscriber_delete(ptr);
	}
	return(retValCnt);
}

#define B0_SUBSCRIBER_INIT_COMMAND "b0.subscriber_init"
int B0_SUBSCRIBER_INIT_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_subscriber* ptr = (b0_subscriber*)lua_touserdata(L, 1);
		b0_subscriber_init(ptr);
	}
	return(retValCnt);
}

#define B0_SUBSCRIBER_POLL_COMMAND "b0.subscriber_poll"
int B0_SUBSCRIBER_POLL_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 2) && lua_islightuserdata(L, 1) && lua_isnumber(L, 2))
	{
		b0_subscriber* ptr = (b0_subscriber*)lua_touserdata(L, 1);
		long timeOut = (long)lua_tointeger(L, 2);
		int cnt = b0_subscriber_poll(ptr, timeOut);
		lua_pushinteger(L, cnt);
		retValCnt = 1;
	}
	return(retValCnt);
}

#define B0_SUBSCRIBER_READ_COMMAND "b0.subscriber_read"
int B0_SUBSCRIBER_READ_CALLBACK(lua_State* L)
{
	int argCnt = lua_gettop(L);
	int retValCnt = 0;
	if ((argCnt >= 1) && lua_islightuserdata(L, 1))
	{
		b0_subscriber* ptr = (b0_subscriber*)lua_touserdata(L, 1);
		size_t dataOutSize;
		char* dataOut = (char*)b0_subscriber_read(ptr, &dataOutSize);
		if (dataOut != NULL)
		{
			lua_pushlstring(L, dataOut, dataOutSize);
			b0_buffer_delete(dataOut);
			retValCnt = 1;
		}
	}
	return(retValCnt);
}

void lua_registerN(lua_State* L, char const* funcName, lua_CFunction functionCallback)
{
	std::string name(funcName);
	if (name.find("b0.") != std::string::npos)
	{
		name.erase(name.begin(), name.begin() + 3);

		lua_getfield(L, LUA_GLOBALSINDEX, "b0");
		if (!lua_istable(L, -1))
		{ // we first need to create the table
			lua_createtable(L, 0, 1);
			lua_setfield(L, LUA_GLOBALSINDEX, "b0");
			lua_pop(L, 1);
			lua_getfield(L, LUA_GLOBALSINDEX, "b0");
		}
		lua_pushstring(L, name.c_str());
		lua_pushcfunction(L, functionCallback);
		lua_settable(L, -3);
		lua_pop(L, 1);
	}
	else
		lua_register(L, funcName, functionCallback);
}

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
int luaopen_b0(lua_State *L)
{
	luaL_dostring(L, "b0={}");

	lua_registerN(L, B0_INIT_COMMAND, B0_INIT_CALLBACK);

	lua_registerN(L, B0_NODE_NEW_COMMAND, B0_NODE_NEW_CALLBACK);
	lua_registerN(L, B0_NODE_DELETE_COMMAND, B0_NODE_DELETE_CALLBACK);
	lua_registerN(L, B0_NODE_INIT_COMMAND, B0_NODE_INIT_CALLBACK);
	lua_registerN(L, B0_NODE_SPIN_ONCE_COMMAND, B0_NODE_SPIN_ONCE_CALLBACK);
	lua_registerN(L, B0_NODE_TIME_USEC_COMMAND, B0_NODE_TIME_USEC_CALLBACK);
	lua_registerN(L, B0_NODE_HARDWARE_TIME_USEC_COMMAND, B0_NODE_HARDWARE_TIME_USEC_CALLBACK);

	lua_registerN(L, B0_SERVICE_CLIENT_NEW_EX_COMMAND, B0_SERVICE_CLIENT_NEW_EX_CALLBACK);
	lua_registerN(L, B0_SERVICE_CLIENT_DELETE_COMMAND, B0_SERVICE_CLIENT_DELETE_CALLBACK);
	lua_registerN(L, B0_SERVICE_CLIENT_CALL_COMMAND, B0_SERVICE_CLIENT_CALL_CALLBACK);

	lua_registerN(L, B0_PUBLISHER_NEW_EX_COMMAND, B0_PUBLISHER_NEW_EX_CALLBACK);
	lua_registerN(L, B0_PUBLISHER_DELETE_COMMAND, B0_PUBLISHER_DELETE_CALLBACK);
	lua_registerN(L, B0_PUBLISHER_INIT_COMMAND, B0_PUBLISHER_INIT_CALLBACK);
	lua_registerN(L, B0_PUBLISHER_PUBLISH_COMMAND, B0_PUBLISHER_PUBLISH_CALLBACK);

	lua_registerN(L, B0_SUBSCRIBER_NEW_EX_COMMAND, B0_SUBSCRIBER_NEW_EX_CALLBACK);
	lua_registerN(L, B0_SUBSCRIBER_DELETE_COMMAND, B0_SUBSCRIBER_DELETE_CALLBACK);
	lua_registerN(L, B0_SUBSCRIBER_INIT_COMMAND, B0_SUBSCRIBER_INIT_CALLBACK);
	lua_registerN(L, B0_SUBSCRIBER_POLL_COMMAND, B0_SUBSCRIBER_POLL_CALLBACK);
	lua_registerN(L, B0_SUBSCRIBER_READ_COMMAND, B0_SUBSCRIBER_READ_CALLBACK);

	return 1;
}
