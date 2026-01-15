#pragma once

#include "luaDataItem.h"
#include "simConst.h"

extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#define SIM_LUA_ARG_NIL_ALLOWED (65536)

class CLuaData  
{
public:
    CLuaData();
    virtual ~CLuaData();

    bool readDataFromLua(lua_State* L,const int* expectedArguments,int requiredArgumentCount,const char* functionName);
    std::vector<CLuaDataItem>* getInDataPtr();

    void pushOutData(const CLuaDataItem& dataItem);
    int writeDataToLua(lua_State* L);

    static void getInputDataForFunctionRegistration(const int* dat,std::vector<int>& outDat);


protected:
    std::vector<CLuaDataItem> _inData;
    std::vector<CLuaDataItem> _outData;
};
