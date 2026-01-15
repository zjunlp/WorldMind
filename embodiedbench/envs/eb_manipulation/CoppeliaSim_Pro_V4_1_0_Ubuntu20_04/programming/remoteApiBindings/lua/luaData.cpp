#include "luaData.h"
//#include <sstream>
//#include <cstring>

CLuaData::CLuaData()
{
}

CLuaData::~CLuaData()
{

}

void CLuaData::getInputDataForFunctionRegistration(const int* dat,std::vector<int>& outDat)
{
    outDat.clear();
    outDat.push_back(dat[0]);
    for (int i=0;i<dat[0];i++)
        outDat.push_back((dat[1+2*i+0]|SIM_LUA_ARG_NIL_ALLOWED)-SIM_LUA_ARG_NIL_ALLOWED);
}

std::vector<CLuaDataItem>* CLuaData::getInDataPtr()
{
    return(&_inData);
}

bool CLuaData::readDataFromLua(lua_State* L,const int* expectedArguments,int requiredArgumentCount,const char* functionName)
{
    _inData.clear();
    int argCnt=lua_gettop(L);
    if (argCnt<requiredArgumentCount)
    {
        printf("%s: not enough arguments.\n",functionName);
        return(false);
    }

    for (int i=0;i<argCnt;i++)
    {
        if (i>=expectedArguments[0])
            break;
        if (lua_isnil(L,i+1))
        { // We have nil. We never directly expect nil.
            // is nil explicitely allowed?
            if (expectedArguments[1+i*2+0]&SIM_LUA_ARG_NIL_ALLOWED)
            { // yes. This is for an argument that can optionally also be nil.
                CLuaDataItem dat;
                _inData.push_back(dat);
            }
            else
            { // no
                if (int(_inData.size())<requiredArgumentCount)
                {
                    printf("%s: argument %i is not correct.\n",functionName,i+1);
                    return(false);
                }
                break; // this argument is nil, so it is like inexistant. But we also won't explore any more arguments, we have enough.
            }
        }
        else
        {
            int dataType=expectedArguments[1+i*2+0];
            if (dataType&sim_lua_arg_table)
            { // we are expecting a table
                if (!lua_istable(L,i+1))
                {
                    printf("%s: argument %i is not a table.\n",functionName,i+1);
                    return(false);
                }
                dataType=(dataType|sim_lua_arg_table|SIM_LUA_ARG_NIL_ALLOWED)-(sim_lua_arg_table|SIM_LUA_ARG_NIL_ALLOWED);
                int tableSize=int(lua_objlen(L,i+1));
                if (tableSize<expectedArguments[1+i*2+1])
                {
                    printf("%s: argument %i (a table) does not contain enough elements.\n",functionName,i+1);
                    return(false);
                }
                std::vector<bool> bools;
                std::vector<int> ints;
                std::vector<float> floats;
                std::vector<double> doubles;
                std::vector<std::string> strings;
                for (int j=0;j<tableSize;j++)
                {
                    lua_rawgeti(L,i+1,j+1);
                    if (dataType==sim_lua_arg_bool)
                    {
                        if (!lua_isboolean(L,-1))
                        {
                            lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                            printf("%s: argument %i (a table) contains invalid elements.\n",functionName,i+1);
                            return(false);
                        }
                        bools.push_back(lua_toboolean(L,-1)!=0);
                    }
                    if (dataType==sim_lua_arg_int)
                    {
                        if (!lua_isnumber(L,-1))
                        {
                            lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                            printf("%s: argument %i (a table) contains invalid elements.\n",functionName,i+1);
                            return(false);
                        }
                        ints.push_back((int)lua_tointeger(L,-1));
                    }
                    if (dataType==sim_lua_arg_float)
                    {
                        if (!lua_isnumber(L,-1))
                        {
                            lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                            printf("%s: argument %i (a table) contains invalid elements.\n",functionName,i+1);
                            return(false);
                        }
                        floats.push_back((float)lua_tonumber(L,-1));
                    }
                    if (dataType==sim_lua_arg_double)
                    {
                        if (!lua_isnumber(L,-1))
                        {
                            lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                            printf("%s: argument %i (a table) contains invalid elements.\n",functionName,i+1);
                            return(false);
                        }
                        doubles.push_back((double)lua_tonumber(L,-1));
                    }
                    if (dataType==sim_lua_arg_string)
                    {
                        if (!lua_isstring(L,-1))
                        {
                            lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                            printf("%s: argument %i (a table) contains invalid elements.\n",functionName,i+1);
                            return(false);
                        }
                        strings.push_back(std::string(lua_tostring(L,-1)));
                    }
                    lua_pop(L,1); // we have to pop the value that was pushed with lua_rawgeti
                }
                if (dataType==sim_lua_arg_bool)
                    _inData.push_back(CLuaDataItem(bools));
                if (dataType==sim_lua_arg_int)
                    _inData.push_back(CLuaDataItem(ints));
                if (dataType==sim_lua_arg_float)
                    _inData.push_back(CLuaDataItem(floats));
                if (dataType==sim_lua_arg_double)
                    _inData.push_back(CLuaDataItem(doubles));
                if (dataType==sim_lua_arg_string)
                    _inData.push_back(CLuaDataItem(strings));
            }
            else
            { // we are not expecting a table
                dataType=(dataType|SIM_LUA_ARG_NIL_ALLOWED)-SIM_LUA_ARG_NIL_ALLOWED;
                if (dataType==sim_lua_arg_bool)
                {
                    if (!lua_isboolean(L,i+1))
                    {
                        printf("%s: argument %i is not a boolean.\n",functionName,i+1);
                        return(false);
                    }
                    _inData.push_back(CLuaDataItem(lua_toboolean(L,i+1)!=0));
                }
                if (dataType==sim_lua_arg_int)
                {
                    if (!lua_isnumber(L,i+1))
                    {
                        printf("%s: argument %i is not a number.\n",functionName,i+1);
                        return(false);
                    }
                    _inData.push_back(CLuaDataItem((int)lua_tointeger(L,i+1)));
                }
                if (dataType==sim_lua_arg_float)
                {
                    if (!lua_isnumber(L,i+1))
                    {
                        printf("%s: argument %i is not a number.\n",functionName,i+1);
                        return(false);
                    }
                    _inData.push_back(CLuaDataItem((float)lua_tonumber(L,i+1)));
                }
                if (dataType==sim_lua_arg_double)
                {
                    if (!lua_isnumber(L,i+1))
                    {
                        printf("%s: argument %i is not a number.\n",functionName,i+1);
                        return(false);
                    }
                    _inData.push_back(CLuaDataItem((double)lua_tonumber(L,i+1)));
                }
                if (dataType==sim_lua_arg_string)
                {
                    if (!lua_isstring(L,i+1))
                    {
                        printf("%s: argument %i is not a string.\n",functionName,i+1);
                        return(false);
                    }
                    _inData.push_back(CLuaDataItem(std::string(lua_tostring(L,i+1))));
                }
                if (dataType==sim_lua_arg_charbuff)
                {
                    if (!lua_isstring(L,i+1))
                    {
                        printf("%s: argument %i is not a string.\n",functionName,i+1);
                        return(false);
                    }
                    size_t dataLength;
                    char* data=(char*)lua_tolstring(L,i+1,&dataLength);
                    _inData.push_back(CLuaDataItem(data,(int)dataLength));
                }
            }
        }
    }
    return(true);
}

void CLuaData::pushOutData(const CLuaDataItem& dataItem)
{
    _outData.push_back(dataItem);
}

int CLuaData::writeDataToLua(lua_State* L)
{
    int itemCnt=int(_outData.size());
    for (int item=0;item<itemCnt;item++)
    {
        if (_outData[item].isTable())
        { // we have a table here
            lua_newtable(L);
            int newTablePos=lua_gettop(L);
            if (_outData[item].getType()==-1)
            { // nil table
                for (int i=0;i<_outData[item].getNilTableSize();i++)
                {
                    lua_pushnil(L);
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
            if (_outData[item].getType()==0)
            { // bool table
                for (unsigned int i=0;i<_outData[item].boolData.size();i++)
                {
                    lua_pushboolean(L,_outData[item].boolData[i]);
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
            if (_outData[item].getType()==1)
            { // int table
                for (unsigned int i=0;i<_outData[item].intData.size();i++)
                {
                    lua_pushinteger(L,_outData[item].intData[i]);
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
            if (_outData[item].getType()==2)
            { // float table
                for (unsigned int i=0;i<_outData[item].floatData.size();i++)
                {
                    lua_pushnumber(L,_outData[item].floatData[i]);
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
            if (_outData[item].getType()==5)
            { // double table
                for (unsigned int i=0;i<_outData[item].doubleData.size();i++)
                {
                    lua_pushnumber(L,_outData[item].doubleData[i]);
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
            if (_outData[item].getType()==3)
            { // string table
                for (unsigned int i=0;i<_outData[item].stringData.size();i++)
                {
                    lua_pushstring(L,_outData[item].stringData[i].c_str());
                    lua_rawseti(L,newTablePos,i+1);
                }
            }
        }
        else
        { // no table here
            if (_outData[item].getType()==-1)
            { // nil 
                lua_pushnil(L);
            }
            if (_outData[item].getType()==0)
            { // bool 
                lua_pushboolean(L,_outData[item].boolData[0]);
            }
            if (_outData[item].getType()==1)
            { // int 
                lua_pushinteger(L,_outData[item].intData[0]);
            }
            if (_outData[item].getType()==2)
            { // float 
                lua_pushnumber(L,_outData[item].floatData[0]);
            }
            if (_outData[item].getType()==5)
            { // double 
                lua_pushnumber(L,_outData[item].doubleData[0]);
            }
            if (_outData[item].getType()==3)
            { // string 
                lua_pushstring(L,_outData[item].stringData[0].c_str());
            }
            if (_outData[item].getType()==4)
            { // buffer 
                lua_pushlstring(L,_outData[item].stringData[0].c_str(),_outData[item].stringData[0].size());
            }
        }
    }
    return(itemCnt);
}
