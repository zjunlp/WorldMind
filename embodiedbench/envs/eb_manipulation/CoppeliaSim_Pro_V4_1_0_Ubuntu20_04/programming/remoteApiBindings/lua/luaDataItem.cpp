#include "luaDataItem.h"

CLuaDataItem::CLuaDataItem()
{
    _nilTableSize=0;
    _isTable=false;
    _type=-1; // nil
}

CLuaDataItem::CLuaDataItem(bool v)
{
    _nilTableSize=0;
    _isTable=false;
    _type=0;
    boolData.push_back(v);
}

CLuaDataItem::CLuaDataItem(int v)
{
    _nilTableSize=0;
    _isTable=false;
    _type=1;
    intData.push_back(v);
}

CLuaDataItem::CLuaDataItem(float v)
{
    _nilTableSize=0;
    _isTable=false;
    _type=2;
    floatData.push_back(v);
}

CLuaDataItem::CLuaDataItem(double v)
{
    _nilTableSize=0;
    _isTable=false;
    _type=5;
    doubleData.push_back(v);
}

CLuaDataItem::CLuaDataItem(const std::string& v)
{
    _nilTableSize=0;
    _isTable=false;
    _type=3;
    stringData.push_back(v);
}

CLuaDataItem::CLuaDataItem(const char* bufferPtr,unsigned int bufferLength)
{
    _nilTableSize=0;
    _isTable=false;
    _type=4;
    std::string v(bufferPtr,bufferLength);
    stringData.push_back(v);
}

CLuaDataItem::CLuaDataItem(const std::vector<bool>& v)
{
    _nilTableSize=0;
    _isTable=true;
    _type=0;
    boolData.assign(v.begin(),v.end());
}

CLuaDataItem::CLuaDataItem(const std::vector<int>& v)
{
    _nilTableSize=0;
    _isTable=true;
    _type=1;
    intData.assign(v.begin(),v.end());
}

CLuaDataItem::CLuaDataItem(const std::vector<float>& v)
{
    _nilTableSize=0;
    _isTable=true;
    _type=2;
    floatData.assign(v.begin(),v.end());
}

CLuaDataItem::CLuaDataItem(const std::vector<double>& v)
{
    _nilTableSize=0;
    _isTable=true;
    _type=5;
    doubleData.assign(v.begin(),v.end());
}

CLuaDataItem::CLuaDataItem(const std::vector<std::string>& v)
{
    _nilTableSize=0;
    _isTable=true;
    _type=3;
    stringData.assign(v.begin(),v.end());
}

CLuaDataItem::~CLuaDataItem()
{
}

bool CLuaDataItem::isTable()
{
    return(_isTable);
}

int CLuaDataItem::getType()
{
    return(_type);
}

void CLuaDataItem::setNilTable(int size)
{
    if (_type==-1)
    {
        _isTable=true;
        _nilTableSize=size;
    }
}

int CLuaDataItem::getNilTableSize()
{
    return(_nilTableSize);
}
