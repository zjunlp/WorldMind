#pragma once

#include <vector>
#include <string>

class CLuaDataItem
{
public:
    CLuaDataItem();
    CLuaDataItem(bool v);
    CLuaDataItem(int v);
    CLuaDataItem(float v);
    CLuaDataItem(double v);
    CLuaDataItem(const std::string& v);
    CLuaDataItem(const char* bufferPtr,unsigned int bufferLength);

    CLuaDataItem(const std::vector<bool>& v);
    CLuaDataItem(const std::vector<int>& v);
    CLuaDataItem(const std::vector<float>& v);
    CLuaDataItem(const std::vector<double>& v);
    CLuaDataItem(const std::vector<std::string>& v);

    virtual ~CLuaDataItem();

    bool isTable();
    int getType();
    void setNilTable(int size);
    int getNilTableSize();

    std::vector<bool> boolData;
    std::vector<int> intData;
    std::vector<float> floatData;
    std::vector<double> doubleData;
    std::vector<std::string> stringData;

protected:
    int _nilTableSize;
    bool _isTable;
    int _type; // -1=nil,0=bool,1=int,2=float,3=string,4=buffer,5=double
};
