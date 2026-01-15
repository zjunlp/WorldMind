function sysCall_beforeCopy(inData)
    -- called before objects are copied. see also sysCall_afterCopy
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be copied")
    end
end
