function sysCall_afterCopy(inData)
    -- called after objects have been copied. see also sysCall_beforeCopy
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was copied")
    end
end
