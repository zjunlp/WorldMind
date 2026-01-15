function sysCall_afterCreate(inData)
    -- Called after objects have been created
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..value.." was created")
    end
end
