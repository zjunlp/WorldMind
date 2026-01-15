function sysCall_afterDelete(inData)
    -- Called after objects were deleted. See also sysCall_beforeDelete
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was deleted")
    end
    -- inData.allObjects indicates if all objects in the scene were deleted
end
