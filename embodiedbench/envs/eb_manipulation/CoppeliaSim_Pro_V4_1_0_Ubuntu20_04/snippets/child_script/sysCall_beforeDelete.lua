function sysCall_beforeDelete(inData)
    -- Called before objects are deleted. See also sysCall_afterDelete
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be deleted")
    end
    -- inData.allObjects indicates if all objects in the scene will be deleted
end
