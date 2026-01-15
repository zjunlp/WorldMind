function model.setPlatformColor(col)
    if model.platformShape then
       sim.setShapeColor(model.platformShape,'RAGNARPLATFORM',sim.colorcomponent_ambient,col)
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.gripper=-1
    local objs=sim.getObjectsInTree(model.handle,sim.handle_all,1)
    for i=1,#objs,1 do
        if sim.readCustomDataBlock(objs[i],simBWF.modelTags.RAGNARGRIPPER) then
            model.gripper=objs[i]
            break
        end
    end
    
    model.modelData=model.readInfo()
    if sim.boolAnd32(model.modelData.bitCoded,1)>0 then
        local obj=sim.getObjectsInTree(model.handle,sim.object_shape_type,0)
        for i=1,#obj,1 do
            local res,col=sim.getShapeColor(obj[i],'RAGNARPLATFORM',sim.colorcomponent_ambient)
            if res>0 then
                model.platformOriginalCol=col
                model.platformShape=obj[i]
                break
            end
        end
    end
    model.prevState=-1
end

function sysCall_sensing()
    if sim.boolAnd32(model.modelData.bitCoded,1)>0 then
        local data={}
        data.id=model.gripper
        local res,retDat=simBWF.query('get_gripperState',data)
        local state=-1
        if res=='ok' then
            state=retDat.gripperState
        end
        if simBWF.isInTestMode() then
            state=1
        end
        if state~=model.prevState then
            if state==-1 then
                model.setPlatformColor(model.platformOriginalCol)
            end
            if state==0 then
                model.setPlatformColor({0,0.5,1})
            end
            if state==1 then
                model.setPlatformColor({1,0,0})
            end
            model.prevState=state
        end
    end
end

function sysCall_cleanup()
    if sim.boolAnd32(model.modelData.bitCoded,1)>0 then
        model.setPlatformColor(model.platformOriginalCol)
    end
end

