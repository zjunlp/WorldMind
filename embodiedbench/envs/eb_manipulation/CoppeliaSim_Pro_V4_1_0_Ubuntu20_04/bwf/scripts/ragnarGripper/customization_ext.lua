model.ext={}

function model.ext.getItemData_pricing()
    local obj={}
    obj.name=simBWF.getObjectAltName(model.handle)
    obj.type='ragnarGripper'
    local c=model.readInfo()
    obj.gripperType=c.subtype
    obj.brVersion=1
    return obj
end

function model.ext.attachOrDetachDetectedPart(dat)
    local gripperAction=dat[1]
    -- gripperAction: 0=opened (place all items), 1=closed (pick new item. Previously picked items will remain attached)
    local platform=dat[5]
    local robotRef=dat[6]
    
    local platformM=sim.getObjectMatrix(platform,-1)

    local attach=gripperAction==1
    local parts={}
    local newParent=model.handles.attachPt
    if attach then
        local allParts=simBWF.getAllInstanciatedParts()
        for i=1,#allParts,1 do
            local part=allParts[i]
            local data=simBWF.readPartInfo(part)
            -- Put the platform into its picking pose (part velocity corrected):
            sim.setObjectPosition(platform,robotRef,dat[2])
            sim.setObjectOrientation(platform,robotRef,dat[3])
            local p=sim.getObjectPosition(platform,-1)
            p={p[1]+data.vel[1]*dat[4],p[2]+data.vel[2]*dat[4],p[3]+data.vel[3]*dat[4]}
            sim.setObjectPosition(platform,-1,p)
            
            if model.isPartDetected(part) then
                parts[1]=part
                -- Remember the previous part parent:
                data.previousParentParent=sim.getObjectParent(part)
                simBWF.writePartInfo(part,data)
                break
            end
        end
    else
        parts=sim.getObjectsInTree(model.handles.attachPt,sim.handle_all,1+2) -- get all first-level children of the model.handles.attachPt
    end
    for i=1,#parts,1 do
        local part=parts[i]
        local p=sim.getModelProperty(part)
        -- Make the item dynamic, respondable and detectable again (detaching), or disable those flags (attaching):
        p=sim.boolOr32(p,sim.modelproperty_not_dynamic+sim.modelproperty_not_respondable+sim.modelproperty_not_detectable)
        if not attach then
            p=p-(sim.modelproperty_not_dynamic+sim.modelproperty_not_respondable+sim.modelproperty_not_detectable)
        end
        sim.setModelProperty(part,p)
        local data=simBWF.readPartInfo(part)
        if not attach then
            -- detaching
            newParent=data.previousParentParent
            data.previousParentParent=nil
            if sim.boolAnd32(data.bitCoded,16)>0 then
                data.bitCoded=data.bitCoded-16
                data.attachStartCmd=sim.getSimulationTime()
            end
            simBWF.writePartInfo(part,data)
            -- Set platform into actual drop pose:
            sim.setObjectPosition(platform,robotRef,dat[2])
            sim.setObjectOrientation(platform,robotRef,dat[3])
        else
            -- Put the platform into its picking pose (part velocity corrected):
            sim.setObjectPosition(platform,robotRef,dat[2])
            sim.setObjectOrientation(platform,robotRef,dat[3])
            local p=sim.getObjectPosition(platform,-1)
            p={p[1]+data.vel[1]*dat[4],p[2]+data.vel[2]*dat[4],p[3]+data.vel[3]*dat[4]}
            sim.setObjectPosition(platform,-1,p)
        end
        sim.setObjectParent(part,newParent,true)
    end
    sim.setObjectMatrix(platform,-1,platformM) -- restore the original platform pose
    return #parts>0
end

function model.ext.outputBrSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local platforms=sim.getObjectsWithTag(simBWF.modelTags.RAGNARGRIPPERPLATFORM,true)
    local present=false
    for i=1,#platforms,1 do
        if simBWF.callCustomizationScriptFunction_noError('model.ext.checkIfPlatformIsAssociatedWithGripper',platforms[i],model.handle) then
            present=true
            break
        end
    end
    local msg=""
    if not present then
        msg="WARNING (set-up): Not attached to any gripper platform"..nm
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.ext.outputPluginSetupMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    local data={}
    data.id=model.handle
    local result,msgs=simBWF.query('get_objectSetupMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs.messages[i]..nm
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.ext.outputPluginRuntimeMessages()
    local nm=' ['..simBWF.getObjectAltName(model.handle)..']'
    local msg=""
    local data={}
    data.id=model.handle
    local result,msgs=simBWF.query('get_objectRuntimeMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs.messages[i]..nm
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end


function model.ext.readInfo()
    return model.readInfo()
end

function model.ext.replaceInfo(data)
    model.writeInfo(data)
    model.updateAppearance()
end

function model.ext.refreshDlg()
    if model.dlg then
        model.dlg.refresh()
    end
end
---------------------------------------------------------------
-- SERIALIZATION (e.g. for replacement of old with new models):
---------------------------------------------------------------

function model.ext.getSerializationData()
    local data={}
    data.objectName=sim.getObjectName(model.handle)
    data.objectAltName=sim.getObjectName(model.handle+sim.handleflag_altname)
    data.matrix=sim.getObjectMatrix(model.handle,-1)
    local parentHandle=sim.getObjectParent(model.handle)
    if parentHandle>=0 then
        data.parentName=sim.getObjectName(parentHandle)
    end
    data.embeddedData=model.readInfo()
    
end

function model.ext.applySerializationData(data)
end
