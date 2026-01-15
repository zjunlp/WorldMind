function model.removeFromPluginRepresentation_onePallet(palletHandle)
    local data={}
    data.id=palletHandle
    simBWF.query('object_delete',data)
    model._previousPackedPluginDatas[palletHandle]=nil
end

function model.updatePluginRepresentation_onePallet(palletHandle)
    model.beforeSendingPalletDataToPlugin(palletHandle)
    local data=simBWF.readPalletInfo(palletHandle)
    data.id=palletHandle
    data.name=simBWF.getObjectAltName(palletHandle)

    if model._previousPackedPluginDatas==nil then
        model._previousPackedPluginDatas={}
    end
    local packedData=sim.packTable(data)
    if packedData~=model._previousPackedPluginDatas[palletHandle] then -- update the plugin only if the data is different from last time here
        model._previousPackedPluginDatas[palletHandle]=packedData
        simBWF.query('pallet_update',data)
    end
end

function model.removeFromPluginRepresentation()
    local allPallets=model.getAllPalletHandles()
    for i=1,#allPallets,1 do
        model.removeFromPluginRepresentation_onePallet(allPallets[i])
    end
end

function model.updatePluginRepresentation()
    local allPallets=model.getAllPalletHandles()
    for i=1,#allPallets,1 do
        model.updatePluginRepresentation_onePallet(allPallets[i])
    end
end

function model.afterReceivingPalletDataFromPlugin(palletHandle)
    -- We store the pallet item models as referenced objects instead:
    local pallet=simBWF.readPalletInfo(palletHandle)
    pallet.id=palletHandle
    simBWF.writePalletInfo(palletHandle,pallet)
    local refParts={}
    for i=1,#pallet.palletItemList,1 do
        local partId=pallet.palletItemList[i].model
        if partId<0 then
            partId=-1
        end
        refParts[i]=partId
    end
    sim.setReferencedHandles(palletHandle,refParts)
end

function model.beforeSendingPalletDataToPlugin(palletHandle)
    local pallet=simBWF.readPalletInfo(palletHandle)
    local refParts=sim.getReferencedHandles(palletHandle)
    for i=1,#pallet.palletItemList,1 do
        if i<=#refParts then
            pallet.palletItemList[i].model=refParts[i]
        end
    end
    pallet.id=palletHandle
    simBWF.writePalletInfo(palletHandle,pallet)
end

-------------------------------------------------------
-- JOBS:
-------------------------------------------------------

function model.handleJobConsistency(removeJobsExceptCurrent)
    -- Make sure stored jobs are consistent with current scene:


    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.createNewJob()
    -- Create new job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.deleteJob()
    -- Delete current job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.renameJob()
    -- Rename job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end

function model.switchJob()
    -- Switch job menu bar cmd
    local oldJob=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
end


