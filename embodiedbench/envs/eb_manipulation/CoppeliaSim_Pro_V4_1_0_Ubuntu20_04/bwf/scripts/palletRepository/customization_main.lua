function model.removePallet(palletHandle)
    sim.removeObject(palletHandle)
    model.removeFromPluginRepresentation_onePallet(palletHandle)
    simBWF.markUndoPoint()
end

function model.getPalletHandle(name)
    local l=sim.getObjectsInTree(model.handles.palletHolder,sim.handle_all,1+2)
    for i=1,#l,1 do
        if simBWF.getObjectAltName(l[i])==name then
            return l[i]
        end
    end
    return -1
end

function model.getPalletWithName(name)
    local l=sim.getObjectsInTree(model.handles.palletHolder,sim.handle_all,1+2)
    for i=1,#l,1 do
        if simBWF.getObjectAltName(l[i])==name then
            return l[i]
        end
    end
    return -1
end

function model.getAllPalletHandles()
    return sim.getObjectsInTree(model.handles.palletHolder,sim.handle_all,1+2)
end

function model.addNewPallet()
    local pallet
    local res
    res,pallet=simBWF.query('pallet_createNew')
    if res~='ok' then
        pallet=nil
    end

    local name='PALLET'
    local nameNb=0
    while model.getPalletHandle(name..nameNb)>=0 do
        nameNb=nameNb+1
    end
    name=name..nameNb
    
    local palletDummy=sim.createDummy(0.001)
    sim.setObjectParent(palletDummy,model.handles.palletHolder,true)
    sim.setObjectPosition(palletDummy,model.handles.palletHolder,{0,0,0})
    sim.setObjectOrientation(palletDummy,model.handles.palletHolder,{0,0,0})
    simBWF.setObjectAltName(palletDummy,name)
    
    
    if pallet then
        pallet.id=palletDummy
        pallet.name=name
    else
        -- for testing:
        pallet={}
        pallet.id=palletDummy
        pallet.name=name
        
        pallet.version=1
        pallet.yaw=0
        pallet.pitch=0
        pallet.roll=0
        pallet.acc=100
        pallet.speed=100
        pallet.tabIndex=0

        pallet.palletItemList={} -- empty array (no pallet points in a new pallet)
        
        local allParts=simBWF.getAllPartsFromPartRepository()
        for i=1,4,1 do
            pallet.palletItemList[i]={}
            pallet.palletItemList[i].id=i-1
            pallet.palletItemList[i].version=1
            local theModel=-1
            if #allParts>0 then
                theModel=allParts[1][2]
            end
            pallet.palletItemList[i].model=theModel
            if i==1 or i==3 then
                pallet.palletItemList[i].colorR=1
                pallet.palletItemList[i].locationX=-0.07
            else
                pallet.palletItemList[i].colorR=0.3
                pallet.palletItemList[i].locationX=0.07
            end
            if i==1 or i==2 then
                pallet.palletItemList[i].colorG=0.8
                pallet.palletItemList[i].colorB=0
                pallet.palletItemList[i].locationY=-0.07
            else
                pallet.palletItemList[i].colorG=0
                pallet.palletItemList[i].colorB=0.8
                pallet.palletItemList[i].locationY=0.07
            end
            pallet.palletItemList[i].locationZ=0
            pallet.palletItemList[i].orientationY=0
            pallet.palletItemList[i].orientationP=0
            pallet.palletItemList[i].orientationR=0
        end
        
        pallet.retangularorigoX=0
        pallet.retangularorigoY=0
        pallet.retangularorigoZ=0
        pallet.retangulariRows=1
        pallet.retangulariColumns=1
        pallet.retangulariLayers=1
        pallet.retangularfRowStep=0.1
        pallet.retangularfColumnStep=0.1
        pallet.retangularfLayerStep=0.1

        pallet.honeycomborigoX=0
        pallet.honeycomborigoY=0
        pallet.honeycomborigoZ=0
        pallet.honeycombiRows=1
        pallet.honeycombiColumns=1
        pallet.honeycombiLayers=1
        pallet.honeycombfRowStep=0.1
        pallet.honeycombfColumnStep=0.1
        pallet.honeycombfLayerStep=0.1
        pallet.honeycomboddFirstRow=false

        pallet.circleorigoX=0
        pallet.circleorigoY=0
        pallet.circleorigoZ=0
        pallet.circlefRadius=0
        pallet.circlefAngleOffset=0
        pallet.circleiCircumferenceObjects=1
        pallet.circleiLayers=1
        pallet.circlefLayerStep=0
        pallet.circleitemInCenter=true
    end
    
    simBWF.writePalletInfo(palletDummy,pallet)
    
    model.afterReceivingPalletDataFromPlugin(palletDummy)
    model.updatePluginRepresentation_onePallet(palletDummy)
    return palletDummy,name
end

function model.duplicatePallet(palletHandle)
    local palTable=sim.copyPasteObjects({palletHandle},0)
    local palletDuplicate=palTable[1]
    sim.setObjectParent(palletDuplicate,model.handles.palletHolder,true)
    model.afterReceivingPalletDataFromPlugin(palletDuplicate) -- The plugin didn't send anything. But we want to use the same part references as the original pallet
    local data=simBWF.readPalletInfo(palletDuplicate)
    data.id=palletDuplicate
    data.name=simBWF.getObjectAltName(palletDuplicate)
    simBWF.writePalletInfo(palletDuplicate,data)
    model.updatePluginRepresentation_onePallet(palletDuplicate)
    return palletDuplicate,data.name
end

function model.handleBackCompatibility()
    local allPallets=model.getAllPalletHandles()
    for i=1,#allPallets,1 do
        local h=allPallets[i]
        local name=simBWF.getObjectAltName(h)
        if string.find(name,'__PALLET__')==1 then
            -- old pallet names had a hidden "__PALLET__" prefix. Try to correct for that:
            name=string.sub(name,11)
            simBWF.setObjectAltName(h,name)
        end
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()
    
    model.handleBackCompatibility()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

--[=[
function sysCall_beforeDelete(data)
    -- Check which pallet needs to be updated after object deletion (i.e. part deleted that the pallet refers to)
    model.palletsToUpdateAfterObjectDeletion={}
    local allPallets=model.getAllPalletHandles()
    for i=1,#allPallets,1 do
        local refParts=sim.getReferencedHandles(allPallets[i])
        for j=1,#refParts,1 do
            if refParts[j]>=0 and data.objectHandles[refParts[j]] then
--            print("Found")
                model.palletsToUpdateAfterObjectDeletion[#model.palletsToUpdateAfterObjectDeletion+1]=allPallets[i]
                break
            end
        end
    end
    if #model.palletsToUpdateAfterObjectDeletion==0 then
        model.palletsToUpdateAfterObjectDeletion=nil
    end
end

function sysCall_afterDelete(data)
    if model.palletsToUpdateAfterObjectDeletion then
        for i=1,#model.palletsToUpdateAfterObjectDeletion,1 do
            model.updatePluginRepresentation_onePallet(model.palletsToUpdateAfterObjectDeletion[i])
        end
    end
    model.palletsToUpdateAfterObjectDeletion=nil
end
--]=]
function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if not model.notFirstDuringSimulation then
        model.dlg.updateEnabledDisabledItems()
        model.notFirstDuringSimulation=true
    end
end

function sysCall_afterSimulation()
    model.dlg.updateEnabledDisabledItems()
    model.notFirstDuringSimulation=nil
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
--    if sim.isHandleValid(model.handle)==1 then
        -- The associated model might already have been destroyed (if it destroys itself in the init phase)
        model.removeFromPluginRepresentation()
--    end
    model.dlg.cleanup()
end
