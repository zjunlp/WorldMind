model.ext={}

function model.ext.announcePalletWasDestroyed(palletId)
    -- We go through all parts and adjust them if needed:
    local parts=model.getPartTable()
    for i=1,#parts,1 do
        local part=parts[i][2]
        local data=model.getPartData(part)
        if (data.palletId==palletId) or (palletId==-1) then
            data.palletId=-1
            model.updatePartData(part,data)
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.ext.insertPart(objectHandle)
    sim.removeObjectFromSelection(sim.handle_all,-1)
    model.insertPart(objectHandle)
    model.dlg.removeDlg()
    sim.setBoolParameter(sim.boolparam_br_partrepository,true)
    model.dlg.showOrHideDlgIfNeeded()
    model.dlg.selectedPartId=objectHandle
    model.dlg.refresh()
end

function model.ext.announcePalletWasRenamed()
    model.dlg.refresh()
end

function model.ext.announcePalletWasCreated()
    model.dlg.refresh()
end

function model.ext.announcePalletWasDestroyed()
    model.dlg.refresh()
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




