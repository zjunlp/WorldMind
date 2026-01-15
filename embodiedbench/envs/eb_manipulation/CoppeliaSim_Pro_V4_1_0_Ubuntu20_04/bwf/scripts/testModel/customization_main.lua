function model.getAvailableConnections()
    local thisInfo=model.readInfo()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model.handle then
            local data=sim.readCustomDataBlock(l[i],model.tagName)
            if data then
                retL[#retL+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end


function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_beforeSimulation()
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,1)>0 then
        sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,0)
    end
    model.simJustStarted=true
--    model.ext.outputBrSetupMessages()
--    model.ext.outputPluginSetupMessages()
    model.dlg.removeDlg()
end


function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,1)
end


function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end


function sysCall_cleanup()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
