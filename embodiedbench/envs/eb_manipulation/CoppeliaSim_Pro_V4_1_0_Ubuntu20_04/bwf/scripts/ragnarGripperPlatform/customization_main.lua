function model.handleBackCompatibility()
    -- In a previous version, dummies previously involved in ik-ik links (for the branches)
    -- were not correctly removed when detaching the platform. Correct for that here:
    for i=1,#model.handles.ikPts,1 do
        local h=model.handles.ikPts[i]
        local dum=sim.getObjectsInTree(h,sim.object_dummy_type,1)
        for j=1,#dum,1 do
            local h2=dum[j]
            local h3=sim.getLinkDummy(h2)
            if h3==-1 then
                sim.removeObject(h2)
            end
        end
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()
    
    model.handleBackCompatibility()

   for i=1,4,1 do
        local data={}
        data.index=i
        sim.writeCustomDataBlock(model.handles.ikPts[i],simBWF.modelTags.RAGNARGRIPPERPLATFORMIKPT,sim.packTable(data))
    end

    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_beforeSimulation()
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    model.dlg.removeDlg()
end

function sysCall_sensing()
    model.ext.outputPluginRuntimeMessages()
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
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end
