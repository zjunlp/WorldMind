function model.signalWasToggled(lowS)
    if not lowS then
        sim.setObjectPosition(model.handles.sigPart,sim.handle_parent,{0,0,-0.02})
    else
        sim.setObjectPosition(model.handles.sigPart,sim.handle_parent,{0,0,0.02})
    end
end

function model.readAndApplySignalState()
    local c=model.readInfo()
    local data={}
    data.id=model.handle

    local resp,retData
    if simBWF.isInTestMode() then
        resp='ok'
        retData={}
        retData.state=math.floor(0.5+math.mod(sim.getSystemTimeInMs(-1)/2000,1))==1
    else
        resp,retData=simBWF.query('outbox_getState',data)
        if resp~='ok' then
            retData.state=false
        end
    end
    if retData.state~=c.signalState then
        c.signalState=retData.state
        model.writeInfo(c)
        model.signalWasToggled(c.signalState)
        model.dlg.refresh()
    end
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
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        model.readAndApplySignalState()
    end
end

function sysCall_beforeSimulation()
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,1)>0 then
        sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(model.handles.sigPart,sim.objintparam_visibility_layer,0)
    end
    model.simJustStarted=true
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
--    model.dlg.removeDlg()
end

function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.ext.outputPluginRuntimeMessages()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
    model.readAndApplySignalState()
end

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model.handles.body,sim.objintparam_visibility_layer,1)
    sim.setObjectInt32Parameter(model.handles.sigPart,sim.objintparam_visibility_layer,1)
    model.dlg.updateEnabledDisabledItems()
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
