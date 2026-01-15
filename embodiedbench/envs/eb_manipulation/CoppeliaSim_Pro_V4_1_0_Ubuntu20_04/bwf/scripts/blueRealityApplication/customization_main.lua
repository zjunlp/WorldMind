function query(cmd,...)
    local args={...}
    if cmd=='printMsg' then
        simBWF.outputMessage(args[1],args[2])
    end
    if cmd=='displayModalMsg' then
        local t=sim.msgbox_type_info
        local title='Message'
        if args[2]==1 then
            t=sim.msgbox_type_warning
            title='Warning'
        end
        if args[2]==2 then
            t=sim.msgbox_type_critical
            title='Error'
        end
        sim.msgBox(t,sim.msgbox_buttons_ok,title,args[1])
    end
end

function model.startSimulation()
    if not model.simulation then
        sim.setBoolParameter(sim.boolparam_realtime_simulation,false)
        sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,0.8,1})
        model.simulation=true
        simBWF.query('simulation_start',{})
    end
end

function model.stopSimulation()
    if model.simulation then
        sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,0.87,0.92})
        model.simulation=false
        simBWF.query('simulation_stop',{})
    end
end

function model.outputGeneralSetupMessages()
    local msg=""
    local data={}
    data.id=-1
    local result,msgs=simBWF.query('get_objectSetupMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs[i]
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.outputGeneralRuntimeMessages()
    local msg=""
    local data={}
    data.id=-1
    local result,msgs=simBWF.query('get_objectRuntimeMessages',data)
    if result=='ok' then
        for i=1,#msgs.messages,1 do
            if msg~='' then
                msg=msg..'\n'
            end
            msg=msg..msgs[i]
        end
    end
    if #msg>0 then
        simBWF.outputMessage(msg,simBWF.MSG_WARN)
    end
end

function model.verifyLayout()
    sim.clearStringSignal('__brMessages__')
    if messageConsole then
        sim.auxiliaryConsoleClose(messageConsole)
        messageConsole=nil
    end
    -- General setup messages:
    model.outputGeneralSetupMessages()
    -- Object-specific setup messages:
    local tags=simBWF.getModelTagsForMessages()
    for i=1,#tags,1 do
        local objs=sim.getObjectsWithTag(tags[i],true)
        for j=1,#objs,1 do
            simBWF.callCustomizationScriptFunction_noError('model.ext.outputBrSetupMessages',objs[j])
            simBWF.callCustomizationScriptFunction_noError('model.ext.outputPluginSetupMessages',objs[j])
        end
    end
end

function model.handleOutputMessageDisplay()
    local msgs=sim.getStringSignal('__brMessages__')
    if msgs then
        local c=model.readInfo()
        msgs=sim.unpackTable(msgs)
        local txt=''
        for i=1,#msgs,1 do
            if msgs[i][2]~=simBWF.MSG_WARN or sim.boolAnd32(c.bitCoded,32)>0 then
                txt=txt..msgs[i][1]..'\n'
            end
        end
        if txt~='' then
            if not messageConsole or sim.auxiliaryConsoleShow(messageConsole,1)<=0 then
                messageConsole=sim.auxiliaryConsoleOpen('Messages',400,4,nil,{800,400})
            end
            sim.auxiliaryConsolePrint(messageConsole,txt)
        end
        sim.clearStringSignal('__brMessages__')
    end
end

function model.refreshAllDialogs()
    local exceptions={}
    exceptions[simBWF.modelTags.BLUEREALITYAPP]=true
    for key,value in pairs(simBWF.modelTags) do
        if not exceptions[value] then
            local objs=sim.getObjectsWithTag(value,true)
            for i=1,#objs,1 do
                simBWF.callCustomizationScriptFunction_noError('model.ext.refreshDlg',objs[i])
            end
        end
    end
end

function sysCall_beforeCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        local tag=simBWF.getModelMainTag(key)
        if tag~='' then
            simBWF.markModelAsCopy(key,true) -- the original model is marked as copy, just before being copied, then...
        end
    end
end

function sysCall_afterCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        local tag=simBWF.getModelMainTag(key)
        if tag~='' then
            simBWF.markModelAsCopy(key,false) -- the original model is unmarked as copy (only the copy remains marked as 'copy')
        end
    end
end

function sysCall_afterDelete(inData)
    if sim.getSimulationState()==sim.simulation_stopped then
        model.refreshAllDialogs()
    end
--    for handle,v in pairs(inData.objectHandles) do
--    end
end

function sysCall_afterCreate(inData)
    if sim.getSimulationState()==sim.simulation_stopped then
        model.refreshAllDialogs()
    end
--    for handle,v in pairs(inData.objectHandles) do
--    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.json=require("dkjson")
    model.http=require("socket.http")
    model.ltn12=require("ltn12")

    sim.setBoolParameter(sim.boolparam_br_jobfunc,true)
    sim.setBoolParameter(sim.boolparam_online_mode,false)
    simBWF.query('online_toggle',{state=false})

    model.online=false
    model.onlineSwitch=nil
    model.simulation=false
    sim.setIntegerSignal('__brUndoPointCounter__',0)
    model.previousUndoPointCounter=0
    model.undoPointStayedSameCounter=-1
    
    model.floor.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")

    
    -- Make sure there is not more than one of such model in the scene:
    local objs=sim.getObjectsWithTag(model.tagName,true)
    if #objs>1 then
        sim.removeModel(model.handle)
        sim.removeObjectFromSelection(sim.handle_all)
        objs=sim.getObjectsWithTag(model.tagName,true)
        sim.addObjectToSelection(sim.handle_single,objs[1])
    else
        model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
        model.updatePluginRepresentation_brApp()
        model.updatePluginRepresentation_generalProperties()
    end
end

function sysCall_nonSimulation()
    model.floor.showOrHideDlgIfNeeded()
    if model.dev then
        model.dev.showOrHideDlgIfNeeded()
    end
    
    -- Following is the central part where we set undo points:
    ---------------------------------
    local cnt=sim.getIntegerSignal('__brUndoPointCounter__')
    if cnt~=model.previousUndoPointCounter then
        model.undoPointStayedSameCounter=8
        model.previousUndoPointCounter=cnt
    end
    if model.undoPointStayedSameCounter>0 then
        model.undoPointStayedSameCounter=model.undoPointStayedSameCounter-1
    else
        if model.undoPointStayedSameCounter==0 then
            sim.announceSceneContentChange() -- to have an undo point
            model.undoPointStayedSameCounter=-1
        end
    end
    ---------------------------------
    
    model.actions.quoteRequest_executeIfNeeded()
    model.actions.roiRequest_executeIfNeeded()
    model.actions.sopRequest_executeIfNeeded()
    
    local onlSw=sim.getBoolParameter(sim.boolparam_online_mode)
    if model.onlineSwitch~=onlSw then
        simBWF.query('online_toggle',{state=onlSw})
        model.onlineSwitch=onlSw
        simBWF.announceOnlineModeChanged(model.onlineSwitch)
    end
    model.handleOutputMessageDisplay()
    model.updatePluginRepresentation_generalProperties()
end

function sysCall_sensing()
    model.outputGeneralRuntimeMessages()
    model.handleOutputMessageDisplay()
    
    local data={}
    local allCams=sim.getObjectsInTree(sim.handle_scene,sim.object_camera_type)
    for i=1,#allCams,1 do
        data[allCams[i]]=sim.getObjectMatrix(allCams[i],-1)
    end
    model.cameraMatrices=data
end

function sysCall_afterSimulation()
    simBWF.query('simulation_toggle',{run=false})
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        if model.online then
            sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,0.87,0.92})
            sim.clearIntegerSignal('__brOnline__')
            model.online=false
--            simBWF.query('online_stop',{})
        end
    else
        if model.simulation then
            sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,0.87,0.92})
            model.simulation=false
--            simBWF.query('simulation_stop',{})
        end
    end
    sim.setObjectInt32Parameter(model.handle,sim.objintparam_visibility_layer,1)
    if messageConsole then
        sim.auxiliaryConsoleClose(messageConsole)
        messageConsole=nil
    end
    sim.clearStringSignal('__brMessages__')
    
    local c=model.readInfo()
    if sim.boolAnd32(c.bitCoded,64)>0 then
        local allCams=sim.getObjectsInTree(sim.handle_scene,sim.object_camera_type)
        for i=1,#allCams,1 do
            local m=model.cameraMatrices[allCams[i]]
            if m then
                sim.setObjectMatrix(allCams[i],-1,m)
            end
        end
    end
    model.cameraMatrices=nil
end

function sysCall_beforeSimulation()
    if messageConsole then
        sim.auxiliaryConsoleClose(messageConsole)
        messageConsole=nil
    end
    model.outputGeneralSetupMessages()
    
    model.floor.removeDlg()
    if model.dev then
        model.dev.removeDlg()
    end
    sim.setObjectInt32Parameter(model.handle,sim.objintparam_visibility_layer,0)
    if sim.getBoolParameter(sim.boolparam_online_mode) then
        if not model.online then
            sim.setBoolParameter(sim.boolparam_realtime_simulation,true)
            sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,1,0.8})
            sim.setIntegerSignal('__brOnline__',1)
            model.online=true
    --        simBWF.query('online_start',{})
        end
    else
        if not model.simulation then
            sim.setBoolParameter(sim.boolparam_realtime_simulation,false)
            sim.setArrayParameter(sim.arrayparam_background_color2,{0.8,0.8,1})
            model.simulation=true
    --        simBWF.query('simulation_start',{})
        end
    end
    simBWF.query('simulation_toggle',{run=true})
end

function sysCall_beforeInstanceSwitch()
    sim.clearStringSignal('__brMessages__')
    if messageConsole then
        sim.auxiliaryConsoleClose(messageConsole)
        messageConsole=nil
    end
    model.floor.removeDlg()
    if model.dev then
        model.dev.removeDlg()
    end
    model.removeFromPluginRepresentation_brApp()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation_brApp()
    model.updatePluginRepresentation_generalProperties()
end

function sysCall_cleanup()
    model.floor.removeDlg()
    if model.dev then
        model.dev.removeDlg()
    end
--    simBWF.announcePalletWasDestroyed(-1) -- all pallets were destroyed
--    simBWF.announcePalletsHaveBeenUpdated({})
    if sim.isHandleValid(model.handle)==1 then
        -- The associated model might already have been destroyed (if it destroys itself in the init phase)
        model.removeFromPluginRepresentation_brApp()
        simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.floor.previousDlgPos)
    end
end

function sysCall_xr(brData)
    local brCallIndex=brData.xrCallIndex
    if (brCallIndex==0) then
        model.actions.variousActionDlg()
    end
    if (brCallIndex==3) then
        sim.setBoolParameter(sim.boolparam_br_partrepository,not sim.getBoolParameter(sim.boolparam_br_partrepository))
    end
    if (brCallIndex==4) then
        sim.setBoolParameter(sim.boolparam_br_palletrepository,not sim.getBoolParameter(sim.boolparam_br_palletrepository))
    end
    if (brCallIndex==5) then
        model.generalProperties.openDlg()
    end
    if (brCallIndex==11) then
        model.verifyLayout()
    end
    if (brCallIndex==model.brCalls.NEWJOB) then
        model.createNewJob()
    end
    if (brCallIndex==model.brCalls.DELETEJOB) then
        model.deleteJob()
    end
    if (brCallIndex==model.brCalls.RENAMEJOB) then
        model.renameJob()
    end
    if (brCallIndex>=model.brCalls.SWITCHJOB) then
        model.switchJob()
    end
end
