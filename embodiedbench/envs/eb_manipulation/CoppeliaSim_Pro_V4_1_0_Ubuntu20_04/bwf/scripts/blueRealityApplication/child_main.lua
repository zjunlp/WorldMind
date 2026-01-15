function sysCall_init()
    model.codeVersion=1
    model.online=simBWF.isSystemOnline()
    model.startTime_real=sim.getSystemTimeInMs(-1)
end

function sysCall_sensing()
    local data=model.readInfo()
    
    if sim.boolAnd32(data.bitCoded,1)>0 then
        if not model.packMlDlg.dlg_wasClosed then
            model.packMlDlg.createDlg()
        end
    else
        model.packMlDlg.closeDlg()
        model.packMlDlg.dlg_wasClosed=nil
    end

    if sim.boolAnd32(data.bitCoded,2)>0 then
        if not model.packMlButtons.dlg_wasClosed then
            model.packMlButtons.createDlg()
        end
    else
        model.packMlButtons.closeDlg()
        model.packMlButtons.dlg_wasClosed=nil
    end
    
    
    model.simplifiedTimeDisplay=sim.boolAnd32(data.bitCoded,8)>0
    if sim.boolAnd32(data.bitCoded,4)>0 then
        if model.simplifiedTimeDisplay~=model.previousSimplifiedTimeDisplay then
            model.timeDlg.closeDlg()
            model.timeDlg.timeUi_previousDlgPos=nil
            model.previousSimplifiedTimeDisplay=model.simplifiedTimeDisplay
        end
        if not model.timeDlg.timeDlg_wasClosed then
            model.timeDlg.createDlg()
        end
    else
        model.timeDlg.closeDlg()
        model.timeDlg.timeDlg_wasClosed=nil
    end
    
    
    if model.timeDlg.ui then
        local t={sim.getSimulationTime(),sim.getSystemTimeInMs(model.startTime_real)/1000}
        local cnt=2
        if model.simplifiedTimeDisplay or model.online then
            cnt=1
        end
        if model.online then
            t={sim.getSystemTimeInMs(model.startTime_real)/1000}
        end
        for i=1,cnt,1 do
            local v=t[i]
            local hour=math.floor(v/3600)
            v=v-3600*hour
            local minute=math.floor(v/60)
            v=v-60*minute
            local second=math.floor(v)
            v=v-second
            local hs=math.floor(v*100)
            local str=simBWF.format("%02d",hour)..':'..simBWF.format("%02d",minute)..':'..simBWF.format("%02d",second)..'.'..simBWF.format("%02d",hs)
            simUI.setLabelText(model.timeDlg.ui,i,str,true)
        end
    end
    
    local msg,data=simBWF.query('packml_getState',{})
    local state='none'
    local buttons={}
    if msg=='ok' then
        state=data.state
        buttons=data.buttons
    else
        -- We fake a state:
        local allStates={'aborting','aborted','clearing','stopping','stopped','suspending','suspended','un-suspending','resetting','complete','completing','execute','starting','idle','holding','hold','un-holding'}
        if not model.__fakeState__ then
            model.__fakeState__=1
        end
        model.__fakeState__=model.__fakeState__+0.1
        if model.__fakeState__>#allStates then
            model.__fakeState__=1
        end
        state=allStates[math.floor(model.__fakeState__)]
        
        -- We fake buttons:
        local allButtonCombinations={{'start','stop','abort'},{'hold','stop','abort'},{'un-hold','stop','abort'},
                                    {'reset','stop','abort'},{'reset','abort'},{'clear'}}
        if not model.__fakePackMlButtons__ then
            model.__fakePackMlButtons__=1
        end
        model.__fakePackMlButtons__=model.__fakePackMlButtons__+0.1
        if model.__fakePackMlButtons__>#allButtonCombinations then
            model.__fakePackMlButtons__=1
        end
        buttons=allButtonCombinations[math.floor(model.__fakePackMlButtons__)]
    end
    
    if model.packMlDlg.ui then
        model.packMlDlg.updateState(state)  
    end
    
    if model.packMlButtons.ui then
        model.packMlButtons.updateState(state,buttons)  
    end
end


function sysCall_cleanup()
    model.timeDlg.closeDlg()
    model.packMlDlg.closeDlg()
    model.packMlButtons.closeDlg()
end
