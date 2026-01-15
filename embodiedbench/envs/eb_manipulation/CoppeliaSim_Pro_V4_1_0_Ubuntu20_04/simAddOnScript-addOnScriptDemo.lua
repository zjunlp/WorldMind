-- this add-on simply adds an alternative UI that allows to start/stop/pause a simulation. This is not very useful,
-- but illustrates how easily CoppeliaSim can be customized using add-ons
-- return sim.syscb_cleanup if you want to stop the add-on script from here!

function sysCall_init()
    sim.addStatusbarMessage("Initialization of the add-on script")
    createUi()
    noEditMode=sim.getInt32Parameter(sim.intparam_edit_mode_type)==0
end

function sysCall_cleanup() 
    sim.addStatusbarMessage("Clean-up of the add-on script")
    removeUi()
end 

function sysCall_addOnScriptSuspend()
    sim.addStatusbarMessage("Suspending add-on script")
    removeUi()
end

function sysCall_addOnScriptResume()
    sim.addStatusbarMessage("Restarting the add-on script")
    createUi()
end

function sysCall_nonSimulation()
    return run()
end

function sysCall_beforeMainScript()
    return run()
end

function run()
    if exitRequest then
        return sim.syscb_cleanup -- the clean-up section will be called and the add-on stopped
    end
    local _noEditMode=sim.getInt32Parameter(sim.intparam_edit_mode_type)==0
    if _noEditMode~=noEditMode then
        if _noEditMode==false then
            removeUi()
        else
            createUi()
        end
        noEditMode=_noEditMode
    end
end

function sysCall_beforeInstanceSwitch()
    sim.addStatusbarMessage("Before switching to another instance (add-on script)")
    removeUi()
end

function sysCall_afterInstanceSwitch()
    sim.addStatusbarMessage("After switching to another instance (add-on script)")
--    updateUi()
    createUi()
end

function sysCall_suspend()
    sim.addStatusbarMessage("Suspending simulation (add-on script)")
    updateUi(true,false,true)
end

function sysCall_resume()
    sim.addStatusbarMessage("Resuming simulation (add-on script)")
    updateUi(false,true,true)
end

function sysCall_beforeSimulation()
    sim.addStatusbarMessage("Simulation starting (add-on script)")
    updateUi(false,true,true)
end

function sysCall_afterSimulation()
    sim.addStatusbarMessage("Simulation ended (add-on script)")
    updateUi(true,false,false)
end

function createUi()
    if not ui then
        local placement='placement="relative" position="-50,50"'
        if previousUiPos then
            placement='placement="absolute" position="'..previousUiPos[1]..','..previousUiPos[2]..'"'
        end
        local xml=[[
            <ui title="Add-on UI" closeable="true" on-close="addOnClose" modal="false" resizable="false" layout="vbox" activate="false" ]]
        xml=xml..placement..[[>
                <button text="Start/resume simulation" on-click="startSim" style="* {min-width: 150px; min-height: 30px;}" id="1"/>
                <button text="Suspend simulation" on-click="suspendSim" style="* {min-width: 150px; min-height: 30px;}" id="2"/>
                <button text="Stop simulation" on-click="stopSim" style="* {min-width: 150px; min-height: 30px;}" id="3"/>
            </ui>
        ]]
        ui=simUI.create(xml)
    end
    updateUi()
end

function updateUi(enableRun,enablePause,enableStop)
    local enabledFlags={enableRun,enablePause,enableStop}
    if enableRun==nil then
        local st=sim.getSimulationState()
        enabledFlags[1]=false
        enabledFlags[2]=false
        enabledFlags[3]=false
        if sim.getInt32Parameter(sim.intparam_edit_mode_type)==0 then
            if st==sim.simulation_stopped then
                enabledFlags[1]=true
            elseif st==sim.simulation_paused then
                enabledFlags[1]=true
                enabledFlags[3]=true
            else
                enabledFlags[2]=true
                enabledFlags[3]=true
            end
        end
    end
    for i=1,3,1 do
        simUI.setEnabled(ui,i,enabledFlags[i])
    end
end

function removeUi()
    if ui then
        previousUiPos={0,0}
        previousUiPos[1],previousUiPos[2]=simUI.getPosition(ui)
        simUI.destroy(ui)
        ui=nil
    end
end

function addOnClose()
    removeUi()
    exitRequest=true
end

function startSim()
    sim.startSimulation()
end

function suspendSim()
    sim.pauseSimulation()
end

function stopSim()
    sim.stopSimulation()
end
