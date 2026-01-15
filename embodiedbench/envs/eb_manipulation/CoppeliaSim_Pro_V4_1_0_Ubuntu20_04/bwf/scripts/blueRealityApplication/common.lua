-- Functions:
-------------------------------------------------------
function model.readInfo()
    -- Read all the data stored in the model
    
    local data=sim.readCustomDataBlock(model.handle,model.tagName)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    
    -- All the data stored in the model. Set-up default values, and remove unused values
    if not data['version'] then
        data['version']=1
    end
    if not data['subtype'] then
        data['subtype']='br-app'
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=show packMl state, 2=show packML buttons,4=show time, 8=simplified time, 16=show OEE, 32= show warnings at sim start, 64=do not Reset cameras
    end
    if data['packMlCode'].aborting == "-- 'Aborting' code:" then
        local code={}
        local proceedScript = [[
function init()
-- Only runs once, initialize variables here, add events, or send one time commands

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

pml.proceed()
end  ]]
    local noProceedScript = [[
function init()
-- Only runs once, initialize variables here, add events, or send one time commands

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

--pml.proceed()
end  ]]
	    local resettingScript = [[
function init()
-- Only runs once, initialize variables here or send commands once
	pml.initializeAllRobots() -- Home all robots
	pml.clearAllTrackingObjects() -- Clear all tracking objects

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

proceed()
end  ]]
	    local executeScript = [[
function init()
-- Only runs once, initialize variables here or send commands once
	pml.startAllRobotProcess() -- Start PnP process for all robots
	pml.activateAllLineControls() -- Activate window belt line control in all configured windows

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

--pml.proceed()
end  ]]
	local stoppingScript = [[
function init()
-- Only runs once, initialize variables here or send commands once
	pml.stopAllRobotProcess() -- Stop PnP process for all robots
	pml.deactivateAllLineControls() -- Deactivate window belt line control

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

pml.proceed()
end  ]]
	local abortingScript = [[
function init()
-- Only runs once, initialize variables here or send commands once
	pml.stopAllRobotProcess() -- Stop PnP process for all robots
    pml.deactivateAllLineControls() -- Deactivate window belt line control
	pml.clearAllVariables() -- Clear all User Variables

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

pml.proceed()
end  ]]
	local clearingScript = [[
function init()
-- Only runs once, initialize variables here or send commands once
	pml.clearAllRobots() -- Take all robots out of E-STOP

end
function actions()
-- Commands and logic to control the behavior of the system. Runs in a loop

end
function conditions()
-- Check signals and conditions to proceed to the next state, stop or abort. Runs in a loop

pml.proceed()
end  ]]
        code.aborting="-- 'Aborting' code: \n" .. abortingScript
        code.aborted="-- 'Aborted' code: \n" .. noProceedScript
        code.clearing="-- 'Clearing' code:\n" .. clearingScript
        code.stopping="-- 'Stopping' code:\n" .. stoppingScript
        code.stopped="-- 'Stopped' code:\n" .. noProceedScript

        code.suspending="-- 'Suspending' code:\n" .. proceedScript
        code.suspended="-- 'Suspended' code:\n" .. noProceedScript
        code.unsuspending="-- 'Un-Suspending' code:\n" .. proceedScript
        code.resetting="-- 'Resetting' code:\n" .. resettingScript

        code.complete="-- 'Complete' code:\n" .. noProceedScript
        code.completing="-- 'Completing' code:\n" .. proceedScript
        code.execute="-- 'Execute' code:\n" .. executeScript
        code.starting="-- 'Starting' code:\n" .. proceedScript
        code.idle="-- 'Idle' code:\n" .. noProceedScript

        code.holding="-- 'Holding' code:\n" .. proceedScript
        code.hold="-- 'Hold' code:\n" .. noProceedScript
        code.unholding="-- 'Un-Holding' code:\n" .. proceedScript
        data['packMlCode']=code
    end
    if not data['floorSizes'] then
        data['floorSizes']={10,10}
    end
    data['pallets']=nil
    if not data.masterIp then
        data.masterIp="127.0.0.1"
    end
    if not data.deactivationTime then
        data.deactivationTime=60
    end
    data.packedPackMlImage=nil
    data.packedPackMlStateImage=nil

    return data
end

function model.writeInfo(data)
    -- Write all the data stored in the model. Before writing, make sure to always first read with readInfo()
    
    if data then
        sim.writeCustomDataBlock(model.handle,model.tagName,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model.handle,model.tagName,'')
    end
end


-- referenced object slots (do not modify):
-------------------------------------------------------



-- Handles and similar:
-------------------------------------------------------
model.handles={}

model.brCalls={}

model.brCalls.NEWJOB=297
model.brCalls.DELETEJOB=298
model.brCalls.RENAMEJOB=299
model.brCalls.SWITCHJOB=300
