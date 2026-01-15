-- This example illustrates how to execute complex commands from
-- a remote API client. You can also use a similar construct for
-- commands that are not directly supported by the remote API.
--
-- Load the demo scene 'remoteApiCommandServerExample.ttt' in CoppeliaSim, then 
-- start the simulation and run this program.
--
-- IMPORTANT: for each successful call to simxStart, there
-- should be a corresponding call to simxFinish at the end!

print('Program started')
require 'remoteApiLua'
simxFinish(-1) -- just in case, close all opened connections
local clientID=simxStart('127.0.0.1',19999,true,true,2000,5)
if clientID~=-1 then
    print('Connected to remote API server')

    -- 1. First send a command to display a specific message in a dialog box:
    local result,intsOut,floatsOut,stringsOut,bufferOut=simxCallScriptFunction(clientID,'remoteApiCommandServer',sim_scripttype_childscript,'displayText_function',{},{},{'Hello world'},'',simx_opmode_blocking)
    if result==simx_return_ok then
        print('Returned message: '..stringsOut[1]) -- display the reply from CoppeliaSim (in this case, just a string)
    else
        print('Remote function call failed')
    end

    -- 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
    local result,intsOut,floatsOut,stringsOut,bufferOut=simxCallScriptFunction(clientID,'remoteApiCommandServer',sim_scripttype_childscript,'createDummy_function',{},{0.1,0.2,0.3},{'MyDummyName'},'',simx_opmode_blocking)
    if result==simx_return_ok then
        print('Dummy handle: '..intsOut[1]) -- // display the reply from CoppeliaSim (in this case, the handle of the created dummy)
    else
        print('Remote function call failed')
    end

    -- 3. Now send a code string to execute some random functions:
    local code=[[
local octreeHandle=simCreateOctree(0.5,0,1)
simInsertVoxelsIntoOctree(octreeHandle,0,{0.1,0.1,0.1},{255,0,255})
return 'done'
]]
    local result,intsOut,floatsOut,stringsOut,bufferOut=simxCallScriptFunction(clientID,'remoteApiCommandServer',sim_scripttype_childscript,'executeCode_function',{},{},{code},'',simx_opmode_blocking)
    if result==simx_return_ok then
        print('Code execution returned: '..stringsOut[1])
    else
        print('Remote function call failed')
    end

    -- Now close the connection to CoppeliaSim:   
    simxFinish(clientID)
else
    print('Failed connecting to remote API server')
end
print('Program ended')
