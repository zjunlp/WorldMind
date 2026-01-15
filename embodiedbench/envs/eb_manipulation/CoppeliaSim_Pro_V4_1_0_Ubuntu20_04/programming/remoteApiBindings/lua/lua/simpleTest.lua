-- Make sure to have the server side running in CoppeliaSim: 
-- in a child script of a CoppeliaSim scene, add following command
-- to be executed just once, at simulation start:
--
-- simRemoteApi.start(19999)
--
-- then start simulation, and run this program.
--
-- IMPORTANT: for each successful call to simxStart, there
-- should be a corresponding call to simxFinish at the end!

print('Program started')
require 'remoteApiLua'
simxFinish(-1) -- just in case, close all opened connections
local clientID=simxStart('127.0.0.1',19999,true,true,2000,5)
if clientID~=-1 then
    print('Connected to remote API server')

    -- Now try to retrieve data in a blocking fashion (i.e. a service call):
    local ret,objectHandles=simxGetObjects(clientID,sim_handle_all,simx_opmode_blocking)
    if ret==simx_return_ok then
        print('Number of objects in the scene: '..#objectHandles)
    else
        print('Remote API function call returned with error code: '..ret)
    end
        
    local clock=os.clock
    local startTime=clock()
    while clock()-startTime < 2 do end  
    
    -- Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime=clock()
    simxGetIntegerParameter(clientID,sim_intparam_mouse_x,simx_opmode_streaming) -- Initialize streaming
    while (clock()-startTime) < 5 do
        local ret,mouseX=simxGetIntegerParameter(clientID,sim_intparam_mouse_x,simx_opmode_buffer) -- Try to retrieve the streamed data
        if (ret==simx_return_ok) then -- After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
            print('Mouse position x: '..mouseX) -- Mouse position x is actualized when the cursor is over CoppeliaSim's window
        end
    end
        
    -- Now send some data to CoppeliaSim in a non-blocking fashion:
    simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',simx_opmode_oneshot)

    -- Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    simxGetPingTime(clientID)

    -- Now close the connection to CoppeliaSim:   
    simxFinish(clientID)
else
    print('Failed connecting to remote API server')
end
print('Program ended')
