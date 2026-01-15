function sysCall_trigger(inData)
    -- Called when the attached vision sensor, proximity sensor or force/torque sensor is triggered
    --
    -- inData when attached object is a vision sensor:
    -- inData.handle: handle of the vision sensor
    -- inData.packedPackets: an array of data packets, packed (use sim.unpackFloatTable)
    --
    -- inData when attached object is a proximity sensor:
    -- inData.handle: handle of the proximity sensor
    -- inData.detectedObjectHandle: handle of detected object
    -- inData.detectedPoint: detected point, relative to sensor frame
    -- inData.normalVector: normal vector at detected point, relative to sensor frame
    --
    -- inData when attached object is a force/torque sensor:
    -- inData.handle: handle of the force sensor
    -- inData.force: current force
    -- inData.torque: current torque
    -- inData.filteredForce: current filtered force
    -- inData.filteredTorque: current filtered torque
    
    -- if the attached object is a vision sensor or proximity sensor, return following:
    outData={}
    outData.trigger=true
    return outData
end
