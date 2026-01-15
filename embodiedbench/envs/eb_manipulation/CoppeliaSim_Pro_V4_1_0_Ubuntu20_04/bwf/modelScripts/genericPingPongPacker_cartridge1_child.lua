putCartridgeDown=function()
    sim.wait(dwellTimeUp)
    sim.rmlMoveToJointPositions({j},-1,{0},{0},{maxVel},{maxAccel},{9999},{0},{0})
    sim.wait(dwellTimeDown)
    sim.rmlMoveToJointPositions({j},-1,{0},{0},{maxVel},{maxAccel},{9999},{-45*math.pi/180},{0})
end

enableStopper=function(enable)
    if enable then
        sim.setObjectInt32Parameter(stopper,sim.objintparam_visibility_layer,1) -- make it visible
        sim.setObjectSpecialProperty(stopper,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable) -- make it collidable, measurable, detectable, etc.
        sim.setObjectInt32Parameter(stopper,sim.shapeintparam_respondable,1) -- make it respondable
        sim.resetDynamicObject(stopper)
    else
        sim.setObjectInt32Parameter(stopper,sim.objintparam_visibility_layer,0)
        sim.setObjectSpecialProperty(stopper,0)
        sim.setObjectInt32Parameter(stopper,sim.shapeintparam_respondable,0)
        sim.resetDynamicObject(stopper)
    end
end

waitForSensor=function(signal)
    while true do
        local r=sim.handleProximitySensor(sens)
        if signal then
            if r>0 then break end
        else
            if r<=0 then break end
        end
        sim.switchThread()
    end
end

waitForCartridgeFull=function()
    while true do
        local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        if data['putCartridgeDown'][1] then
            break
        end
        sim.switchThread()
    end
end

setCartridgeEmpty=function()
    sim.setThreadAutomaticSwitch(false)
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
    data=sim.unpackTable(data)
    data['putCartridgeDown'][1]=false
    sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,sim.packTable(data))
    sim.setThreadAutomaticSwitch(true)
end

model=sim.getObjectHandle('genericPingPongPacker')
local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
data=sim.unpackTable(data)
maxVel=data['cartridgeVelocity']
maxAccel=data['cartridgeAcceleration']
dwellTimeDown=data['cartridgeDwellTimeDown']
dwellTimeUp=data['cartridgeDwellTimeUp']
j=sim.getObjectHandle('genericPingPongPacker_cartridge1_upDownJoint')
sens=sim.getObjectHandle('genericPingPongPacker_cartridge1_sensor')
stopper=sim.getObjectHandle('genericPingPongPacker_cartridge1_stopper')

while sim.getSimulationState()~=sim.simulation_advancing_abouttostop do
    waitForSensor(true)
    enableStopper(true)
    sim.wait(1)
    waitForCartridgeFull()
    putCartridgeDown()
    setCartridgeEmpty()
    enableStopper(false)
    waitForSensor(false)
    waitForSensor(true)
    waitForSensor(false)
end


