-- Functions:
-------------------------------------------------------
function model.completeDataConveyorSpecific(data)
    if not data.conveyorSpecific then
        data.conveyorSpecific={}
    end
    if not data.conveyorSpecific.borderHeight then
        data.conveyorSpecific.borderHeight=0.2
    end
    if not data.conveyorSpecific.wallThickness then
        data.conveyorSpecific.wallThickness=0.005
    end
    if not data.conveyorSpecific.bitCoded then
        data.conveyorSpecific.bitCoded=1+2+4+8 -- 1=leftOpen, 2=rightOpen, 4=frontOpen, 8=backOpen, 16=roundEnds, 32=textured
    end
end

-- Additional handles:
-------------------------------------------------------
model.specHandles={}

local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
sim.setInt32Parameter(sim.intparam_error_report_mode,0) -- do not report errors
model.specHandles.rotJoints={}
model.specHandles.rotJoints[1]=sim.getObjectHandle('genericConveyorTypeA_jointB')
model.specHandles.rotJoints[2]=sim.getObjectHandle('genericConveyorTypeA_jointC')
sim.setInt32Parameter(sim.intparam_error_report_mode,err) -- report errors again

model.specHandles.middleParts={}
model.specHandles.middleParts[1]=sim.getObjectHandle('genericConveyorTypeA_sides')
model.specHandles.middleParts[2]=sim.getObjectHandle('genericConveyorTypeA_textureA')
model.specHandles.middleParts[3]=sim.getObjectHandle('genericConveyorTypeA_forwarderA')

model.specHandles.endParts={}
local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
sim.setInt32Parameter(sim.intparam_error_report_mode,0) -- do not report errors
model.specHandles.endParts[1]=sim.getObjectHandle('genericConveyorTypeA_textureB')
model.specHandles.endParts[2]=sim.getObjectHandle('genericConveyorTypeA_textureC')
sim.setInt32Parameter(sim.intparam_error_report_mode,err) -- report errors again
model.specHandles.endParts[3]=sim.getObjectHandle('genericConveyorTypeA_B')
model.specHandles.endParts[4]=sim.getObjectHandle('genericConveyorTypeA_C')
model.specHandles.endParts[5]=sim.getObjectHandle('genericConveyorTypeA_forwarderB')
model.specHandles.endParts[6]=sim.getObjectHandle('genericConveyorTypeA_forwarderC')

model.specHandles.sides={}
model.specHandles.sides[1]=sim.getObjectHandle('genericConveyorTypeA_leftSide')
model.specHandles.sides[2]=sim.getObjectHandle('genericConveyorTypeA_rightSide')
model.specHandles.sides[3]=sim.getObjectHandle('genericConveyorTypeA_frontSide')
model.specHandles.sides[4]=sim.getObjectHandle('genericConveyorTypeA_backSide')

model.specHandles.textureHolder=sim.getObjectHandle('genericConveyorTypeA_textureHolder')

