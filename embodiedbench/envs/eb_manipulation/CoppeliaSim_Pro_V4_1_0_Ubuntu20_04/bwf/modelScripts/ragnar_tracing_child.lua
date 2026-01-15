getRobotHandle=function(objectHandle)
    while true do
        local p=sim.getModelProperty(objectHandle)
        if sim.boolAnd32(p,sim.modelproperty_not_model)==0 then
            return objectHandle
        end
        objectHandle=sim.getObjectParent(objectHandle)
    end
end

getColorFromIntensity=function(intensity)
    local col={0.16,0.16,0.16,0.16,0.16,1,1,0.16,0.16,1,1,0.16}
    if intensity>1 then intensity=1 end
    if intensity<0 then intensity=0 end
    intensity=math.exp(4*(intensity-1))
    local d=math.floor(intensity*3)
    if (d>2) then d=2 end
    local r=(intensity-d/3)*3
    local coll={}
    coll[1]=col[3*d+1]*(1-r)+col[3*(d+1)+1]*r
    coll[2]=col[3*d+2]*(1-r)+col[3*(d+1)+2]*r
    coll[3]=col[3*d+3]*(1-r)+col[3*(d+1)+3]*r
    return coll
end

function sysCall_init()
    dummy=sim.getObjectAssociatedWithScript(sim.handle_self)
    model=getRobotHandle(dummy)
    local ragnarSettings=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
    ragnarSettings=sim.unpackTable(ragnarSettings)
    showTrajectory=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(ragnarSettings['bitCoded'],1)>0)
    if showTrajectory then
        local lineBufferSize=100
        cont=sim.addDrawingObject(sim.drawing_lines+sim.drawing_itemcolors+sim.drawing_emissioncolor+sim.drawing_cyclic,3,0,-1,lineBufferSize)
        prevPos=sim.getObjectPosition(dummy,-1)
        prevTime=sim.getSimulationTime()
    end
end


function sysCall_sensing()
    if showTrajectory then
        local ragnarSettings=sim.readCustomDataBlock(model,simBWF.modelTags.RAGNAR)
        ragnarSettings=sim.unpackTable(ragnarSettings)
        local maxSpeed=ragnarSettings.maxVel

        local p=sim.getObjectPosition(dummy,-1)
        local t=sim.getSimulationTime()
        local dx={p[1]-prevPos[1],p[2]-prevPos[2],p[3]-prevPos[3]}
        local dt=t-prevTime
        local l=math.sqrt(dx[1]*dx[1]+dx[2]*dx[2]+dx[3]*dx[3])
        local speed=0
        if dt>0 then
            speed=l/dt
        end
        local c=getColorFromIntensity(speed/maxSpeed)
        local data={prevPos[1],prevPos[2],prevPos[3],p[1],p[2],p[3],c[1],c[2],c[3]}
        sim.addDrawingObjectItem(cont,data)
        prevPos={p[1],p[2],p[3]}
        prevTime=t
    end
end
