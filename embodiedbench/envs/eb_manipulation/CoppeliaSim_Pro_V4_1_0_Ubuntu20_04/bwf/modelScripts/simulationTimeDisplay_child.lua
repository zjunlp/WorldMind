function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    local data=sim.readCustomDataBlock(model,'XYZ_SIMULATIONTIME_INFO')
    data=sim.unpackTable(data)
    simplified=sim.boolAnd32(data['bitCoded'],1)==1
    if not sim.getBoolParameter(sim_boolparam_headless) then
        if simplified then
            local xml =[[
                    <label text="Time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="1" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
            ]]
            ui=simBWF.createCustomUi(xml,'Time','bottomLeft',true,nil,false,false,false,'layout="form"')
        else
            local xml =[[
                    <label text="Simulation time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="1" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label text="Real-time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="2" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
            ]]
            ui=simBWF.createCustomUi(xml,'Simulation Time','bottomLeft',true,nil,false,false,false,'layout="form"')
        end
    end
    startTime=sim.getSystemTimeInMs(-1)
end

function sysCall_sensing()
    if ui then
        local t={sim.getSimulationTime(),sim.getSystemTimeInMs(startTime)/1000}
        local cnt=2
        if simplified then
            cnt=1
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
            simUI.setLabelText(ui,i,str,true)
        end
    end
end


function sysCall_cleanup()
    if ui then
        simUI.destroy(ui)
    end
end