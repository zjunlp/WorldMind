function model.prepareStatisticsDialog(enabled)
    if enabled then
        local xml =[[
                <label id="1" text="Part destruction count: 0" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
        ]]
        statUi=simBWF.createCustomUi(xml,sim.getObjectName(model.handle)..' Statistics','bottomLeft',true--[[,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
    end
end

function model.updateStatisticsDialog(enabled)
    if statUi then
        simUI.setLabelText(statUi,1,"Part destruction count: "..model.destructionCount,true)
    end
end

function sysCall_init()
    model.codeVersion=1
    
    local data=model.readInfo()
    model.operational=sim.boolAnd32(data['bitCoded'],2)==0
    model.destructionCount=0
    model.prepareStatisticsDialog(sim.boolAnd32(data['bitCoded'],128)>0)
    model.width=data['width']
    model.length=data['length']
    model.height=data['height']
end


function sysCall_sensing()
    if model.operational then
        local parts=simBWF.getAllInstanciatedParts()
        for i=1,#parts,1 do
            if sim.isHandleValid(parts[i])==1 then
                -- Make sure to check if that object still exists. Can happen when several parts are attached to each other
                local p=sim.getObjectPosition(parts[i],model.handle)
                if math.abs(p[1])<model.width*0.5 and math.abs(p[2])<model.length*0.5 and p[3]<model.height and p[3]>=0 then
                    sim.removeModel(parts[i])
                    model.destructionCount=model.destructionCount+1
                end
            end
        end
    end
    model.updateStatisticsDialog()
end

