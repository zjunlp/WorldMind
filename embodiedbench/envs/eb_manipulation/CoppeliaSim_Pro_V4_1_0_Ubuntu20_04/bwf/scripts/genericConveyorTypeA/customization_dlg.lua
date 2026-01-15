function model.dlg.borderHeightChange(ui,id,newVal)
    local conf=model.readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.005 then w=0.005 end
        if w>0.2 then w=0.2 end
        if w~=conf.conveyorSpecific.borderHeight then
            simBWF.markUndoPoint()
            conf.conveyorSpecific.borderHeight=w
            model.writeInfo(conf)
            model.updateConveyor()
        end
    end
    model.dlg.refresh()
end

function model.dlg.wallThicknessChange(ui,id,newVal)
    local conf=model.readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.001 then w=0.001 end
        if w>0.02 then w=0.02 end
        if w~=conf.conveyorSpecific.wallThickness then
            simBWF.markUndoPoint()
            conf.conveyorSpecific.wallThickness=w
            model.writeInfo(conf)
            model.updateConveyor()
        end
    end
    model.dlg.refresh()
end

function model.dlg.leftSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,1)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.dlg.refresh()
    model.updateConveyor()
end

function model.dlg.rightSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,2)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.dlg.refresh()
    model.updateConveyor()
end

function model.dlg.frontSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,4)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-4
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.dlg.refresh()
    model.updateConveyor()
end

function model.dlg.backSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,8)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-8
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.dlg.refresh()
    model.updateConveyor()
end

function model.dlg.roundedEndsClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,16)
    if newVal~=0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-16
    end
    model.writeInfo(conf)
    simBWF.markUndoPoint()
    model.updateConveyor()
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.texturedClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,32)
    if newVal~=0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-32
    end
    model.writeInfo(conf)
    simBWF.markUndoPoint()
    model.updateConveyor()
end

function model.dlg.redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor()
    model.setColor(newVal/100,g,b,s)
end

function model.dlg.greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor()
    model.setColor(r,newVal/100,b,s)
end

function model.dlg.blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor()
    model.setColor(r,g,newVal/100,s)
end

function model.dlg.specularChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor()
    model.setColor(r,g,b,newVal/100)
end

function model.dlg.updateEnabledDisabledItems_specific()
    if model.dlg.ui then
        local c=model.readInfo()
        local re=sim.boolAnd32(c.conveyorSpecific.bitCoded,16)==0
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        
        simUI.setEnabled(model.dlg.ui,2000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2003,simStopped,true)

        simUI.setEnabled(model.dlg.ui,3000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3003,simStopped and re,true)
        simUI.setEnabled(model.dlg.ui,3004,simStopped and re,true)
        simUI.setEnabled(model.dlg.ui,3005,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3006,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3007,simStopped,true)
    end
end

function model.dlg.refresh_specific()
    if model.dlg.ui then
        local red,green,blue,spec=model.getColor()
        local config=model.readInfo()

        simUI.setSliderValue(model.dlg.ui,2000,red*100,true)
        simUI.setSliderValue(model.dlg.ui,2001,green*100,true)
        simUI.setSliderValue(model.dlg.ui,2002,blue*100,true)
        simUI.setSliderValue(model.dlg.ui,2003,spec*100,true)
        
        simUI.setEditValue(model.dlg.ui,3000,simBWF.format("%.0f",config.conveyorSpecific.borderHeight/0.001),true)
        simUI.setEditValue(model.dlg.ui,3005,simBWF.format("%.0f",config.conveyorSpecific.wallThickness/0.001),true)

        simUI.setCheckboxValue(model.dlg.ui,3001,(sim.boolAnd32(config.conveyorSpecific.bitCoded,1)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3002,(sim.boolAnd32(config.conveyorSpecific.bitCoded,2)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3003,(sim.boolAnd32(config.conveyorSpecific.bitCoded,4)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3004,(sim.boolAnd32(config.conveyorSpecific.bitCoded,8)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3006,(sim.boolAnd32(config.conveyorSpecific.bitCoded,16)==0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3007,(sim.boolAnd32(config.conveyorSpecific.bitCoded,32)==0) and 2 or 0,true)
        
    end
end

function model.dlg.getSpecificTabContent()
    local xml = [[
    <tab title="Color">
            <group layout="form" flat="false">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange" id="2000"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange" id="2001"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange" id="2002"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.specularChange" id="2003"/>
            </group>
    </tab>
    <tab title="More">
            <group layout="form" flat="false">
                <label text="Border height (mm)"/>
                <edit on-editing-finished="model.dlg.borderHeightChange" id="3000"/>

                <label text="Wall thickness (mm)"/>
                <edit on-editing-finished="model.dlg.wallThicknessChange" id="3005"/>
                
                <label text="Textured"/>
                <checkbox text="" on-change="model.dlg.texturedClicked" id="3007"/>

                <label text="Left side open"/>
                <checkbox text="" on-change="model.dlg.leftSideOpenClicked" id="3001"/>

                <label text="Right side open"/>
                <checkbox text="" on-change="model.dlg.rightSideOpenClicked" id="3002"/>

                <label text="Rounded ends"/>
                <checkbox text="" on-change="model.dlg.roundedEndsClicked" id="3006"/>

                <label text="Front side open"/>
                <checkbox text="" on-change="model.dlg.frontSideOpenClicked" id="3003"/>

                <label text="Back side open"/>
                <checkbox text="" on-change="model.dlg.backSideOpenClicked" id="3004"/>
            </group>
    </tab>
        ]]
    return xml
end
