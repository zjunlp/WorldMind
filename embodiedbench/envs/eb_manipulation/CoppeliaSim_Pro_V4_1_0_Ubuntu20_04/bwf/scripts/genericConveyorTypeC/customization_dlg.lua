function model.dlg.padHeightChange(ui,id,newVal)
    local conf=model.readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.01 then w=0.01 end
        if w>0.2 then w=0.2 end
        if w~=conf.conveyorSpecific.padHeight then
            simBWF.markUndoPoint()
            conf.conveyorSpecific.padHeight=w
            model.writeInfo(conf)
            model.updateConveyor()
        end
    end
    model.dlg.refresh()
end

function model.dlg.padThicknessChange(ui,id,newVal)
    local conf=model.readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.001 then w=0.001 end
        if w>0.2 then w=0.2 end
        if w~=conf.conveyorSpecific.padThickness then
            simBWF.markUndoPoint()
            conf.conveyorSpecific.padThickness=w
            model.writeInfo(conf)
            model.updateConveyor()
        end
    end
    model.dlg.refresh()
end

function model.dlg.padSpacingChange(ui,id,newVal)
    local conf=model.readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.02 then w=0.02 end
        if w>5 then w=5 end
        if w~=conf.conveyorSpecific.padSpacing then
            simBWF.markUndoPoint()
            conf.conveyorSpecific.padSpacing=w
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
    model.updateConveyor()
    model.dlg.refresh()
end

function model.dlg.rightSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,2)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.updateConveyor()
    model.dlg.refresh()
end

function model.dlg.frontSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,4)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-4
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.updateConveyor()
    model.dlg.refresh()
end

function model.dlg.backSideOpenClicked(ui,id,newVal)
    local conf=model.readInfo()
    conf.conveyorSpecific.bitCoded=sim.boolOr32(conf.conveyorSpecific.bitCoded,8)
    if newVal==0 then
        conf.conveyorSpecific.bitCoded=conf.conveyorSpecific.bitCoded-8
    end
    simBWF.markUndoPoint()
    model.writeInfo(conf)
    model.updateConveyor()
    model.dlg.refresh()
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
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        
        simUI.setEnabled(model.dlg.ui,2000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2003,simStopped,true)

        simUI.setEnabled(model.dlg.ui,3000,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3004,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3005,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3006,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3007,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3008,simStopped,true)
        
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

        simUI.setEditValue(model.dlg.ui,3000,simBWF.format("%.0f",config.conveyorSpecific.padHeight/0.001),true)
        simUI.setEditValue(model.dlg.ui,3002,simBWF.format("%.0f",config.conveyorSpecific.padSpacing/0.001),true)
        simUI.setEditValue(model.dlg.ui,3001,simBWF.format("%.0f",config.conveyorSpecific.padThickness/0.001),true)
        simUI.setEditValue(model.dlg.ui,3008,simBWF.format("%.0f",config.conveyorSpecific.wallThickness/0.001),true)
        simUI.setCheckboxValue(model.dlg.ui,3004,(sim.boolAnd32(config.conveyorSpecific.bitCoded,1)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3005,(sim.boolAnd32(config.conveyorSpecific.bitCoded,2)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3006,(sim.boolAnd32(config.conveyorSpecific.bitCoded,4)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(model.dlg.ui,3007,(sim.boolAnd32(config.conveyorSpecific.bitCoded,8)~=0) and 2 or 0,true)
        simUI.setLabelText(model.dlg.ui,3003,simBWF.format("%.0f",model.getActualPadSpacing()/0.001),true)
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
    <tab title="More" layout="form">
                <label text="Pad height (mm)"/>
                <edit on-editing-finished="model.dlg.padHeightChange" id="3000"/>

                <label text="Pad thickness (mm)"/>
                <edit on-editing-finished="model.dlg.padThicknessChange" id="3001"/>

                <label text="Max. pad spacing (mm)"/>
                <edit on-editing-finished="model.dlg.padSpacingChange" id="3002"/>

                <label text="Actual pad spacing (mm)"/>
                <label text="xxx" id="3003"/>
                
                <label text="Left side open"/>
                <checkbox text="" on-change="model.dlg.leftSideOpenClicked" id="3004"/>

                <label text="Right side open"/>
                <checkbox text="" on-change="model.dlg.rightSideOpenClicked" id="3005"/>

                <label text="Front side open"/>
                <checkbox text="" on-change="model.dlg.frontSideOpenClicked" id="3006"/>

                <label text="Back side open"/>
                <checkbox text="" on-change="model.dlg.backSideOpenClicked" id="3007"/>

                <label text="Wall thickness (mm)"/>
                <edit on-editing-finished="model.dlg.wallThicknessChange" id="3008"/>
    </tab>
        ]]
    return xml
end
