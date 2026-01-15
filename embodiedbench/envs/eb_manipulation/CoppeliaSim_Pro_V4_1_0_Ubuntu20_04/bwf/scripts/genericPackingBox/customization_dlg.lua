function model.dlg.refresh_specific()
    local config=model.readInfo()
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
    simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f , %.0f , %.0f",config.partSpecific.width*1000,config.partSpecific.length*1000,config.partSpecific.height*1000),true)
    simUI.setEditValue(model.dlg.ui,4,simBWF.format("%.0f",config.partSpecific['thickness']/0.001),true)

    simUI.setCheckboxValue(model.dlg.ui,10,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.partSpecific['bitCoded'],1)~=0),true)
    simUI.setEditValue(model.dlg.ui,11,simBWF.format("%.0f",config.partSpecific['closePartALength']*100),true)
    simUI.setEditValue(model.dlg.ui,12,simBWF.format("%.0f",config.partSpecific['closePartAWidth']*100),true)
    simUI.setCheckboxValue(model.dlg.ui,13,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.partSpecific['bitCoded'],2)~=0),true)
    simUI.setEditValue(model.dlg.ui,14,simBWF.format("%.0f",config.partSpecific['closePartBLength']*100),true)
    simUI.setEditValue(model.dlg.ui,15,simBWF.format("%.0f",config.partSpecific['closePartBWidth']*100),true)

    simUI.setCheckboxValue(model.dlg.ui,888,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.partSpecific['bitCoded'],4)~=0),true)

    simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.2f",config.partSpecific['mass']),true)
    simUI.setEditValue(model.dlg.ui,21,simBWF.format("%.2f",config.partSpecific['inertiaFactor']),true)
    simUI.setEditValue(model.dlg.ui,22,simBWF.format("%.2f",config.partSpecific['lidTorque']),true)
    simUI.setEditValue(model.dlg.ui,23,simBWF.format("%.2f",config.partSpecific['lidSpring']),true)
    simUI.setEditValue(model.dlg.ui,24,simBWF.format("%.2f",config.partSpecific['lidDamping']),true)

    local red,green,blue,spec=model.getColor()
    simUI.setSliderValue(model.dlg.ui,30,red*100,true)
    simUI.setSliderValue(model.dlg.ui,31,green*100,true)
    simUI.setSliderValue(model.dlg.ui,32,blue*100,true)
    simUI.setSliderValue(model.dlg.ui,33,spec*100,true)

    simUI.setEnabled(model.dlg.ui,11,sim.boolAnd32(config.partSpecific['bitCoded'],1)~=0,true)
    simUI.setEnabled(model.dlg.ui,12,sim.boolAnd32(config.partSpecific['bitCoded'],1)~=0,true)
    simUI.setEnabled(model.dlg.ui,14,sim.boolAnd32(config.partSpecific['bitCoded'],2)~=0,true)
    simUI.setEnabled(model.dlg.ui,15,sim.boolAnd32(config.partSpecific['bitCoded'],2)~=0,true)
end

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t={c.partSpecific.width,c.partSpecific.length,c.partSpecific.height}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]<0.05 then t[i]=0.05 end
        if t[i]>2 then t[i]=2 end
        i=i+1
    end
    c.partSpecific.width=t[1]
    c.partSpecific.length=t[2]
    c.partSpecific.height=t[3]
    model.writeInfo(c)
    model.updateModel()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.thicknessChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>0.02 then v=0.02 end
        if v~=c.partSpecific['thickness'] then
            simBWF.markUndoPoint()
            c.partSpecific['thickness']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.lidA_callback(ui,id,newVal)
    local c=model.readInfo()
    c.partSpecific['bitCoded']=sim.boolOr32(c.partSpecific['bitCoded'],1)
    if newVal==0 then
        c.partSpecific['bitCoded']=c.partSpecific['bitCoded']-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateModel()
    model.dlg.refresh()
end

function model.dlg.lidB_callback(ui,id,newVal)
    local c=model.readInfo()
    c.partSpecific['bitCoded']=sim.boolOr32(c.partSpecific['bitCoded'],2)
    if newVal==0 then
        c.partSpecific['bitCoded']=c.partSpecific['bitCoded']-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateModel()
    model.dlg.refresh()
end

function model.dlg.lidALengthChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c.partSpecific['closePartALength'] then
            simBWF.markUndoPoint()
            c.partSpecific['closePartALength']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.lidAWidthChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c.partSpecific['closePartAWidth'] then
            simBWF.markUndoPoint()
            c.partSpecific['closePartAWidth']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.lidBLengthChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c.partSpecific['closePartBLength'] then
            simBWF.markUndoPoint()
            c.partSpecific['closePartBLength']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.lidBWidthChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v/100
        if v<0.1 then v=0.1 end
        if v>1 then v=1 end
        if v~=c.partSpecific['closePartBWidth'] then
            simBWF.markUndoPoint()
            c.partSpecific['closePartBWidth']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.massChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c.partSpecific['mass'] then
            simBWF.markUndoPoint()
            c.partSpecific['mass']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.inertiaFactorChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.1 then v=0.1 end
        if v>10 then v=10 end
        if v~=c.partSpecific['inertiaFactor'] then
            simBWF.markUndoPoint()
            c.partSpecific['inertiaFactor']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.torqueChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c.partSpecific['lidTorque'] then
            simBWF.markUndoPoint()
            c.partSpecific['lidTorque']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.springConstantChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>100 then v=100 end
        if v~=c.partSpecific['lidSpring'] then
            simBWF.markUndoPoint()
            c.partSpecific['lidSpring']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.dampingChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>10 then v=10 end
        if v~=c.partSpecific['lidDamping'] then
            simBWF.markUndoPoint()
            c.partSpecific['lidDamping']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
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

function model.dlg.texture_callback(ui,id,newVal)
    local c=model.readInfo()
    c.partSpecific['bitCoded']=sim.boolOr32(c.partSpecific['bitCoded'],4)
    if newVal==0 then
        c.partSpecific['bitCoded']=c.partSpecific['bitCoded']-4
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateModel()
    model.dlg.refresh()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    simUI.setEditValue(ui,1365,simBWF.getObjectAltName(model.handle),true)
end

function model.dlg.getSpecificTabContent()
    local xml = [[
        <tab title="General">
            <group layout="form" flat="false">
            <label text="Name"/>
            <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

            <label text="Size (X, Y, Z, in mm)"/>
            <edit on-editing-finished="model.dlg.sizeChange_callback" id="1"/>

            <label text="Thickness (mm)"/>
            <edit on-editing-finished="model.dlg.thicknessChange_callback" id="4"/>
            </group>

        </tab>
        <tab title="Closing lids">
        <group layout="form" flat="false">
            <label text="Lid A" style="* {font-weight: bold;}"/>  <label text=""/>
            
            <checkbox text="Length (%)" on-change="model.dlg.lidA_callback" id="10" />
            <edit on-editing-finished="model.dlg.lidALengthChange_callback" id="11"/>

            <label text="Width (%)"/>
            <edit on-editing-finished="model.dlg.lidAWidthChange_callback" id="12"/>
            </group>

        <group layout="form" flat="false">
            <label text="Lid B" style="* {font-weight: bold;}"/>  <label text=""/>
            
            <checkbox text="Length (%)" on-change="model.dlg.lidB_callback" id="13" />
            <edit on-editing-finished="model.dlg.lidBLengthChange_callback" id="14"/>

            <label text="Width (%)"/>
            <edit on-editing-finished="model.dlg.lidBWidthChange_callback" id="15"/>
            </group>
        </tab>
        <tab title="Colors/Texture">
        <group layout="form" flat="false">
                <label text="Textured"/>
                <checkbox text="" on-change="model.dlg.texture_callback" id="888" />
                
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange" id="30"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange" id="31"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange" id="32"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.specularChange" id="33"/>
        </group>
        </tab>
        <tab title="More">
        <group layout="form" flat="false">
            <label text="Mass (Kg)"/>
            <edit on-editing-finished="model.dlg.massChange_callback" id="20"/>

            <label text="Inertia adjustment factor"/>
            <edit on-editing-finished="model.dlg.inertiaFactorChange_callback" id="21"/>

            <label text="Lid max. torque"/>
            <edit on-editing-finished="model.dlg.torqueChange_callback" id="22"/>

            <label text="Lid spring constant"/>
            <edit on-editing-finished="model.dlg.springConstantChange_callback" id="23"/>

            <label text="Lid damping"/>
            <edit on-editing-finished="model.dlg.dampingChange_callback" id="24"/>

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
        </group>
        </tab>
    ]]
    return xml
end