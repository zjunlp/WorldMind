function model.dlg.refresh_specific()
    local config=model.readInfo()
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
    simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f , %.0f , %.0f",config.partSpecific.width*1000,config.partSpecific.length*1000,config.partSpecific.height*1000),true)
    simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.2f",config.partSpecific.mass),true)
    local red,green,blue=model.getColor()
    simUI.setSliderValue(model.dlg.ui,30,red*100,true)
    simUI.setSliderValue(model.dlg.ui,31,green*100,true)
    simUI.setSliderValue(model.dlg.ui,32,blue*100,true)
end

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t={c.partSpecific.width,c.partSpecific.length,c.partSpecific.height}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]<0.005 then t[i]=0.005 end
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

function model.dlg.massChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c.partSpecific.mass then
            simBWF.markUndoPoint()
            c.partSpecific.mass=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=model.getColor()
    model.setColor(newVal/100,g,b)
end

function model.dlg.greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=model.getColor()
    model.setColor(r,newVal/100,b)
end

function model.dlg.blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=model.getColor()
    model.setColor(r,g,newVal/100)
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

            <label text="Mass (Kg)"/>
            <edit on-editing-finished="model.dlg.massChange_callback" id="20"/>


            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
            </group>
        </tab>
        <tab title="Colors">
        <group layout="form" flat="false">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange" id="30"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange" id="31"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange" id="32"/>
        </group>
        </tab>
        ]]
    return xml
end
