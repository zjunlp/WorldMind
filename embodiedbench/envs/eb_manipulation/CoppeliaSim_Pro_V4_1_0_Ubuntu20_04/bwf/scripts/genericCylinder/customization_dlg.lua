function model.dlg.refresh_specific()
    local config=model.readInfo()
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
    simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f",config.partSpecific['diameter']/0.001),true)
    simUI.setEditValue(model.dlg.ui,3,simBWF.format("%.0f",config.partSpecific['height']/0.001),true)
    simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.2f",config.partSpecific['mass']),true)
    
    simUI.setSliderValue(model.dlg.ui,4,config.partSpecific['count'],true)
    simUI.setEditValue(model.dlg.ui,5,simBWF.format("%.2f",config.partSpecific['offset']),true)
    
    local red,green,blue=model.getColor()
    simUI.setSliderValue(model.dlg.ui,30,red*100,true)
    simUI.setSliderValue(model.dlg.ui,31,green*100,true)
    simUI.setSliderValue(model.dlg.ui,32,blue*100,true)
end

function model.dlg.diameterChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>2 then v=2 end
        if v~=c.partSpecific['diameter'] then
            simBWF.markUndoPoint()
            c.partSpecific['diameter']=v
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.heightChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>2 then v=2 end
        if v~=c.partSpecific['height'] then
            simBWF.markUndoPoint()
            c.partSpecific['height']=v
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


function model.dlg.countChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>3 then v=3 end
        if v~=c.partSpecific['count'] then
            simBWF.markUndoPoint()
            c.partSpecific['count']=v
            if v>1 and c.partSpecific['offset']==0 then
                c.partSpecific['offset']=0.1
            end
            model.writeInfo(c)
            model.updateModel()
        end
    end
    model.dlg.refresh()
end

function model.dlg.offsetChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>1 then v=1 end
        if v~=c.partSpecific['offset'] then
            simBWF.markUndoPoint()
            c.partSpecific['offset']=v
            if v==0 and c.partSpecific['count']>1 then
                c.partSpecific['count']=1
            end
            model.writeInfo(c)
            model.updateModel()
        end
    end
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
    local xml =[[
        <tab title="General">
        <group layout="form" flat="false">
            <label text="Name"/>
            <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
            
            <label text="Diameter (mm)"/>
            <edit on-editing-finished="model.dlg.diameterChange_callback" id="1"/>

            <label text="Height (mm)"/>
            <edit on-editing-finished="model.dlg.heightChange_callback" id="3"/>

            <label text="Mass (Kg)"/>
            <edit on-editing-finished="model.dlg.massChange_callback" id="20"/>


            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
            </group>
        </tab>
        <tab title="Roundness">
        <group layout="form" flat="false">
            <label text="multi-cylinder count"/>
            <hslider minimum="1" maximum="3" on-change="model.dlg.countChange_callback" id="4"/>

            <label text="offset (in % of diameter)"/>
            <edit on-editing-finished="model.dlg.offsetChange_callback" id="5"/>
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