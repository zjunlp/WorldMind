function model.dlg.refresh_specific()
    local config=model.readInfo()
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
    simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f",config.partSpecific['width']/0.001),true)
    simUI.setEditValue(model.dlg.ui,2,simBWF.format("%.0f",config.partSpecific['length']/0.001),true)
    simUI.setEditValue(model.dlg.ui,3,simBWF.format("%.0f",config.partSpecific['height']/0.001),true)
    simUI.setEditValue(model.dlg.ui,4,simBWF.format("%.0f",config.partSpecific['borderHeight']/0.001),true)
    simUI.setEditValue(model.dlg.ui,5,simBWF.format("%.0f",config.partSpecific['borderThickness']/0.001),true)
    simUI.setEditValue(model.dlg.ui,6,simBWF.format("%.2f",config.partSpecific['mass']),true)
    simUI.setCheckboxValue(model.dlg.ui,19,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.partSpecific['bitCoded'],1)~=0),true)
    local off=config.partSpecific['placeOffset']
    simUI.setEditValue(model.dlg.ui,40,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)

    simUI.setEnabled(model.dlg.ui,29,sim.boolAnd32(config.partSpecific['bitCoded'],1)==0,true)
    local pocketT=config.partSpecific['pocketType']
    simUI.setEnabled(model.dlg.ui,204,pocketT==1,true)
    simUI.setEnabled(model.dlg.ui,205,pocketT==2,true)
    simUI.setRadiobuttonValue(model.dlg.ui,103,simBWF.getRadiobuttonValFromBool(pocketT==0),true)
    simUI.setRadiobuttonValue(model.dlg.ui,104,simBWF.getRadiobuttonValFromBool(pocketT==1),true)
    simUI.setRadiobuttonValue(model.dlg.ui,105,simBWF.getRadiobuttonValFromBool(pocketT==2),true)

    local lineP=config.partSpecific['linePocket']
    simUI.setEditValue(model.dlg.ui,4000,simBWF.format("%.0f",lineP[1]/0.001),true)
    simUI.setEditValue(model.dlg.ui,4001,simBWF.format("%.0f",lineP[2]/0.001),true)
    simUI.setEditValue(model.dlg.ui,4002,tostring(lineP[3]),true)
    simUI.setEditValue(model.dlg.ui,4003,tostring(lineP[4]),true)

    local honeyP=config.partSpecific['honeyPocket']
    simUI.setEditValue(model.dlg.ui,5000,simBWF.format("%.0f",honeyP[1]/0.001),true)
    simUI.setEditValue(model.dlg.ui,5001,simBWF.format("%.0f",honeyP[2]/0.001),true)
    simUI.setEditValue(model.dlg.ui,5002,tostring(honeyP[3]),true)
    simUI.setEditValue(model.dlg.ui,5003,tostring(honeyP[4]),true)
    simUI.setCheckboxValue(model.dlg.ui,5004,simBWF.getCheckboxValFromBool(honeyP[5]),true)
    
    local red,green,blue,spec=model.getColor1()
    simUI.setSliderValue(model.dlg.ui,20,red*100,true)
    simUI.setSliderValue(model.dlg.ui,21,green*100,true)
    simUI.setSliderValue(model.dlg.ui,22,blue*100,true)
    simUI.setSliderValue(model.dlg.ui,23,spec*100,true)
    red,green,blue,spec=model.getColor2()
    simUI.setSliderValue(model.dlg.ui,30,red*100,true)
    simUI.setSliderValue(model.dlg.ui,31,green*100,true)
    simUI.setSliderValue(model.dlg.ui,32,blue*100,true)
    simUI.setSliderValue(model.dlg.ui,33,spec*100,true)
    
end

function model.dlg.lengthChange(ui,id,newVal)
    local c=model.readInfo()
    local bt=c.partSpecific['borderThickness']
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.05 then l=0.05 end
        if l<bt*2+0.01 then l=bt*2+0.01 end
        if l>2 then l=2 end
        if l~=c.partSpecific['length'] then
            simBWF.markUndoPoint()
            c.partSpecific['length']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.widthChange(ui,id,newVal)
    local c=model.readInfo()
    local bt=c.partSpecific['borderThickness']
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.05 then l=0.05 end
        if l<bt*2+0.01 then l=bt*2+0.01 end
        if l>2 then l=2 end
        if l~=c.partSpecific['width'] then
            simBWF.markUndoPoint()
            c.partSpecific['width']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.heightChange(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c.partSpecific['height'] then
            simBWF.markUndoPoint()
            c.partSpecific['height']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.borderHeightChange(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0 end
        if l>1 then l=1 end
        if l~=c.partSpecific['borderHeight'] then
            simBWF.markUndoPoint()
            c.partSpecific['borderHeight']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.borderThicknessChange(ui,id,newVal)
    local c=model.readInfo()
    local mm=math.min(c.partSpecific['width'],c.partSpecific['length'])
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l>mm/2-0.01 then l=mm/2-0.01 end
        if l~=c.partSpecific['borderThickness'] then
            simBWF.markUndoPoint()
            c.partSpecific['borderThickness']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.massChange(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        if l<0.05 then l=0.05 end
        if l>10 then l=10 end
        if l~=c.partSpecific['mass'] then
            simBWF.markUndoPoint()
            c.partSpecific['mass']=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.sameColors_callback(ui,id,newVal)
    local c=model.readInfo()
    c.partSpecific['bitCoded']=sim.boolOr32(c.partSpecific['bitCoded'],1)
    if newVal==0 then
        c.partSpecific['bitCoded']=c.partSpecific['bitCoded']-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.setColor1(model.getColor1())
    model.dlg.refresh()
end

function model.dlg.redChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor1()
    model.setColor1(newVal/100,g,b,s)
end

function model.dlg.greenChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor1()
    model.setColor1(r,newVal/100,b,s)
end

function model.dlg.blueChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor1()
    model.setColor1(r,g,newVal/100,s)
end

function model.dlg.specularChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor1()
    model.setColor1(r,g,b,newVal/100)
end

function model.dlg.redChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor2()
    model.setColor2(newVal/100,g,b,s)
end

function model.dlg.greenChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor2()
    model.setColor2(r,newVal/100,b,s)
end

function model.dlg.blueChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor2()
    model.setColor2(r,g,newVal/100,s)
end

function model.dlg.specularChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=model.getColor2()
    model.setColor2(r,g,b,newVal/100)
end

function model.dlg.patternTypeClick_callback(ui,id)
    local c=model.readInfo()
    c.partSpecific['pocketType']=id-103
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateTray()
    model.dlg.refresh()
end

function model.dlg.linePattern_heightChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c.partSpecific['linePocket'][1] then
            simBWF.markUndoPoint()
            c.partSpecific['linePocket'][1]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.linePattern_thicknessChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c.partSpecific['linePocket'][2] then
            simBWF.markUndoPoint()
            c.partSpecific['linePocket'][2]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.linePattern_rowsChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<1 then l=1 end
        if l>20 then l=20 end
        if l~=c.partSpecific['linePocket'][3] then
            simBWF.markUndoPoint()
            c.partSpecific['linePocket'][3]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.linePattern_colsChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<1 then l=1 end
        if l>20 then l=20 end
        if l~=c.partSpecific['linePocket'][4] then
            simBWF.markUndoPoint()
            c.partSpecific['linePocket'][4]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end




function model.dlg.honeyPattern_heightChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c.partSpecific['honeyPocket'][1] then
            simBWF.markUndoPoint()
            c.partSpecific['honeyPocket'][1]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.honeyPattern_thicknessChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c.partSpecific['honeyPocket'][2] then
            simBWF.markUndoPoint()
            c.partSpecific['honeyPocket'][2]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.honeyPattern_rowsChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c.partSpecific['honeyPocket'][3] then
            simBWF.markUndoPoint()
            c.partSpecific['honeyPocket'][3]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.honeyPattern_colsChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c.partSpecific['honeyPocket'][4] then
            simBWF.markUndoPoint()
            c.partSpecific['honeyPocket'][4]=l
            model.writeInfo(c)
            model.updateTray()
        end
    end
    model.dlg.refresh()
end

function model.dlg.honeyPattern_rowIsOddChange_callback(ui,id,newVal)
    local c=model.readInfo()
    c.partSpecific['honeyPocket'][5]=(newVal>0)
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateTray()
    model.dlg.refresh()
end

function model.dlg.placeOffsetChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c.partSpecific['placeOffset']={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.updateTray()
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
                
                <label text="Width (mm)"/>
                <edit on-editing-finished="model.dlg.widthChange" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="model.dlg.lengthChange" id="2"/>

                <label text="Base thickness (mm)"/>
                <edit on-editing-finished="model.dlg.heightChange" id="3"/>

                <label text="Border height (mm)"/>
                <edit on-editing-finished="model.dlg.borderHeightChange" id="4"/>

                <label text="Border thickness (mm)"/>
                <edit on-editing-finished="model.dlg.borderThicknessChange" id="5"/>

                <label text="Mass (Kg)"/>
                <edit on-editing-finished="model.dlg.massChange" id="6"/>
                
                <label text="Place offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.placeOffsetChange_callback" id="40"/>

                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
            </group>
    </tab>
    <tab title="Pockets">
        <tabs id="78">
            <tab title="None">
            <radiobutton text="Do not create any pockets" on-click="model.dlg.patternTypeClick_callback" id="103" />
            </tab>

            <tab title="Rectangle type">
            <radiobutton text="Create pockets arranged in a rectangular pattern" on-click="model.dlg.patternTypeClick_callback" id="104" />
            <group layout="form"  flat="true" id="204">
                <label text="Height (mm)"/>
                <edit on-editing-finished="model.dlg.linePattern_heightChange_callback" id="4000"/>

                <label text="Thickness (mm)"/>
                <edit on-editing-finished="model.dlg.linePattern_thicknessChange_callback" id="4001"/>

                <label text="Rows"/>
                <edit on-editing-finished="model.dlg.linePattern_rowsChange_callback" id="4002"/>

                <label text="Columns"/>
                <edit on-editing-finished="model.dlg.linePattern_colsChange_callback" id="4003"/>
            </group>
            </tab>

            <tab title="Honeycomb type">
            <radiobutton text="Create pockets arranged in a honeycomb pattern" on-click="model.dlg.patternTypeClick_callback" id="105" />
            <group layout="form"  flat="true"  id="205">
                <label text="Height (mm)"/>
                <edit on-editing-finished="model.dlg.honeyPattern_heightChange_callback" id="5000"/>

                <label text="Thickness (mm)"/>
                <edit on-editing-finished="model.dlg.honeyPattern_thicknessChange_callback" id="5001"/>

                <label text="Rows (longest)"/>
                <edit on-editing-finished="model.dlg.honeyPattern_rowsChange_callback" id="5002"/>

                <label text="Columns"/>
                <edit on-editing-finished="model.dlg.honeyPattern_colsChange_callback" id="5003"/>

                <label text="1st row is odd"/>
                <checkbox text="" on-change="model.dlg.honeyPattern_rowIsOddChange_callback" id="5004" />
            </group>
            </tab>
        </tabs>
    </tab>
    <tab title="Colors">
        <checkbox text="Base and borders have the same color" on-change="model.dlg.sameColors_callback" id="19" />
        <tabs>
        <tab title="Base" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange1" id="20"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange1" id="21"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange1" id="22"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.specularChange1" id="23"/>
        </tab>
        <tab title="Borders" layout="form" id="29">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.redChange2" id="30"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.greenChange2" id="31"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.blueChange2" id="32"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="model.dlg.specularChange2" id="33"/>
        </tab>
        </tabs>
    </tab>
    ]]
--[[
        local c=model.readInfo()
        local pattern=c.partSpecific['pocketType'] -- 0=none, 1=rectangle, 2=honeycomb
        local pat={}
        pat[0]=0
        pat[1]=1
        pat[2]=2
        simUI.setCurrentTab(model.dlg.ui,78,pat[pattern],true)
        --]]
    return xml
end