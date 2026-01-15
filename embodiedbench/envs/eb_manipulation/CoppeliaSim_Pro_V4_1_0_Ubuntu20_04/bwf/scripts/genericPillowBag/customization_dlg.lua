function model.dlg.refresh_specific()
    local config=model.readInfo()
    local partInfo=model.readPartInfo()
    local partConf=partInfo['labelData']
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
    simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f , %.0f , %.0f",config.partSpecific.width*1000,config.partSpecific.length*1000,config.partSpecific.height*1000),true)

    local size=partConf['smallLabelSize']
    simUI.setEditValue(model.dlg.ui,50,simBWF.format("%.0f , %.0f",size[1]*1000,size[2]*1000),true)
    local size=partConf['largeLabelSize']
    simUI.setEditValue(model.dlg.ui,51,simBWF.format("%.0f , %.0f",size[1]*1000,size[2]*1000),true)

    simUI.setCheckboxValue(model.dlg.ui,40,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],8)~=0),true)

    simUI.setCheckboxValue(model.dlg.ui,41,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],64)~=0),true)

    simUI.setCheckboxValue(model.dlg.ui,888,simBWF.getCheckboxValFromBool(sim.boolAnd32(config.partSpecific['bitCoded'],4)~=0),true)
    simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.2f",config.partSpecific['mass']),true)
    local red,green,blue=model.getColor()
    simUI.setSliderValue(model.dlg.ui,30,red*100,true)
    simUI.setSliderValue(model.dlg.ui,31,green*100,true)
    simUI.setSliderValue(model.dlg.ui,32,blue*100,true)

    simUI.setEnabled(model.dlg.ui,41,sim.boolAnd32(partConf['bitCoded'],8)~=0,true)
    simUI.setEnabled(model.dlg.ui,42,sim.boolAnd32(partConf['bitCoded'],8)~=0,true)
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
        if t[i]>1 then t[i]=1 end
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

function model.dlg.label1_callback(ui,id,newVal)
    local inf=model.readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    inf['labelData']=c
    model.writePartInfo(inf)
    model.updateModel()
    model.dlg.refresh()
end

function model.dlg.largeLabel1_callback(ui,id,newVal)
    local inf=model.readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    inf['labelData']=c
    model.writePartInfo(inf)
    model.updateModel()
    model.dlg.refresh()
end

function model.dlg.smallLabelSizeChange_callback(ui,id,newVal)
    local inf=model.readPartInfo()
    local c=inf['labelData']
    local i=1
    local t={0.01,0.01}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=10 end
        t[i]=t[i]*0.001
        if t[i]>0.3 then t[i]=0.3 end
        if t[i]<0.01 then t[i]=0.01 end
        i=i+1
    end
    local ov=c['smallLabelSize']
    if ov[1]~=t[1] or ov[2]~=t[2] then
        c['smallLabelSize']={t[1],t[2]}
        inf['labelData']=c
        model.writePartInfo(inf)
        model.updateModel()
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.largeLabelSizeChange_callback(ui,id,newVal)
    local inf=model.readPartInfo()
    local c=inf['labelData']
    local i=1
    local t={0.01,0.01}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=10 end
        t[i]=t[i]*0.001
        if t[i]>0.3 then t[i]=0.3 end
        if t[i]<0.01 then t[i]=0.01 end
        i=i+1
    end
    local ov=c['largeLabelSize']
    if ov[1]~=t[1] or ov[2]~=t[2] then
        c['largeLabelSize']={t[1],t[2]}
        inf['labelData']=c
        model.writePartInfo(inf)
        model.updateModel()
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.placeDlgStart(title,fieldIndex,tempComment)
    local inf=model.readPartInfo()
    local prop=inf['labelData']['placementCode']
    local s="800 400"
    local p="200 200"
    if model.dlg.distributionDlgSize then
        s=model.dlg.distributionDlgSize[1]..' '..model.dlg.distributionDlgSize[2]
    end
    if model.dlg.distributionDlgPos then
        p=model.dlg.distributionDlgPos[1]..' '..model.dlg.distributionDlgPos[2]
    end
    local xml = [[ <editor title="]]..title..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            



    local initialText=prop[fieldIndex]
    if tempComment then
        initialText=initialText.."--[[tmpRem\n\n"..tempComment.."\n\n--]]"
    end
    local modifiedText
    modifiedText,model.dlg.distributionDlgSize,model.dlg.distributionDlgPos=sim.openTextEditor(initialText,xml)

    local toExecute="local boxSizeX,boxSizeY,boxSizeZ,labelSizeX,labelSizeY,labelRadius=1,1,1,1,1,1\n return {"..modifiedText.."}"
    local res,theTable=sim.executeLuaCode(toExecute)
    local success=false
    if theTable then
        if type(theTable)=='table' and #theTable>=2 then
            if type(theTable[1])=='table' and type(theTable[2])=='table' and #theTable[1]>=3 and #theTable[2]>=3 then
                success=true
                for i=1,3,1 do
                    if type(theTable[1][i])~='number' or type(theTable[2][i])~='number' then
                        success=false
                        break
                    end
                end
                if success then
                    modifiedText=simBWF.removeTmpRem(modifiedText)
                    prop[fieldIndex]=modifiedText
                    inf['labelData']['placementCode']=prop
                    model.writePartInfo(inf)
                    simBWF.markUndoPoint()
                    model.updateModel()
                end
            end
        end
    end
    if not success then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Input Error',"The placement code is ill-formated.")
    end
end

function model.dlg.getPlaceDlgHelpText()
    local txt=[[
a) Usage:
   <labelPosition>,<labelOrientation> where <labelPosition> is {posX,posY,posZ} (in meters) and <labelOrientation> is {rotAroundX,rotAroundY,rotAroundZ} (in radians) 

b) You may use the variables 'boxSizeX', 'boxSizeY', 'boxSizeZ', 'labelSizeX', 'labelSizeY' and 'labelRadius'

c) Example:
   {0,0,boxSizeZ*0.5+0.001},{0,0,0}

d) Placement is relative to the box's reference frame]]
    return txt
end

function model.dlg.placementLabel1_callback(ui,id,newVal)
    local tmpTxt=model.dlg.getPlaceDlgHelpText()
    model.dlg.placeDlgStart('Label 1 placement',1,tmpTxt)
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
        <tab title="Labels">
            <group layout="form" flat="false">
                <label text="Small label size (X, Y, in mm)"/>
                <edit on-editing-finished="model.dlg.smallLabelSizeChange_callback" id="50"/>
                <label text="Large label size (X, Y, in mm)"/>
                <edit on-editing-finished="model.dlg.largeLabelSizeChange_callback" id="51"/>
            </group>

            <group layout="grid" flat="false">
                <checkbox text="Label 1" on-change="model.dlg.label1_callback" id="40" />
                <checkbox text="Large label" on-change="model.dlg.largeLabel1_callback" id="41" />
                <button text="Label placement code"  on-click="model.dlg.placementLabel1_callback" id="42" />
            </group>
        </tab>
        <tab title="Colors/Texture">
        <group layout="form" flat="false">
                <checkbox text="Textured" on-change="model.dlg.texture_callback" id="888" />
                <label text=""/>
                
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