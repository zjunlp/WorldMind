model.dlg={}

function model.dlg.refreshDlg()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.0f , %.0f , %.0f",config.size[1]*1000,config.size[2]*1000,config.size[3]*1000),true)
        simUI.setCheckboxValue(model.dlg.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,32,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)
        
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,enabled,true)
        simUI.setEnabled(model.dlg.ui,20,enabled,true)
        simUI.setEnabled(model.dlg.ui,26,(sim.boolAnd32(config['bitCoded'],16)~=0)and enabled,true)
        simUI.setEnabled(model.dlg.ui,3,enabled,true)
        simUI.setEnabled(model.dlg.ui,4,enabled,true)
        simUI.setEnabled(model.dlg.ui,32,enabled,true)
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.hidden_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refreshDlg()
end

function model.dlg.console_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refreshDlg()
end

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.size
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0.001 then t[i]=0.001 end
            if t[i]>2 then t[i]=2 end
        end
        if i==2 then
            if t[i]<0.001 then t[i]=0.001 end
            if t[i]>1 then t[i]=1 end
        end
        if i==3 then
            if t[i]<0.05 then t[i]=0.05 end
            if t[i]>0.5 then t[i]=0.5 end
        end
        i=i+1
    end
    c.size=t
    model.writeInfo(c)
    model.setSizes()
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
end

function loadTheString()
    local f=loadstring("local counter=0\n".."return "..theString)
    return f
end

function model.dlg.startDistributionDlg(title,fieldName,tempComment)
    local prop=model.readInfo()
    local s="800 400"
    local p="200 200"
    if model.dlg.distributionDlgSize then
        s=model.dlg.distributionDlgSize[1]..' '..model.dlg.distributionDlgSize[2]
    end
    if model.dlg.distributionDlgPos then
        p=model.dlg.distributionDlgPos[1]..' '..model.dlg.distributionDlgPos[2]
    end
    local xml = [[ <editor title="]]..title..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            



    local initialText=prop[fieldName]
    if tempComment then
        initialText=initialText.."--[[tmpRem\n\n"..tempComment.."\n\n--]]"
    end
    local modifiedText
    modifiedText,model.dlg.distributionDlgSize,model.dlg.distributionDlgPos=sim.openTextEditor(initialText,xml)
    theString='{'..modifiedText..'}' -- variable needs to be global here
    local bla,f=pcall(loadTheString)
    local success=false
    if f then
        local res,err=xpcall(f,function(err) return debug.traceback(err) end)
        if res then
            modifiedText=simBWF.removeTmpRem(modifiedText)
            prop[fieldName]=modifiedText
            model.writeInfo(prop)
            simBWF.markUndoPoint()
            success=true
        end
    end
    if not success then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Input Error',"The distribution string is ill-formated.")
    end
end

function model.dlg.partColorDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<rgbColor>},{partialProportion2,<rgbColor>} where <rgbColor> is a color value of type {red,green,blue}, where each color component can vary between 0 and 1

b) Example:
   {2.1,{1,0,0}},{1.5,{0,0,1}} --> out of (2.1+1.5) items, 2.1 will be red, and 1.5 will be blue (statistically)

c) You may use the variable 'counter' as in following example:
   {counter%2,{1,0,0}},{(counter+1)%2,{0,0,1}} --> colors alternate between red and blue
   
d) You may use '<DEFAULT>', to refer to the default value, as in following example:
   {1,"<DEFAULT>"},{1,{0,0,1}} --> out of 2 items, one will keep its color, the other turn blue (statistically)]]
    model.dlg.startDistributionDlg('Part Color Distribution','partColorDistribution',tmpTxt)
end

function model.dlg.partColorEnable_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refreshDlg()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refreshDlg()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
                <label text="Size (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.sizeChange_callback" id="20"/>

                <checkbox text="Change part color, distribution" on-change="model.dlg.partColorEnable_callback" id="32" />
                <button text="Edit"  on-click="model.dlg.partColorDistribution_callback" id="26" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="model.dlg.hidden_callback" id="3" />

                <label text="Show debug console"/>
                <checkbox text="" on-change="model.dlg.console_callback" id="4" />
            </group>
        ]]
        
        
--                <checkbox text="Change part destination, distribution" on-change="model.dlg.partDestinationNameEnable_callback" id="31" />
--                <button text="Edit"  on-click="model.dlg.partDestinationNameDistribution_callback" id="24" />
        
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil,false,false,false,'')

        model.dlg.refreshDlg()
    end
end

function model.dlg.showDlg()
    if not model.dlg.ui then
        model.dlg.createDlg()
    end
end

function model.dlg.removeDlg()
    if model.dlg.ui then
        local x,y=simUI.getPosition(model.dlg.ui)
        model.dlg.previousDlgPos={x,y}
        simUI.destroy(model.dlg.ui)
        model.dlg.ui=nil
    end
end


function model.dlg.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model.handle then
        model.dlg.showDlg()
    else
        model.dlg.removeDlg()
    end
end

function model.dlg.init()
    model.dlg.mainTabIndex=0
    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
end
