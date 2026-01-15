model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)

        simUI.setEditValue(model.dlg.ui,20,simBWF.format("%.0f , %.0f , %.0f",config.width*1000,config.length*1000,config.height*1000),true)
        simUI.setCheckboxValue(model.dlg.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)==0),true)
        simUI.setCheckboxValue(model.dlg.ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],128)~=0),true)
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
    model.dlg.refresh()
end

function model.dlg.showStatisticsClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.enabled_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal~=0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end


function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t={c.width,c.length,c.height}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 or i==2 then
            if t[i]<0.2 then t[i]=0.2 end
            if t[i]>5 then t[i]=5 end
        end
        if i==3 then
            if t[i]<0.01 then t[i]=0.01 end
            if t[i]>1 then t[i]=1 end
        end
        i=i+1
    end
    c.width=t[1]
    c.length=t[2]
    c.height=t[3]
    model.writeInfo(c)
    model.setSizes()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
                <label text="Size (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.sizeChange_callback" id="20"/>
                
                <label text="Enabled"/>
                <checkbox text="" on-change="model.dlg.enabled_callback" id="4" />

                <label text="Hidden during simulation"/>
                <checkbox text="" on-change="model.dlg.hidden_callback" id="3" />

                <label text="Show statistics"/>
                 <checkbox text="" checked="false" on-change="model.dlg.showStatisticsClick_callback" id="6"/>
             </group>
        ]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil,false,false,false,'')

        model.dlg.refresh()
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
