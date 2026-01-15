function model.dlg.refresh_specific()
    local config=model.readInfo()
    simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.getSpecificTabContent()
    local xml = [[
        <tab title="General">
        <group layout="form" flat="false">
            <label text="Name"/>
            <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
            </group>
        </tab>
    ]]
    return xml
end
