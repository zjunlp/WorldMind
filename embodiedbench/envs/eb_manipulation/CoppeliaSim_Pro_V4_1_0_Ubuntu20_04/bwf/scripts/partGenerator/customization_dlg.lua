model.dlg={}
    
function model.dlg.refresh()
    if model.dlg.ui then
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        
        if model.selectedObj<0 then
            simUI.setLabelText(model.dlg.ui,1,"Select exactly one raw, untagged object you wish to\n mark as 'part', then click 'Generate part'")
        else
            simUI.setLabelText(model.dlg.ui,1,"object ready to be tagged as 'part'")
        end
        
        simUI.setEnabled(model.dlg.ui,2,model.selectedObj>=0,true)
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.generate_callback()
    if model.selectedObj>=0 then
        -- Mark it as model if needed:
        local p=sim.getModelProperty(model.selectedObj)
        if sim.boolAnd32(p,sim.modelproperty_not_model)~=0 then
            sim.setModelProperty(model.selectedObj,p-sim.modelproperty_not_model)
        end
        -- Attach the generic part info struct:
        local data={}
        data.version=1
        sim.writeCustomDataBlock(model.selectedObj,simBWF.modelTags.GENERIC_PART,sim.packTable(data))
        -- Set/attach the correct customization script:
        local s=sim.getCustomizationScriptAssociatedWithObject(model.selectedObj)
        if s<0 then
            s=sim.addScript(sim.scripttype_customizationscript)
            sim.associateScriptWithObject(s,model.selectedObj)
        end
        sim.setScriptText(s,"require('/bwf/scripts/genericPart/main')")
    end
    
--    simBWF.callCustomizationScriptFunction('model.ext.insertPart',repo,model)
--    finalizeModel=true
--    sim.removeScript(sim.handle_self)
end

function model.dlg.onClose_callback()
    model.dlg.removeDlg()
    sim.removeObject(model.handle)
end

function model.dlg.createDlg()
    if (not model.dlg.ui) then
        local xml =[[
                <label text="text" id="1"/>
                <label text="" style="* {margin-left: 150px;}"/>
                <button text="Generate part" on-click="model.dlg.generate_callback" id="2"/>
                <label text="" style="* {margin-left: 320px;}"/>
        ]]

        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,1,1),model.dlg.previousDlgPos,true,"model.dlg.onClose_callback"--[[,modal,resizable,activate,additionalUiAttribute--]])

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
    model.dlg.showDlg()
    local s=sim.getObjectSelection()
    local h=-1
    if s and #s==1 then
        local t=sim.readCustomDataBlockTags(s[1])
        local ng=false
        if t then
            for i=1,#t,1 do
                for key,value in pairs(simBWF.modelTags) do
                    if t[i]==value then
                        ng=true
                        break
                    end
                end
            end
        end
        if not ng then
            h=s[1]
        end
    end
    if h~=model.selectedObj then
        model.selectedObj=h
        model.dlg.refresh()
    end
end

function model.dlg.init()
    model.dlg.mainTabIndex=0
    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    if sim.isHandleValid(model.handle)>0 then
        simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
    end
end
