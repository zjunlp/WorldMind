function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='pricing'
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- empty for now
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_PRICING_INFO')
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    getDefaultInfoForNonExistingFields(data)
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,'XYZ_PRICING_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_PRICING_INFO','')
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        
        
        local sel=simBWF.getSelectedEditWidget(ui)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function computePriceClick_callback()


end

function createDlg()
    if not ui then
        local xml =[[
                <button text="Compute price"  style="* {min-width: 150px;min-height: 30px;}" on-click="computePriceClick_callback" id="1" />
        ]]
        ui=simBWF.createCustomUi(xml,'Pricing',previousDlgPos,false,nil,false,false,false)

        setDlgItemContent()
    end
end

function showDlg()
    if not ui then
        createDlg()
    end
end

function removeDlg()
    if ui then
        local x,y=simUI.getPosition(ui)
        previousDlgPos={x,y}
        simUI.destroy(ui)
        ui=nil
    end
end

showOrHideUiIfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg()
    else
        removeDlg()
    end
end

function sysCall_init()
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    
    local objs=sim.getObjectsWithTag('XYZ_PRICING_INFO',true)
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
    if #objs>1 then
        sim.removeObject(model)
        sim.removeObjectFromSelection(sim.handle_all)
        objs=sim.getObjectsWithTag('XYZ_PRICING_INFO',true)
        sim.addObjectToSelection(sim.handle_single,objs[1])
    else
        updatePluginRepresentation()
    end
end

function sysCall_nonSimulation()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,1)
end

function sysCall_beforeSimulation()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,0)
    removeDlg()
end

function sysCall_beforeInstanceSwitch()
    removeDlg()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    removeDlg()
    removeFromPluginRepresentation()
    if sim.isHandleValid(model)==1 then
        -- the associated object might already have been destroyed
        simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
    end
end

