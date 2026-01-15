local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.THERMOFORMER,1)
        model.conveyorType='T'
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/genericThermoformer/common")
                require("/bwf/scripts/genericThermoformer/customization_main")
                require("/bwf/scripts/genericThermoformer/customization_data")
                require("/bwf/scripts/genericThermoformer/customization_dlg")
                require("/bwf/scripts/genericThermoformer/customization_ext")
            end
        else
            -- Child script
            if model.modelVersion==1 then
                require("/bwf/scripts/genericThermoformer/common")
                require("/bwf/scripts/genericThermoformer/child_main")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
