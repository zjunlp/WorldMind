local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.INPUTBOX)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/inputBox/common")
                require("/bwf/scripts/inputBox/customization_main")
                require("/bwf/scripts/inputBox/customization_data")
                require("/bwf/scripts/inputBox/customization_ext")
                require("/bwf/scripts/inputBox/customization_dlg")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
