local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.PARTTAGGER)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/genericPartTagger/common")
                require("/bwf/scripts/genericPartTagger/customization_main")
                require("/bwf/scripts/genericPartTagger/customization_data")
                require("/bwf/scripts/genericPartTagger/customization_dlg")
                require("/bwf/scripts/genericPartTagger/customization_ext")
            end
        else
            -- Child script
            if model.modelVersion==1 then
                require("/bwf/scripts/genericPartTagger/common")
                require("/bwf/scripts/genericPartTagger/child_main")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
