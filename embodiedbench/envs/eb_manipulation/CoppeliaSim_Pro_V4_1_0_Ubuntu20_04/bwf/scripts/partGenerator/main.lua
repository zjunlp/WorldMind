local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        if simBWF.getApplication()~='br' then
            model={}
            simBWF.appendCommonModelData(model,simBWF.modelTags.PART_GENERATOR)
            if isCustomizationScript then
                -- Customization script
                if model.modelVersion==1 then
                    require("/bwf/scripts/partGenerator/common")
                    require("/bwf/scripts/partGenerator/customization_main")
                    require("/bwf/scripts/partGenerator/customization_dlg")
                end
            else
                -- Child script

            end
            sysCall_init() -- one of above's 'require' redefined that function
        end
    end
end
