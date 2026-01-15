local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.RAGNARDETECTOR)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/ragnarDetector/common")
                require("/bwf/scripts/ragnarDetector/customization_main")
                require("/bwf/scripts/ragnarDetector/customization_data")
                require("/bwf/scripts/ragnarDetector/customization_ext")
                require("/bwf/scripts/ragnarDetector/customization_constr")
                require("/bwf/scripts/ragnarDetector/customization_dlg")
            end
        else
            -- Child script
            if model.modelVersion==1 then
                require("/bwf/scripts/ragnarDetector/common")
                require("/bwf/scripts/ragnarDetector/child_main")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
