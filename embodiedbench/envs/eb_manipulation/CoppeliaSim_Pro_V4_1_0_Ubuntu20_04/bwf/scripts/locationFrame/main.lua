local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.LOCATIONFRAME)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/locationFrame/common")
                require("/bwf/scripts/locationFrame/customization_main")
                require("/bwf/scripts/locationFrame/customization_data")
                require("/bwf/scripts/locationFrame/customization_ext")
                require("/bwf/scripts/locationFrame/customization_constr")
                require("/bwf/scripts/locationFrame/customization_dlg")
            end
        else
            -- Child script
            if model.modelVersion==1 then
                require("/bwf/scripts/locationFrame/common")
                require("/bwf/scripts/locationFrame/child_main")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
