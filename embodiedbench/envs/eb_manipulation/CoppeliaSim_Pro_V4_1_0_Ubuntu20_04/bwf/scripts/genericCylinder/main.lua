local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.CYLINDER_PART)
        model.partType='cylinder'
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/part_common/common")
                require("/bwf/scripts/part_common/customization_main")
                require("/bwf/scripts/part_common/customization_data")
                require("/bwf/scripts/part_common/customization_dlg")
                -- Cylinder specific:
                require("/bwf/scripts/genericCylinder/common")
                require("/bwf/scripts/genericCylinder/customization_main")
                require("/bwf/scripts/genericCylinder/customization_dlg")
            end
        else
            -- Child script
            
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
