local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.SHIPPINGBOX_PART)
        model.partType='shippingBox'
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                -- Common:
                require("/bwf/scripts/part_common/common")
                require("/bwf/scripts/part_common/customization_main")
                require("/bwf/scripts/part_common/customization_data")
                require("/bwf/scripts/part_common/customization_dlg")
                -- Box specific:
                require("/bwf/scripts/genericShippingBox/common")
                require("/bwf/scripts/genericShippingBox/customization_main")
                require("/bwf/scripts/genericShippingBox/customization_dlg")
            end
        else
            -- Child script
            
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
