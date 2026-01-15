local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        model.dlg={}
        model.handle=sim.getObjectAssociatedWithScript(sim.handle_self)
        if isCustomizationScript then
            -- Customization script
            require("/bwf/scripts/ragnarRef/customization_main")
        else
            -- Child script
            
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
