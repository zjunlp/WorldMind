function sysCall_init()
    local modelName=sim.getObjectName(sim.getObjectAssociatedWithScript(sim.handle_self))
    local msg="Model '"..modelName.."' is an old version, and\nis not supported anymore.\n\nMake sure to use an updated model instead."
    local xml ='<label text="'..msg..'" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px; color: red;}"/>'
    ui=simBWF.createCustomUi(xml,'Error','center',true,'onClose',false,false,false)
end

function onClose()
    if ui then
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_cleanup()
    onClose()
end

