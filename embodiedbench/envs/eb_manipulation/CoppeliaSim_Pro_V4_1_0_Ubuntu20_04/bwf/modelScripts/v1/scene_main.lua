function sysCall_init()
    local msg="The main script in this scene is an old version, and\nis not supported anymore.\n\nMake sure to use an updated scene instead."
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

