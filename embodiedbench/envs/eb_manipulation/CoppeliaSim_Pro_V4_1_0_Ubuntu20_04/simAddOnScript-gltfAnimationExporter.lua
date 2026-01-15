function sysCall_init()
    sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,'GLTF Animation Export','GLTF animation export is active. Content of current simulation will be recorded, and will be saved when the simulation will stop.')
    simGLTF.recordAnimation(true)
end

function sysCall_addOnScriptSuspend()
    simGLTF.recordAnimation(false)
    simGLTF.clear()
    return sim.syscb_cleanup
end

function sysCall_afterSimulation()
    if simGLTF.animationFrameCount()>0 then
        local scenePath=sim.getStringParameter(sim.stringparam_scene_path)
        local sceneName=sim.getStringParameter(sim.stringparam_scene_name):match("(.+)%..+")
        if sceneName==nil then sceneName='untitled' end
            local fileName=sim.fileDialog(sim.filedlg_type_save,'Export animation to glTF...',scenePath,sceneName..'.gltf','glTF file','gltf')
        if fileName~=nil then
            simGLTF.exportAnimation()
            simGLTF.saveASCII(fileName)
            simGLTF.recordAnimation(false)
            simGLTF.clear()
        end
        return sim.syscb_cleanup
    end
end

function sysCall_beforeInstanceSwitch()
    simGLTF.recordAnimation(false)
    simGLTF.clear()
    return sim.syscb_cleanup
end
