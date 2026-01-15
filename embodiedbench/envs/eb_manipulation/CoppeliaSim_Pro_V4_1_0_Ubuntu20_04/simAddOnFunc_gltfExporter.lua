local scenePath=sim.getStringParameter(sim.stringparam_scene_path)
local sceneName=sim.getStringParameter(sim.stringparam_scene_name):match("(.+)%..+")
if sceneName==nil then sceneName='untitled' end
local fileName=sim.fileDialog(sim.filedlg_type_save,'Export to glTF...',scenePath,sceneName..'.gltf','glTF file','gltf')
if fileName==nil then return end
simGLTF.clear()
simGLTF.exportAllObjects()
simGLTF.saveASCII(fileName)
sim.addStatusbarMessage('Exported glTF content to '..fileName)
