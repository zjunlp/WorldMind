function sysCall_init()
    -- add simTest's dir to Lua's package.path:
    local simTest_dir=sim.getStringNamedParam('simTest_dir')
    local dirSep=package.config:sub(1,1)
    local pathSep=package.config:sub(3,3)
    local luaWildCard=package.config:sub(5,5)
    package.path=string.format('%s%s%s.lua%s%s',simTest_dir,dirSep,luaWildCard,pathSep,package.path)
    sim.registerScriptVariable('package.path','"'..package.path..'"')
    sim.executeScriptString('package.path="'..package.path..'"@',sim.scripttype_sandboxscript)
    require 'simTest'

    simTest.logInfo('started (platform='..os.platform()..')')

    -- read .lua files from input_dir:
    local input_dir=sim.getStringNamedParam('input_dir')
    local luaFiles=simTest.listLuaFiles(input_dir)
    haveChildScripts=false
    haveCustomizationScripts=false
    local initCode={}
    for i,luaFile in ipairs(luaFiles) do
        simTest.logInfo('processing lua file "'..luaFile..'"')
        local luaCode=simTest.getFileContent(input_dir..'/'..luaFile)
        local meta=simTest.getMeta(luaCode)
        if meta.mode==nil then
            simTest.fatalError('script "'..luaFile..'" does not contain a "mode" parameter')
        end
        local threaded=meta.threaded=='true' or meta.threaded=='1'
        if meta.mode=='init' then
            initCode[luaFile]=luaCode
        elseif meta.mode=='child' then
            simTest.logInfo('added child script for '..luaFile)
            simTest.createDummyWithScript('simTest_'..luaFile:gsub('%.lua$',''),luaCode,sim.scripttype_childscript+(threaded and sim.scripttype_threaded or 0))
            haveChildScripts=true
        elseif meta.mode=='customization' then
            simTest.logInfo('added customization script for '..luaFile)
            simTest.createDummyWithScript('simTest_'..luaFile:gsub('%.lua$',''),luaCode,sim.scripttype_customizationscript)
            simTest.logInfo('haveCustomizationScripts=true')
            haveCustomizationScripts=true
        else
            simTest.fatalError('invalid value "'..meta.mode..'" for parameter "mode" in script "'..luaFile..'"')
        end
    end

    -- execute scripts with '@mode init'
    for luaFile,code in pairs(initCode) do
        simTest.logInfo('executing init code for '..luaFile)
        result,value=sim.executeScriptString(code..'@',sim.scripttype_sandboxscript)
        if result==-1 then
            simTest.fatalError('error in '..luaFile..': '..value)
        end
    end

    if haveCustomizationScripts then
        simTest.logInfo('skipping control of the simulation because there are customization scripts')
    else
        if haveChildScripts then
            simTest.logInfo('auto-starting the simulation...')
            sim.startSimulation()
        else
            simTest.logInfo('quitting because there are no child/customization scripts to run')
            sim.quitSimulator(true)
        end
    end
end

function sysCall_afterSimulation()
    if not haveCustomizationScripts then
        simTest.logInfo('simulation has stopped. quitting...')
        sim.quitSimulator(true)
    end
end
