function os.capture(cmd,raw)
    local f=assert(io.popen(cmd,'r'))
    local s=assert(f:read('*a'))
    f:close()
    if raw then return s end
    s=string.gsub(s,'^%s+','')
    s=string.gsub(s,'%s+$','')
    s=string.gsub(s,'[\n\r]+',' ')
    return s
end

function os.platform()
    local win=package.config:sub(1,1)=='\\'
    if win then return 'windows' end
    local uname=os.capture('uname')
    if uname=='Linux' then return 'linux' end
    if uname=='Darwin' then return 'macos' end
    return 'unknown'
end

simTest={}

function simTest.log(v,msg)
    sim.addLog(v,'simTest: '..msg)
end

function simTest.logInfo(msg)
    simTest.log(sim.verbosity_errors,msg)
end

function simTest.logWarning(msg)
    simTest.log(sim.verbosity_errors,msg)
end

function simTest.logError(msg)
    simTest.log(sim.verbosity_errors,msg)
end

function simTest.fatalError(msg)
    simTest.logError(msg)
    sim.quitSimulator(true)
end

function simTest.listLuaFiles(dirName)
    local files={}
    local plat=os.platform()
    local g
    if plat=='windows' then
        g=io.popen('dir "'..dirName..'" /b /ad')
    else
        g=io.popen('ls -1 "'..dirName..'"')
    end
    for file in g:lines() do
        if file:match('%.lua$') then
            table.insert(files,file)
        end
    end
    return files
end

function simTest.getFileContent(fileName)
    local f=io.open(fileName,'rb')
    if not f then
        simTest.fatalError('file "'..luaFile..'" does not exist')
    end
    local content=f:read('*a')
    f:close()
    return content
end

function simTest.getMeta(luaCode)
    local meta={}
    for k, v in string.gmatch(luaCode,'\n--@(%w+) ([^\n]*)\n') do
        meta[k]=v
    end
    return meta
end

function simTest.createDummyWithScript(objName,luaCode,scriptType)
    local dummyHandle=sim.createDummy(0.01)
    sim.setObjectName(dummyHandle,objName)
    local scriptHandle=sim.addScript(scriptType)
    sim.associateScriptWithObject(scriptHandle,dummyHandle)
    sim.setScriptText(scriptHandle,luaCode)
    return dummyHandle
end

function simTest.writeOutputFile(fileName,content)
    local output_dir=sim.getStringNamedParam('output_dir')
    f=io.open(output_dir..'/'..fileName,'w+')
    if not f then
        simTest.fatalError('cannot open file "'..fileName..'" for writing')
    end
    f:write(content)
    f:close()
end

function simTest.setTestResult(r)
    simTest.writeOutputFile('result.txt',r)
end

function simTest.setTestExitCode(e)
    simTest.writeOutputFile('exitcode.txt',e)
end

function simTest.success(msg)
    if msg==nil then
        simTest.logInfo('test succeeded')
    else
        simTest.logInfo('test succeeded: '..msg)
    end
    simTest.setTestResult('success')
    simTest.setTestExitCode(0)
    sim.quitSimulator(true)
end

function simTest.failure(msg)
    if msg==nil then
        simTest.logInfo('test failed')
    else
        simTest.logInfo('test failed: '..msg)
    end
    simTest.setTestResult('failure')
    simTest.setTestExitCode(1)
    sim.quitSimulator(true)
end

return simTest
