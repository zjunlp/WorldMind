local sim={}
__HIDDEN__={}
__HIDDEN__.dlg={}
printToConsole=print -- will be overwritten further down

-- Various useful functions:
----------------------------------------------------------
function sim.yawPitchRollToAlphaBetaGamma(yawAngle,pitchAngle,rollAngle)
    local Rx=sim.buildMatrix({0,0,0},{rollAngle,0,0})
    local Ry=sim.buildMatrix({0,0,0},{0,pitchAngle,0})
    local Rz=sim.buildMatrix({0,0,0},{0,0,yawAngle})
    local m=sim.multiplyMatrices(Ry,Rx)
    m=sim.multiplyMatrices(Rz,m)
    local alphaBetaGamma=sim.getEulerAnglesFromMatrix(m)
    local alpha=alphaBetaGamma[1]
    local beta=alphaBetaGamma[2]
    local gamma=alphaBetaGamma[3]
    return alpha,beta,gamma
end

function sim.alphaBetaGammaToYawPitchRoll(alpha,beta,gamma)
    local m=sim.buildMatrix({0,0,0},{alpha,beta,gamma})
    local v=m[9]
    if v>1 then v=1 end
    if v<-1 then v=-1 end
    local pitchAngle=math.asin(-v)
    local yawAngle,rollAngle
    if math.abs(v)<0.999999 then
        rollAngle=math.atan2(m[10],m[11])
        yawAngle=math.atan2(m[5],m[1])
    else
        -- Gimbal lock
        rollAngle=math.atan2(-m[7],m[6])
        yawAngle=0
    end
    return yawAngle,pitchAngle,rollAngle
end

function sim.getObjectsWithTag(tagName,justModels)
    local retObjs={}
    local objs=sim.getObjectsInTree(sim.handle_scene)
    for i=1,#objs,1 do
        if (not justModels) or (sim.boolAnd32(sim.getModelProperty(objs[i]),sim.modelproperty_not_model)==0) then
            local dat=sim.readCustomDataBlockTags(objs[i])
            if dat then
                for j=1,#dat,1 do
                    if dat[j]==tagName then
                        retObjs[#retObjs+1]=objs[i]
                        break
                    end
                end
            end
        end
    end
    return retObjs
end

function sim.getObjectHandle_noErrorNoSuffixAdjustment(name)
    local suff=sim.getNameSuffix(nil)
    sim.setNameSuffix(-1)
    local retVal=sim.getObjectHandle(name..'@silentError')
    sim.setNameSuffix(suff)
    return retVal
end

function sim.executeLuaCode(theCode)
    local f=loadstring(theCode)
    if f then
        local a,b=pcall(f)
        return a,b
    else
        return false,'compilation error'
    end
end

function sim.fastIdleLoop(enable)
    local data=sim.readCustomDataBlock(sim.handle_app,'__IDLEFPSSTACKSIZE__')
    local stage=0
    local defaultIdleFps
    if data then
        data=sim.unpackInt32Table(data)
        stage=data[1]
        defaultIdleFps=data[2]
    else
        defaultIdleFps=sim.getInt32Parameter(sim.intparam_idle_fps)
    end
    if enable then
        stage=stage+1
    else
        if stage>0 then
            stage=stage-1
        end
    end
    if stage>0 then
        sim.setInt32Parameter(sim.intparam_idle_fps,0)
    else
        sim.setInt32Parameter(sim.intparam_idle_fps,defaultIdleFps)
    end
    sim.writeCustomDataBlock(sim.handle_app,'__IDLEFPSSTACKSIZE__',sim.packInt32Table({stage,defaultIdleFps}))
end

function sim.isPluginLoaded(pluginName)
    local index=0
    local moduleName=''
    while moduleName do
        moduleName=sim.getModuleName(index)
        if (moduleName==pluginName) then
            return(true)
        end
        index=index+1
    end
    return(false)
end

function isArray(t)
    local i = 0
    for _ in pairs(t) do
        i = i + 1
        if t[i] == nil then return false end
    end
    return true
end

function sim.setDebugWatchList(l)
    __HIDDEN__.debug.watchList=l
end

function sim.getUserVariables()
    local ng={}
    if __HIDDEN__.initGlobals then
        for key,val in pairs(_G) do
            if not __HIDDEN__.initGlobals[key] then
                ng[key]=val
            end
        end
    else
        ng=_G
    end
    -- hide a few additional system variables:
    ng.sim_current_script_id=nil
    ng.sim_call_type=nil
    ng.sim_code_function_to_run=nil
    ng.__notFirst__=nil
    ng.__scriptCodeToRun__=nil
    ng.__HIDDEN__=nil
    return ng
end

function sim.getMatchingPersistentDataTags(pattern)
    local result = {}
    for index, value in ipairs(sim.getPersistentDataTags()) do
        if value:match(pattern) then
            result[#result + 1] = value
        end
    end
    return result
end

function print(...)
    sim.setThreadAutomaticSwitch(false)
    sim.addLog(sim.verbosity_scriptinfos+sim.verbosity_undecorated,getAsString(...))
    sim.setThreadAutomaticSwitch(true)
end

function getAsString(...)
    local a={...}
    local t=''
    if #a==1 and type(a[1])=='string' then
--        t=string.format('"%s"', a[1])
        t=string.format('%s', a[1])
    else
        for i=1,#a,1 do
            if i~=1 then
                t=t..','
            end
            if type(a[i])=='table' then
                t=t..__HIDDEN__.tableToString(a[i],{},99)
            else
                t=t..__HIDDEN__.anyToString(a[i],{},99)
            end
        end
    end
    if #a==0 then
        t='nil'
    end
    return(t)
end

function table.pack(...)
    return {n=select("#", ...); ...}
end

function printf(fmt,...)
    local a=table.pack(...)
    for i=1,a.n do
        if type(a[i])=='table' then
            a[i]=__HIDDEN__.anyToString(a[i],{},99)
        elseif type(a[i])=='nil' then
            a[i]='nil'
        end
    end
    print(string.format(fmt,unpack(a,1,a.n)))
end


function sim.displayDialog(title,mainTxt,style,modal,initTxt,titleCols,dlgCols,prevPos,dlgHandle)
    if sim.getBoolParameter(sim_boolparam_headless) then
        return -1
    end
    assert(type(title)=='string' and type(mainTxt)=='string' and type(style)=='number' and type(modal)=='boolean',"One of the function's argument type is not correct")
    if type(initTxt)~='string' then
        initTxt=''
    end
    local retVal=-1
    local center=true
    if sim.boolAnd32(style,sim.dlgstyle_dont_center)>0 then
        center=false
        style=style-sim.dlgstyle_dont_center
    end
    assert(not modal or sim.isScriptExecutionThreaded()>0,"Can't use modal operation with non-threaded scripts")
    if modal and style==sim.dlgstyle_message then
        modal=false
    end
    local xml='<ui title="'..title..'" closeable="false" resizable="false"'
    if modal then
        xml=xml..' modal="true"'
    else
        xml=xml..' modal="false"'
    end

    if prevPos then
        xml=xml..' placement="absolute" position="'..prevPos[1]..','..prevPos[2]..'">'
    else
        if center then
            xml=xml..' placement="center">'
        else
            xml=xml..' placement="relative" position="-50,50">'
        end
    end
    mainTxt=string.gsub(mainTxt,"&&n","\n")
    xml=xml..'<label text="'..mainTxt..'"/>'
    if style==sim.dlgstyle_input then
        xml=xml..'<edit on-editing-finished="__HIDDEN__.dlg.input_callback" id="1"/>'
    end
    if style==sim.dlgstyle_ok or style==sim.dlgstyle_input then
        xml=xml..'<group layout="hbox" flat="true">'
        xml=xml..'<button text="Ok" on-click="__HIDDEN__.dlg.ok_callback"/>'
        xml=xml..'</group>'
    end
    if style==sim.dlgstyle_ok_cancel then
        xml=xml..'<group layout="hbox" flat="true">'
        xml=xml..'<button text="Ok" on-click="__HIDDEN__.dlg.ok_callback"/>'
        xml=xml..'<button text="Cancel" on-click="__HIDDEN__.dlg.cancel_callback"/>'
        xml=xml..'</group>'
    end
    if style==sim.dlgstyle_yes_no then
        xml=xml..'<group layout="hbox" flat="true">'
        xml=xml..'<button text="Yes" on-click="__HIDDEN__.dlg.yes_callback"/>'
        xml=xml..'<button text="No" on-click="__HIDDEN__.dlg.no_callback"/>'
        xml=xml..'</group>'
    end
    xml=xml..'</ui>'
    local ui=simUI.create(xml)
    if style==sim.dlgstyle_input then
        simUI.setEditValue(ui,1,initTxt)
    end
    if not __HIDDEN__.dlg.openDlgs then
        __HIDDEN__.dlg.openDlgs={}
        __HIDDEN__.dlg.openDlgsUi={}
    end
    if not __HIDDEN__.dlg.nextHandle then
        __HIDDEN__.dlg.nextHandle=0
    end
    if dlgHandle then
        retVal=dlgHandle
    else
        retVal=__HIDDEN__.dlg.nextHandle
        __HIDDEN__.dlg.nextHandle=__HIDDEN__.dlg.nextHandle+1
    end
    __HIDDEN__.dlg.openDlgs[retVal]={ui=ui,style=style,state=sim.dlgret_still_open,input=initTxt,title=title,mainTxt=mainTxt,titleCols=titleCols,dlgCols=dlgCols}
    __HIDDEN__.dlg.openDlgsUi[ui]=retVal
    
    if modal then
        while __HIDDEN__.dlg.openDlgs[retVal].state==sim.dlgret_still_open do
            sim.switchThread()
        end
    end
    return retVal
end

function sim.endDialog(dlgHandle)
    if not sim.getBoolParameter(sim_boolparam_headless) then
        assert(type(dlgHandle)=='number' and __HIDDEN__.dlg.openDlgs and __HIDDEN__.dlg.openDlgs[dlgHandle],"Argument 1 is not a valid dialog handle")
        if __HIDDEN__.dlg.openDlgs[dlgHandle].state==sim.dlgret_still_open then
            __HIDDEN__.dlg.removeUi(dlgHandle)
        end
        if __HIDDEN__.dlg.openDlgs[dlgHandle].ui then
            __HIDDEN__.dlg.openDlgsUi[__HIDDEN__.dlg.openDlgs[dlgHandle].ui]=nil
        end
        __HIDDEN__.dlg.openDlgs[dlgHandle]=nil
    end
end

function sim.getDialogInput(dlgHandle)
    if sim.getBoolParameter(sim_boolparam_headless) then
        return ''
    end
    local retVal
    assert(type(dlgHandle)=='number' and __HIDDEN__.dlg.openDlgs and __HIDDEN__.dlg.openDlgs[dlgHandle],"Argument 1 is not a valid dialog handle")
    retVal=__HIDDEN__.dlg.openDlgs[dlgHandle].input
    return retVal
end

function sim.getDialogResult(dlgHandle)
    if sim.getBoolParameter(sim_boolparam_headless) then
        return -1
    end
    local retVal=-1
    assert(type(dlgHandle)=='number' and __HIDDEN__.dlg.openDlgs and __HIDDEN__.dlg.openDlgs[dlgHandle],"Argument 1 is not a valid dialog handle")
    retVal=__HIDDEN__.dlg.openDlgs[dlgHandle].state
    return retVal
end

function math.random2(lower,upper)
    -- same as math.random, but each script has its own generator
    local r=sim.getRandom()
    if lower then
        local b=1
        local d
        if upper then
            b=lower
            d=upper-b
        else
            d=lower-b
        end
        local e=d/(d+1)
        r=b+math.floor(r*d/e)
    end
    return r
end

function math.randomseed2(seed)
    -- same as math.randomseed, but each script has its own generator
    sim.getRandom(seed)
end

function sim.throttle(t,f)
    if __HIDDEN__.lastExecTime==nil then __HIDDEN__.lastExecTime={} end
    local h=string.dump(f)
    local now=sim.getSystemTime()
    if __HIDDEN__.lastExecTime[h]==nil or __HIDDEN__.lastExecTime[h]+t<now then
        f()
        __HIDDEN__.lastExecTime[h]=now
    end
end

function sysCallEx_beforeInstanceSwitch()
    __HIDDEN__.dlg.switch()
end

function sysCallEx_afterInstanceSwitch()
    __HIDDEN__.dlg.switchBack()
end

function sysCallEx_addOnScriptSuspend()
    __HIDDEN__.dlg.switch()
end

function sysCallEx_addOnScriptResume()
    __HIDDEN__.dlg.switchBack()
end

function sysCallEx_cleanup()
    if __HIDDEN__.dlg.openDlgsUi then
        for key,val in pairs(__HIDDEN__.dlg.openDlgsUi) do
            simUI.destroy(key)
        end
    end
end

----------------------------------------------------------


-- Hidden, internal functions:
----------------------------------------------------------
function __HIDDEN__.comparableTables(t1,t2)
    return ( isArray(t1)==isArray(t2) ) or ( isArray(t1) and #t1==0 ) or ( isArray(t2) and #t2==0 )
end

function __HIDDEN__.tableToString(tt,visitedTables,maxLevel,indent)
	indent = indent or 0
    maxLevel=maxLevel-1
	if type(tt) == 'table' then
        if maxLevel<=0 then
            return tostring(tt)
        else
            if  visitedTables[tt] then
                return tostring(tt)..' (already visited)'
            else
                visitedTables[tt]=true
                local sb = {}
                if isArray(tt) then
                    table.insert(sb, '{')
                    for i = 1, #tt do
                        table.insert(sb, __HIDDEN__.anyToString(tt[i], visitedTables,maxLevel, indent))
                        if i < #tt then table.insert(sb, ', ') end
                    end
                    table.insert(sb, '}')
                else
                    table.insert(sb, '{\n')
                    -- Print the map content ordered according to type, then key:
                    local a = {}
                    for n in pairs(tt) do table.insert(a, n) end
                    table.sort(a)
                    local tp={'boolean','number','string','function','userdata','thread','table'}
                    for j=1,#tp,1 do
                        for i,n in ipairs(a) do
                            if type(tt[n])==tp[j] then
                                table.insert(sb, string.rep(' ', indent+4))
                                table.insert(sb, tostring(n))
                                table.insert(sb, '=')
                                table.insert(sb, __HIDDEN__.anyToString(tt[n], visitedTables,maxLevel, indent+4))
                                table.insert(sb, ',\n')
                            end
                        end                
                    end
                    table.insert(sb, string.rep(' ', indent))
                    table.insert(sb, '}')
                end
                visitedTables[tt]=false -- siblings pointing onto a same table should still be explored!
                return table.concat(sb)
            end
        end
    else
        return __HIDDEN__.anyToString(tt, visitedTables,maxLevel, indent)
    end
end

function __HIDDEN__.anyToString(x, visitedTables,maxLevel,tblindent)
    local tblindent = tblindent or 0
    if 'nil' == type(x) then
        return tostring(nil)
    elseif 'table' == type(x) then
        return __HIDDEN__.tableToString(x, visitedTables,maxLevel, tblindent)
    elseif 'string' == type(x) then
        return __HIDDEN__.getShortString(x)
    else
        return tostring(x)
    end
end

function __HIDDEN__.getShortString(x)
    if type(x)=='string' then
        if string.find(x,"\0") then
            return "[buffer string]"
        else
            local a,b=string.gsub(x,"[%a%d%p%s]", "@")
            if b~=#x then
                return "[string containing special chars]"
            else
                if #x>160 then
                    return "[long string]"
                else
                    return string.format('"%s"', x)
                end
            end
        end
    end
    return "[not a string]"
end

function __HIDDEN__.executeAfterLuaStateInit()
    quit=sim.quitSimulator
    exit=sim.quitSimulator
    sim.registerScriptFunction('quit@sim','quit()')
    sim.registerScriptFunction('exit@sim','exit()')
    sim.registerScriptFunction('sim.setDebugWatchList@sim','sim.setDebugWatchList(table vars)')
    sim.registerScriptFunction('sim.getUserVariables@sim','table variables=sim.getUserVariables()')
    sim.registerScriptFunction('sim.getMatchingPersistentDataTags@sim','table tags=sim.getMatchingPersistentDataTags(pattern)')

    sim.registerScriptFunction('sim.displayDialog@sim','number dlgHandle=sim.displayDialog(string title,string mainText,number style,\nboolean modal,string initTxt)')
    sim.registerScriptFunction('sim.getDialogResult@sim','number result=sim.getDialogResult(number dlgHandle)')
    sim.registerScriptFunction('sim.getDialogInput@sim','string input=sim.getDialogInput(number dlgHandle)')
    sim.registerScriptFunction('sim.endDialog@sim','number result=sim.endDialog(number dlgHandle)')
    sim.registerScriptFunction('sim.yawPitchRollToAlphaBetaGamma@sim','number alphaAngle,number betaAngle,number gammaAngle=sim.yawPitchRollToAlphaBetaGamma(\nnumber yawAngle,number pitchAngle,number rollAngle)')
    sim.registerScriptFunction('sim.alphaBetaGammaToYawPitchRoll@sim','number yawAngle,number pitchAngle,number rollAngle=sim.alphaBetaGammaToYawPitchRoll(\nnumber alphaAngle,number betaAngle,number gammaAngle)')
    
    
    if __initFunctions then
        for i=1,#__initFunctions,1 do
            __initFunctions[i]()
        end
        __initFunctions=nil
    end
    
    __HIDDEN__.initGlobals={}
    for key,val in pairs(_G) do
        __HIDDEN__.initGlobals[key]=true
    end
    __HIDDEN__.initGlobals.__HIDDEN__=nil
    __HIDDEN__.executeAfterLuaStateInit=nil
end

function __HIDDEN__.dlg.ok_callback(ui)
    local h=__HIDDEN__.dlg.openDlgsUi[ui]
    __HIDDEN__.dlg.openDlgs[h].state=sim.dlgret_ok
    if __HIDDEN__.dlg.openDlgs[h].style==sim.dlgstyle_input then
        __HIDDEN__.dlg.openDlgs[h].input=simUI.getEditValue(ui,1)
    end
    __HIDDEN__.dlg.removeUi(h)
end

function __HIDDEN__.dlg.cancel_callback(ui)
    local h=__HIDDEN__.dlg.openDlgsUi[ui]
    __HIDDEN__.dlg.openDlgs[h].state=sim.dlgret_cancel
    __HIDDEN__.dlg.removeUi(h)
end

function __HIDDEN__.dlg.input_callback(ui,id,val)
    local h=__HIDDEN__.dlg.openDlgsUi[ui]
    __HIDDEN__.dlg.openDlgs[h].input=val
end

function __HIDDEN__.dlg.yes_callback(ui)
    local h=__HIDDEN__.dlg.openDlgsUi[ui]
    __HIDDEN__.dlg.openDlgs[h].state=sim.dlgret_yes
    __HIDDEN__.dlg.removeUi(h)
end

function __HIDDEN__.dlg.no_callback(ui)
    local h=__HIDDEN__.dlg.openDlgsUi[ui]
    __HIDDEN__.dlg.openDlgs[h].state=sim.dlgret_no
    __HIDDEN__.dlg.removeUi(h)
end

function __HIDDEN__.dlg.removeUi(handle)
    local ui=__HIDDEN__.dlg.openDlgs[handle].ui
    local x,y=simUI.getPosition(ui)
    __HIDDEN__.dlg.openDlgs[handle].previousPos={x,y}
    simUI.destroy(ui)
    __HIDDEN__.dlg.openDlgsUi[ui]=nil
    __HIDDEN__.dlg.openDlgs[handle].ui=nil
end

function __HIDDEN__.dlg.switch()
    if __HIDDEN__.dlg.openDlgsUi then
        for key,val in pairs(__HIDDEN__.dlg.openDlgsUi) do
            local ui=key
            local h=val
            __HIDDEN__.dlg.removeUi(h)
        end
    end
end

function __HIDDEN__.dlg.switchBack()
    if __HIDDEN__.dlg.openDlgsUi then
        local dlgs=sim.unpackTable(sim.packTable(__HIDDEN__.dlg.openDlgs)) -- make a deep copy
        for key,val in pairs(dlgs) do
            if val.state==sim.dlgret_still_open then
                __HIDDEN__.dlg.openDlgs[key]=nil
                sim.displayDialog(val.title,val.mainTxt,val.style,false,val.input,val.titleCols,val.dlgCols,val.previousPos,key)
            end
        end
    end
end
----------------------------------------------------------

-- Hidden, debugging functions:
----------------------------------------------------------
__HIDDEN__.debug={}
function __HIDDEN__.debug.entryFunc(info)
    local scriptName=info[1]
    local funcName=info[2]
    local funcType=info[3]
    local callIn=info[4]
    local debugLevel=info[5]
    local sysCall=info[6]
    local simTime=info[7]
    local simTimeStr=''
    if (debugLevel~=sim.scriptdebug_vars_interval) or (not __HIDDEN__.debug.lastInterval) or (sim.getSystemTimeInMs(-1)>__HIDDEN__.debug.lastInterval+1000) then
        __HIDDEN__.debug.lastInterval=sim.getSystemTimeInMs(-1)
        if sim.getSimulationState()~=sim.simulation_stopped then
            simTimeStr=simTime..' '
        end
        if (debugLevel>=sim.scriptdebug_vars) or (debugLevel==sim.scriptdebug_vars_interval) then
            local prefix='DEBUG: '..simTimeStr..' '
            local t=__HIDDEN__.debug.getVarChanges(prefix)
            if t then
                sim.addLog(sim.verbosity_msgs,t)
            end
        end
        if (debugLevel==sim.scriptdebug_allcalls) or (debugLevel==sim.scriptdebug_callsandvars) or ( (debugLevel==sim.scriptdebug_syscalls) and sysCall) then
            local t='DEBUG: '..simTimeStr
            if callIn then
                t=t..' --> '
            else
                t=t..' <-- '
            end
            t=t..funcName..' ('..funcType..')'
            sim.addLog(sim.verbosity_msgs,t)
        end
    end
end

function __HIDDEN__.debug.getVarChanges(pref)
    local t=''
    __HIDDEN__.debug.userVarsOld=__HIDDEN__.debug.userVars
    __HIDDEN__.debug.userVars=sim.unpackTable(sim.packTable(sim.getUserVariables())) -- deep copy
    if __HIDDEN__.debug.userVarsOld then
        if __HIDDEN__.debug.watchList and type(__HIDDEN__.debug.watchList)=='table' and #__HIDDEN__.debug.watchList>0 then
            for i=1,#__HIDDEN__.debug.watchList,1 do
                local str=__HIDDEN__.debug.watchList[i]
                if type(str)=='string' then
                    local var1=__HIDDEN__.debug.getVar('__HIDDEN__.debug.userVarsOld.'..str)
                    local var2=__HIDDEN__.debug.getVar('__HIDDEN__.debug.userVars.'..str)
                    if var1~=nil or var2~=nil then
                        t=__HIDDEN__.debug.getVarDiff(pref,str,var1,var2)
                    end
                end
            end
        else
            t=__HIDDEN__.debug.getVarDiff(pref,'',__HIDDEN__.debug.userVarsOld,__HIDDEN__.debug.userVars)
        end
    end
    __HIDDEN__.debug.userVarsOld=nil
    if #t>0 then
--        t=t:sub(1,-2) -- remove last linefeed
        t=t:sub(1,-4) -- remove last linefeed
        return t
    end
end

function __HIDDEN__.debug.getVar(varName)
    local f=loadstring('return '..varName)
    if f then
        local res,val=pcall(f)
        if res and val then
            return val
        end
    end
end

function __HIDDEN__.debug.getVarDiff(pref,varName,oldV,newV)
    local t=''
    local lf='<br>'--'\n'
    if ( type(oldV)==type(newV) ) and ( (type(oldV)~='table') or __HIDDEN__.comparableTables(oldV,newV) )  then  -- comparableTables: an empty map is seen as an array
        if type(newV)~='table' then
            if newV~=oldV then
                t=t..pref..'mod: '..varName..' ('..type(newV)..'): '..__HIDDEN__.getShortString(tostring(newV))..lf
            end
        else
            if isArray(oldV) and isArray(newV) then -- an empty map is seen as an array
                -- removed items:
                if #oldV>#newV then
                    for i=1,#oldV-#newV,1 do
                        t=t..__HIDDEN__.debug.getVarDiff(pref,varName..'['..i+#oldV-#newV..']',oldV[i+#oldV-#newV],nil)
                    end
                end
                -- added items:
                if #newV>#oldV then
                    for i=1,#newV-#oldV,1 do
                        t=t..__HIDDEN__.debug.getVarDiff(pref,varName..'['..i+#newV-#oldV..']',nil,newV[i+#newV-#oldV])
                    end
                end
                -- modified items:
                local l=math.min(#newV,#oldV)
                for i=1,l,1 do
                    t=t..__HIDDEN__.debug.getVarDiff(pref,varName..'['..i..']',oldV[i],newV[i])
                end
            else
                local nvarName=varName
                if nvarName~='' then nvarName=nvarName..'.' end
                -- removed items:
                for k,vo in pairs(oldV) do
                    if newV[k]==nil then
                        t=t..__HIDDEN__.debug.getVarDiff(pref,nvarName..k,vo,nil)
                    end
                end
                
                -- added items:
                for k,vn in pairs(newV) do
                    if oldV[k]==nil then
                        t=t..__HIDDEN__.debug.getVarDiff(pref,nvarName..k,nil,vn)
                    end
                end
                
                -- modified items:
                for k,vo in pairs(oldV) do
                    if newV[k] then
                        t=t..__HIDDEN__.debug.getVarDiff(pref,nvarName..k,vo,newV[k])
                    end
                end
            end
        end
    else
        if oldV==nil then
            if type(newV)~='table' then
                t=t..pref..'new: '..varName..' ('..type(newV)..'): '..__HIDDEN__.getShortString(tostring(newV))..lf
            else
                t=t..pref..'new: '..varName..' ('..type(newV)..')'..lf
                if isArray(newV) then
                    for i=1,#newV,1 do
                        t=t..__HIDDEN__.debug.getVarDiff(pref,varName..'['..i..']',nil,newV[i])
                    end
                else
                    local nvarName=varName
                    if nvarName~='' then nvarName=nvarName..'.' end
                    for k,v in pairs(newV) do
                        t=t..__HIDDEN__.debug.getVarDiff(pref,nvarName..k,nil,v)
                    end
                end
            end
        elseif newV==nil then
            if type(oldV)~='table' then
                t=t..pref..'del: '..varName..' ('..type(oldV)..'): '..__HIDDEN__.getShortString(tostring(oldV))..lf
            else
                t=t..pref..'del: '..varName..' ('..type(oldV)..')'..lf
            end
        else
            -- variable changed type.. register that as del and new:
            t=t..__HIDDEN__.debug.getVarDiff(pref,varName,oldV,nil)
            t=t..__HIDDEN__.debug.getVarDiff(pref,varName,nil,newV)
        end
    end
    return t
end
----------------------------------------------------------

-- Old stuff, mainly for backward compatibility:
----------------------------------------------------------
function sim.include(relativePathAndFile,cmd) require("sim_old") return sim.include(relativePathAndFile,cmd) end
function sim.includeRel(relativePathAndFile,cmd) require("sim_old") return sim.includeRel(relativePathAndFile,cmd) end
function sim.includeAbs(absPathAndFile,cmd) require("sim_old") return sim.includeAbs(absPathAndFile,cmd) end
function sim.canScaleObjectNonIsometrically(objHandle,scaleAxisX,scaleAxisY,scaleAxisZ) require("sim_old") return sim.canScaleObjectNonIsometrically(objHandle,scaleAxisX,scaleAxisY,scaleAxisZ) end
function sim.canScaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,ignoreNonScalableItems) require("sim_old") return sim.canScaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,ignoreNonScalableItems) end
function sim.scaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) require("sim_old") return sim.scaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) end
function sim.UI_populateCombobox(ui,id,items_array,exceptItems_map,currentItem,sort,additionalItemsToTop_array) require("sim_old") return sim.UI_populateCombobox(ui,id,items_array,exceptItems_map,currentItem,sort,additionalItemsToTop_array) end
----------------------------------------------------------

return sim
