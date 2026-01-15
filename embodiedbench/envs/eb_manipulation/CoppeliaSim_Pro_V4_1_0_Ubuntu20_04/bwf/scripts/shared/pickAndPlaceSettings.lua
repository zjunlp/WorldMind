model.pickPlaceDlg={}

function model.pickPlaceDlg.refresh()
    if model.pickPlaceDlg.pickPlaceData.ui then
        local sel=simBWF.getSelectedEditWidget(model.pickPlaceDlg.pickPlaceData.ui)
        if model.pickPlaceDlg.pickPlaceData.forGripper then
            simUI.setEnabled(model.pickPlaceDlg.pickPlaceData.ui,1001,true,true)
            simUI.setEnabled(model.pickPlaceDlg.pickPlaceData.ui,1002,true,true)
            simUI.setEnabled(model.pickPlaceDlg.pickPlaceData.ui,1003,true,true)
        else
            local enabled=model.pickPlaceDlg.pickPlaceData.overrideGripperSettings
            simUI.setCheckboxValue(model.pickPlaceDlg.pickPlaceData.ui,4,simBWF.getCheckboxValFromBool(enabled),true)
            simUI.setEnabled(model.pickPlaceDlg.pickPlaceData.ui,1001,enabled,true)
            simUI.setEnabled(model.pickPlaceDlg.pickPlaceData.ui,1002,enabled,true)
        end
        simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,1,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.speed*100),true)
        simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,2,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.accel*100),true)
        
        for i=1,2,1 do
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,10+i,simBWF.format("%.3f",model.pickPlaceDlg.pickPlaceData.dwellTime[i]),true)
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,20+i,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.approachHeight[i]*1000),true)
            simUI.setCheckboxValue(model.pickPlaceDlg.pickPlaceData.ui,100+i,simBWF.getCheckboxValFromBool(model.pickPlaceDlg.pickPlaceData.useAbsoluteApproachHeight[i]),true)
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,90+i,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.departHeight[i]*1000),true)
            local off=model.pickPlaceDlg.pickPlaceData.offset[i]
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,30+i,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,40+i,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.rounding[i]*1000),true)
            simUI.setEditValue(model.pickPlaceDlg.pickPlaceData.ui,50+i,simBWF.format("%.0f",model.pickPlaceDlg.pickPlaceData.nullingAccuracy[i]*1000),true)
            --if i==1 then -- currently doesn't make sense with place 
            simUI.setCheckboxValue(model.pickPlaceDlg.pickPlaceData.ui,80+i,simBWF.getCheckboxValFromBool(model.pickPlaceDlg.pickPlaceData.relativeToBelt[i]),true)
            --nd
            if not model.pickPlaceDlg.pickPlaceData.forGripper then
                break
            end
        end
        simBWF.setSelectedEditWidget(model.pickPlaceDlg.pickPlaceData.ui,sel)
    end
end

--function loadTheString()
--    local f=loadstring("local counter=0\n".."return "..theString)
--    return f
--end

function model.pickPlaceDlg.editActions_callback()
    local s="500 800"
    local p="200 200"
    if actionsDlgSize then
        s=actionsDlgSize[1]..' '..actionsDlgSize[2]
    end
    if actionsDlgPos then
        p=actionsDlgPos[1]..' '..actionsDlgPos[2]
    end
    local xml = [[ <editor title="Actions" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            

    local txt="actionTemplates=\n{\n"
    for key,value in pairs(model.pickPlaceDlg.pickPlaceData.actionTemplates) do
        txt=txt.."    "..key..'={cmd="'..value.cmd..'"},\n'
    end
    txt=txt.."}\n\n"

    txt=txt.."pickActions=\n{\n"
    for i=1,#model.pickPlaceDlg.pickPlaceData.pickActions do
        local action=model.pickPlaceDlg.pickPlaceData.pickActions[i]
        txt=txt..simBWF.format('    {name="%s",dt=%.3f},\n',action.name,action.dt)
    end
    txt=txt.."}\n\n"
    txt=txt.."--[[multiPickActions\n\nActions used for multipick\n (2nd,3rd,4th,...)\n\n--]]\n\n"
    txt=txt.."multiPickActions=\n{\n"
    for i=1,#model.pickPlaceDlg.pickPlaceData.multiPickActions do
        local action=model.pickPlaceDlg.pickPlaceData.multiPickActions[i]
        txt=txt..simBWF.format('    {name="%s",dt=%.3f},\n',action.name,action.dt)
    end
    txt=txt.."}\n\n"


    if model.pickPlaceDlg.pickPlaceData.forGripper then
        txt=txt.."placeActions=\n{\n"
        for i=1,#model.pickPlaceDlg.pickPlaceData.placeActions do
            local action=model.pickPlaceDlg.pickPlaceData.placeActions[i]
            txt=txt..simBWF.format('    {name="%s",dt=%.3f},\n',action.name,action.dt)
        end
        txt=txt.."}\n"
    end
    
    txt=txt.."\n--[[tmpRem\n\nAction names are case sensitive!\n\n--]]"
    
    while true do
        local success=false
        local errorMsg=""
        txt,actionsDlgSize,actionsDlgPos=sim.openTextEditor(txt,xml)
        
        actionTemplates=nil
        pickActions=nil
        multiPickActions=nil
        placeActions=nil
        local t=loadstring(txt)
        if type(t)=='function' then
            local res,err=xpcall(t,function(err) return debug.traceback(err) end)
            if res and type(actionTemplates)=='table' and type(pickActions)=='table' and type(multiPickActions)=='table' and ( type(placeActions)=='table' or (not model.pickPlaceDlg.pickPlaceData.forGripper) ) then
                local templates={}
                local pickAct={}
                local multiPickAct={}
                local placeAct={}
                for key,value in pairs(actionTemplates) do
                    if type(value.cmd)=='string' then
                        templates[key]={cmd=value.cmd}
                    else
                        errorMsg="Action template '"..key.."' has missing or wrong 'cmd' field"
                        break
                    end
                end
                if #errorMsg==0 then
                    local actions={pickActions, multiPickActions,placeActions}
                    local actionsStr={"Pick", "MultiPick","Place"}
                    local actionsTmp={pickAct, multiPickAct,placeAct}
                    for j=1,3,1 do
                        local act=actions[j]
                        local actStr=actionsStr[j]
                        local actTmp=actionsTmp[j]
                        for i=1,#act,1 do
                            local value=act[i]
                            if type(value)=='table' then
                                if type(value.name)=='string' and templates[value.name] then
                                    if type(value.dt)=='number' then
                                        actTmp[#actTmp+1]={name=value.name,dt=value.dt}
                                    else
                                        errorMsg=actStr.." action item at index "..i.." has missing or wrong 'dt' field"
                                        break
                                    end
                                else
                                    errorMsg=actStr.." action item at index "..i.." has missing or wrong 'name' field"
                                    break
                                end
                            else
                                errorMsg=actStr.." action item at index "..i.." is not a table"
                                break
                            end
                        end
                        if #errorMsg~=0 then
                            break
                        end
                        if j>=2 and not model.pickPlaceDlg.pickPlaceData.forGripper then
                            break
                        end
                    end
                end
                if #errorMsg==0 then
                    success=true
                    model.pickPlaceDlg.pickPlaceData.actionTemplates=templates
                    model.pickPlaceDlg.pickPlaceData.pickActions=pickAct
                    model.pickPlaceDlg.pickPlaceData.multiPickActions=multiPickAct
                    model.pickPlaceDlg.pickPlaceData.placeActions=placeAct
                end
            else
                if res then
                    if type(placeActions)~='table' and model.pickPlaceDlg.pickPlaceData.forGripper then
                        errorMsg="Missing table 'placeActions'"
                    end
                    if type(pickActions)~='table' then
                        errorMsg="Missing table 'pickActions'"
                    end
                    if type(multiPickActions)~='table' then
                        errorMsg="Missing table 'multiPickActions'"
                    end
                    if type(actionTemplates)~='table' then
                        errorMsg="Missing table 'actionTemplates'"
                    end
                end
            end
        end
        if success then
            break
        end
        if #errorMsg==0 then
            if sim.msgbox_return_no==sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_yesno,'Input Error',"The input contains errors. Do you wish to correct them?.") then
                break
            end
        else
            if sim.msgbox_return_no==sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_yesno,'Input Error',"The input contains an error:\n\n"..errorMsg.."\n\nDo you wish to correct it?") then
                break
            end
        end
    end
    actionTemplates=nil
    pickActions=nil
    multiPickActions=nil
    placeActions=nil
end

function model.pickPlaceDlg.velocityChange_callback(ui,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100 then newValue=100 end
        newValue=newValue/100
        if newValue~=model.pickPlaceDlg.pickPlaceData.speed then
            model.pickPlaceDlg.pickPlaceData.speed=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.accelerationChange_callback(ui,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100 then newValue=100 end
        newValue=newValue/100
        if newValue~=model.pickPlaceDlg.pickPlaceData.accel then
            model.pickPlaceDlg.pickPlaceData.accel=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.dwellTimeChange_callback(ui,id,newValue)
    local ind=id-10
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0.01 then newValue=0.01 end
        if newValue>1 then newValue=1 end
        if newValue~=model.pickPlaceDlg.pickPlaceData.dwellTime[ind] then
            model.pickPlaceDlg.pickPlaceData.dwellTime[ind]=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.approachHeightChange_callback(ui,id,newValue)
    local ind=id-20
    newValue=tonumber(newValue)
    if newValue then
        if newValue<10 then newValue=10 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=model.pickPlaceDlg.pickPlaceData.approachHeight[ind] then
            model.pickPlaceDlg.pickPlaceData.approachHeight[ind]=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.useAbsoluteApproachHeight_callback(ui,id,newValue)
    local ind=id-100
    model.pickPlaceDlg.pickPlaceData.useAbsoluteApproachHeight[ind]=not model.pickPlaceDlg.pickPlaceData.useAbsoluteApproachHeight[ind]
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.departHeightChange_callback(ui,id,newValue)
    local ind=id-90
    newValue=tonumber(newValue)
    if newValue then
        if newValue<10 then newValue=10 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=model.pickPlaceDlg.pickPlaceData.departHeight[ind] then
            model.pickPlaceDlg.pickPlaceData.departHeight[ind]=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.offsetChange_callback(ui,id,newValue)
    local ind=id-30
    local i=1
    local t={0,0,0}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    model.pickPlaceDlg.pickPlaceData.offset[ind]={t[1],t[2],t[3]}
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.roundingChange_callback(ui,id,newValue)
    local ind=id-40
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>500 then newValue=500 end
        newValue=newValue/1000
        if newValue~=model.pickPlaceDlg.pickPlaceData.rounding[ind] then
            model.pickPlaceDlg.pickPlaceData.rounding[ind]=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.nullingChange_callback(ui,id,newValue)
    local ind=id-50
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>50 then newValue=50 end
        newValue=newValue/1000
        if newValue~=model.pickPlaceDlg.pickPlaceData.nullingAccuracy[ind] then
            model.pickPlaceDlg.pickPlaceData.nullingAccuracy[ind]=newValue
        end
    end
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.relativeToBeltChange_callback(ui,id,newValue)
    local ind=id-80
    model.pickPlaceDlg.pickPlaceData.relativeToBelt[ind]=not model.pickPlaceDlg.pickPlaceData.relativeToBelt[ind]
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.overrideGripperChange_callback(ui,id,newValue)
    model.pickPlaceDlg.pickPlaceData.overrideGripperSettings=not model.pickPlaceDlg.pickPlaceData.overrideGripperSettings
    model.pickPlaceDlg.refresh()
end

function model.pickPlaceDlg.apply_callback()
    if not model.pickPlaceDlg.areRobotPickPlaceDataSame(model.pickPlaceDlg.pickPlaceData,model.pickPlaceDlg.pickPlaceData.original) then
        model.pickPlaceDlg.pickPlaceData.applyCallback(model.pickPlaceDlg.pickPlaceData)
        model.pickPlaceDlg.pickPlaceData.original=nil
        local original={}
        for key,value in pairs(model.pickPlaceDlg.pickPlaceData) do
            original[key]=sim.unpackTable(sim.packTable({value}))[1] -- will make a deep copy
        end
        model.pickPlaceDlg.pickPlaceData.original=original
    end
end

function model.pickPlaceDlg.close_callback()
    local leave=true
    if not model.pickPlaceDlg.areRobotPickPlaceDataSame(model.pickPlaceDlg.pickPlaceData,model.pickPlaceDlg.pickPlaceData.original) then
        leave=(sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,"Unsaved changes","Do want to leave without applying the changes?")) 
    end
    if leave then
        local x,y=simUI.getPosition(model.pickPlaceDlg.pickPlaceData.ui)
        model.pickPlaceDlg.pickPlaceData.closeCallback({x,y})
        simUI.destroy(model.pickPlaceDlg.pickPlaceData.ui)
        model.pickPlaceDlg.pickPlaceData=nil
    end
end

function model.pickPlaceDlg.areRobotPickPlaceDataSame(data1,data2)
    if data1.overrideGripperSettings~=data2.overrideGripperSettings then return false end
    if data1.speed~=data2.speed then return false end
    if data1.accel~=data2.accel then return false end
    for i=1,2,1 do
        if data1.dwellTime[i]~=data2.dwellTime[i] then return false end
        if data1.approachHeight[i]~=data2.approachHeight[i] then return false end
        if data1.useAbsoluteApproachHeight[i]~=data2.useAbsoluteApproachHeight[i] then return false end
        if data1.departHeight[i]~=data2.departHeight[i] then return false end
        if data1.rounding[i]~=data2.rounding[i] then return false end
        if data1.nullingAccuracy[i]~=data2.nullingAccuracy[i] then return false end
        for j=1,3,1 do
            if data1.offset[i][j]~=data2.offset[i][j] then return false end
        end
        if sim.packTable(data1.actionTemplates)~=sim.packTable(data2.actionTemplates) then return false end
        if sim.packTable(data1.pickActions)~=sim.packTable(data2.pickActions) then return false end
        if sim.packTable(data1.multiPickActions)~=sim.packTable(data2.multiPickActions) then return false end
        if sim.packTable(data1.placeActions)~=sim.packTable(data2.placeActions) then return false end
        if data1.relativeToBelt[i]~=data2.relativeToBelt[i] then return false end
    end
    return true
end

function model.pickPlaceDlg.display(settingsData,title,forGripper,applyCallback,closeCallback,previousDlgPos)
    local xml=''
    
    if not forGripper then
        xml=xml..'[[<checkbox text="Override gripper pick settings" on-change="model.pickPlaceDlg.overrideGripperChange_callback" id="4" />'
    end
    xml=xml..'<tabs>'
    xml=xml..[[
            <tab title="Pick" layout="form" id="1002">
                <label text="Dwell time (s)"/>
                <edit on-editing-finished="model.pickPlaceDlg.dwellTimeChange_callback" id="11"/>
                
                <label text="Approach height (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.approachHeightChange_callback" id="21"/>

                <label text="Depart height (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.departHeightChange_callback" id="91"/>
                
                <label text=""/>
                <checkbox text="relative to red ball" on-change="model.pickPlaceDlg.useAbsoluteApproachHeight_callback" id="101" />

                <label text="Rounding (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.roundingChange_callback" id="41"/>

                <label text="Nulling accuracy (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.nullingChange_callback" id="51"/>

                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.offsetChange_callback" id="31"/>

                <label text=""/>
                <checkbox text="relative to belt" on-change="model.pickPlaceDlg.relativeToBeltChange_callback" id="81" />
            </tab>
            ]]
    if forGripper then
        xml=xml..[[
            <tab title="Place" layout="form" id="1003">
                <label text="Dwell time (s)"/>
                <edit on-editing-finished="model.pickPlaceDlg.dwellTimeChange_callback" id="12"/>
                
                <label text="Approach height (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.approachHeightChange_callback" id="22"/>

                <label text="Depart height (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.departHeightChange_callback" id="92"/>

                <label text=""/>
                <checkbox text="relative to red ball" on-change="model.pickPlaceDlg.useAbsoluteApproachHeight_callback" id="102" />
                
                <label text="Rounding (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.roundingChange_callback" id="42"/>

                <label text="Nulling accuracy (mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.nullingChange_callback" id="52"/>

                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.pickPlaceDlg.offsetChange_callback" id="32"/>

                <label text=""/>
                <checkbox text="relative to belt" on-change="model.pickPlaceDlg.relativeToBeltChange_callback" id="82" />
            </tab>
            ]]
    end
    local tabTitle='Pick && place'
    if not forGripper then
        tabTitle='More'
    end
    xml=xml..'<tab title="'..tabTitle..'" layout="form" id="1001">'
    xml=xml..[[
                <label text="Maximum speed (%)"/>
                <edit on-editing-finished="model.pickPlaceDlg.velocityChange_callback" id="1"/>

                <label text="Maximum acceleration (%)"/>
                <edit on-editing-finished="model.pickPlaceDlg.accelerationChange_callback" id="2"/>

                <label text="Gripper actions"/>
                <button text="Edit" on-click="model.pickPlaceDlg.editActions_callback"/>
                ]]
    xml=xml..'</tab>'
    xml=xml..'</tabs>'
    xml=xml..[[
            <group layout="form" flat="true">
                <button text="Apply" on-click="model.pickPlaceDlg.apply_callback"/>
                <button text="Close" on-click="model.pickPlaceDlg.close_callback"/>
            </group>
            ]]
--                <label text="<a href='#L1'>help</a>" style="* {qproperty-alignment: AlignRight}" on-link-activated="f"/>

    model.pickPlaceDlg.pickPlaceData={}
    model.pickPlaceDlg.pickPlaceData.ui=simBWF.createCustomUi(xml,title,previousDlgPos,false,'',true,false)-- ,activate,additionalUiAttribute--]])
    model.pickPlaceDlg.pickPlaceData.applyCallback=applyCallback
    model.pickPlaceDlg.pickPlaceData.closeCallback=closeCallback
    model.pickPlaceDlg.pickPlaceData.forGripper=forGripper
    model.pickPlaceDlg.pickPlaceData.original={}
    for key,value in pairs(settingsData) do
        model.pickPlaceDlg.pickPlaceData[key]=sim.unpackTable(sim.packTable({value}))[1] -- will make a deep copy
        model.pickPlaceDlg.pickPlaceData.original[key]=sim.unpackTable(sim.packTable({value}))[1] -- will make a deep copy
    end
    model.pickPlaceDlg.refresh()
end
