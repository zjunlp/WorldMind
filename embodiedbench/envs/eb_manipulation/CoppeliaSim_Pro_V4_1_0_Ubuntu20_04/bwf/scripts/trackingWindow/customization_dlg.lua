model.dlg={}
model.dlg.calibrationDlg={}

function model.dlg.calibrationDlg.onCloseDlg()
    local data={}
    data.idRobot=model.getAssociatedRobotHandle()
    data.id=model.handle
    simBWF.query('trackingWindow_trainEnd',data)
    simUI.destroy(model.dlg.calibrationDlg.data.ui)
    model.dlg.calibrationDlg.data=nil
    sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,"Calibration","Calibration procedure aborted")
end

function model.dlg.calibrationDlg.calibrationBallClick_callback(ui,id,newVal)
    local toleranceTesting=0.1
    local weHaveAProblem=false
    local associatedRobot=model.getAssociatedRobotHandle()
    local associatedRobotRef=simBWF.callCustomizationScriptFunction('model.ext.getReferenceObject',associatedRobot)
    if #model.dlg.calibrationDlg.data.relativeBallPositions==0 then
        -- just clicked red.
        simUI.setEnabled(model.dlg.calibrationDlg.data.ui,2,false)
        simUI.setEnabled(model.dlg.calibrationDlg.data.ui,3,true)
        simUI.setLabelText(model.dlg.calibrationDlg.data.ui,1,"Move the gripper platform to the green ball, then click 'Green' below")

        local data={}
        data.idRobot=model.getAssociatedRobotHandle()
        data.id=model.handle
        data.ballIndex=0
        local reply,replyData=simBWF.query('trackingWindow_train',data)
        if simBWF.isInTestMode() then
            reply='ok'
            local dat=sim.getObjectPosition(model.handles.calibrationBalls[1],associatedRobotRef)
            replyData.pos={dat[1]-toleranceTesting*math.random()*toleranceTesting*2,dat[2]-toleranceTesting*math.random()*toleranceTesting*2,dat[3]-toleranceTesting*math.random()*toleranceTesting*2}
            replyData.ballIndex=data.ballIndex
        end
        if reply=='ok' then
            if replyData.ballIndex==data.ballIndex then
                model.dlg.calibrationDlg.data.relativeBallPositions[1]=replyData.pos
            else
                weHaveAProblem="Strange reply from the plugin"
            end
        else
            if reply then
                weHaveAProblem=reply.error
            else
                weHaveAProblem="Problem with the plugin"
            end
        end
    else
        if #model.dlg.calibrationDlg.data.relativeBallPositions==1 then
            -- just clicked green.
            simUI.setEnabled(model.dlg.calibrationDlg.data.ui,3,false)
            simUI.setEnabled(model.dlg.calibrationDlg.data.ui,4,true)
            simUI.setLabelText(model.dlg.calibrationDlg.data.ui,1,"Move the gripper platform to the blue ball, then click 'Blue' below")

            local data={}
            data.idRobot=model.getAssociatedRobotHandle()
            data.id=model.handle
            data.ballIndex=1
            local reply,replyData=simBWF.query('trackingWindow_train',data)
            if simBWF.isInTestMode() then
                reply='ok'
                local dat=sim.getObjectPosition(model.handles.calibrationBalls[2],associatedRobotRef)
                replyData.pos={dat[1]-toleranceTesting*math.random()*toleranceTesting*2,dat[2]-toleranceTesting*math.random()*toleranceTesting*2,dat[3]-toleranceTesting*math.random()*toleranceTesting*2}
                replyData.ballIndex=data.ballIndex
            end
            if reply=='ok' then
                if replyData.ballIndex==data.ballIndex then
                    model.dlg.calibrationDlg.data.relativeBallPositions[2]=replyData.pos
                    local d=simBWF.getPtPtDistance(model.dlg.calibrationDlg.data.relativeBallPositions[1],model.dlg.calibrationDlg.data.relativeBallPositions[2])
                    if d<0.08 then
                        weHaveAProblem='The green ball is too close to the red ball.'
                    end
                else
                    weHaveAProblem="Strange reply from the plugin"
                end
            else
                if reply then
                    weHaveAProblem=reply.error
                else
                    weHaveAProblem="Problem with the plugin"
                end
            end
        else
            -- just clicked blue.
            local data={}
            data.idRobot=associatedRobot
            data.id=model.handle
            data.ballIndex=2
            local reply,replyData=simBWF.query('trackingWindow_train',data)
            if simBWF.isInTestMode() then
                reply='ok'
                local dat=sim.getObjectPosition(model.handles.calibrationBalls[3],associatedRobotRef)
                replyData.pos={dat[1]-toleranceTesting*math.random()*toleranceTesting*2,dat[2]-toleranceTesting*math.random()*toleranceTesting*2,dat[3]-toleranceTesting*math.random()*toleranceTesting*2}
                replyData.ballIndex=data.ballIndex
            end
            if reply=='ok' then
                if replyData.ballIndex==data.ballIndex then
                    model.dlg.calibrationDlg.data.relativeBallPositions[3]=replyData.pos
                    local d1=simBWF.getPtPtDistance(model.dlg.calibrationDlg.data.relativeBallPositions[1],model.dlg.calibrationDlg.data.relativeBallPositions[3])
                    local d2=simBWF.getPtPtDistance(model.dlg.calibrationDlg.data.relativeBallPositions[2],model.dlg.calibrationDlg.data.relativeBallPositions[3])
                    if d1<0.08 or d2<0.08 then
                        weHaveAProblem='The blue ball is too close to the red or green ball.'
                    else
                        local c=model.readInfo()
                        local calData=model.dlg.calibrationDlg.data.relativeBallPositions
                        c['calibration']=calData
                        -- Find the matrix:
                        local m=simBWF.getMatrixFromCalibrationBallPositions(calData[1],calData[2],calData[3])
                        c['calibrationMatrix']=m
                        model.writeInfo(c)
                        simUI.destroy(model.dlg.calibrationDlg.data.ui)
                        model.dlg.calibrationDlg.data=nil
                        model.applyCalibrationColor()
                        local data={}
                        data.idRobot=model.getAssociatedRobotHandle()
                        data.id=model.handle
                        simBWF.query('trackingWindow_trainEnd',data)
                        model.updatePluginRepresentation()
                    end
                else
                    weHaveAProblem="Strange reply from the plugin"
                end
            else
                if reply then
                    weHaveAProblem=reply.error
                else
                    weHaveAProblem="Problem with the plugin"
                end
            end
        end
    end
    if weHaveAProblem then
        local data={}
        data.idRobot=model.getAssociatedRobotHandle()
        data.id=model.handle
        simBWF.query('trackingWindow_trainEnd',data)
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Calibration",weHaveAProblem)
        simUI.destroy(model.dlg.calibrationDlg.data.ui)
        model.dlg.calibrationDlg.data=nil
    end
end

-------------------------------------------------------
-------------------------------------------------------

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local config=model.readInfo()
        local startStopLine=sim.boolAnd32(config['bitCoded'],16)~=0
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,23,simStopped,true)
        simUI.setEnabled(model.dlg.ui,1,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3,simStopped,true)
        simUI.setEnabled(model.dlg.ui,5,simStopped,true)
        simUI.setEnabled(model.dlg.ui,51,startStopLine,true)
        simUI.setEnabled(model.dlg.ui,52,startStopLine,true)
        simUI.setEnabled(model.dlg.ui,53,startStopLine,true)
        simUI.setEnabled(model.dlg.ui,11,simStopped,true)
        simUI.setEnabled(model.dlg.ui,100,simStopped,true)
        simUI.setEnabled(model.dlg.ui,101,simStopped,true)
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)

        simUI.setEditValue(model.dlg.ui,21,simBWF.format("%.0f , %.0f , %.0f",config.sizes[1]*1000,config.sizes[2]*1000,config.sizes[3]*1000),true)
        simUI.setEditValue(model.dlg.ui,24,simBWF.format("%.0f , %.0f , %.0f",config.offsets[1]*1000,config.offsets[2]*1000,config.offsets[3]*1000),true)

        local d=config['calibrationBallOffset']
        simUI.setEditValue(model.dlg.ui,23,simBWF.format("%.0f , %.0f , %.0f",d[1]/0.001,d[2]/0.001,d[3]/0.001),true)

        simUI.setCheckboxValue(model.dlg.ui,50,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)
        simUI.setEditValue(model.dlg.ui,51,simBWF.format("%.0f",config['stopLinePos']/0.001),true)
        simUI.setEditValue(model.dlg.ui,52,simBWF.format("%.0f",config['startLinePos']/0.001),true)
        simUI.setEditValue(model.dlg.ui,61,simBWF.format("%.0f",config['upstreamMarginPos']/0.001),true)
        simUI.setCheckboxValue(model.dlg.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,1,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,2,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)

        local c=model.readInfo()
        local loc=model.getAvailableInputs()
        model.dlg.comboInput=simBWF.populateCombobox(model.dlg.ui,11,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT)),true,{{simBWF.NONE_TEXT,-1}})
        local locPT=simBWF.getAllPartsFromPartRepository()
        model.dlg.comboPartType=simBWF.populateCombobox(model.dlg.ui,53,locPT,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PARTTYPE)),true,{{simBWF.NONE_TEXT,-1}})
        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end

function model.dlg.hidden_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.calibrationBallsHidden_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.createParts_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.showPoints_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.sizes
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0.1 then t[i]=0.1 end
            if t[i]>1 then t[i]=1 end
        end
        if i==2 then
            if t[i]<0.1 then t[i]=0.1 end
            if t[i]>1 then t[i]=1 end
        end
        if i==3 then
            if t[i]<0.1 then t[i]=0.1 end
            if t[i]>1 then t[i]=1 end
        end
        i=i+1
    end
    c.sizes=t
    model.writeInfo(c)
    model.setSizes()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.offsetChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.offsets
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<-0.5 then t[i]=-0.5 end
            if t[i]>0.5 then t[i]=0.5 end
        end
        if i==2 then
            if t[i]<-1 then t[i]=-1 end
            if t[i]>1 then t[i]=1 end
        end
        if i==3 then
            if t[i]<-0.5 then t[i]=-0.5 end
            if t[i]>0.5 then t[i]=0.5 end
        end
        i=i+1
    end
    c.offsets=t
    model.writeInfo(c)
    model.setSizes()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.startStopLine_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    model.writeInfo(c)
    model.setSizes()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.partTypeChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboPartType[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PARTTYPE,newLoc)
    local c=model.readInfo()
    model.writeInfo(c)
    model.setSizes()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.stopLineChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v>1 then v=1 end
        if v<0.01 then v=0.01 end
        local w=c['startLinePos']
        if v<w+0.01 then
            w=v-0.01
        end
        if v~=c['stopLinePos'] or w~=c['startLinePos'] then
            c['stopLinePos']=v
            c['startLinePos']=w
            model.writeInfo(c)
            model.setSizes()
            model.updatePluginRepresentation()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.startLineChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v>1-0.01 then v=1-0.01 end
        local w=c['stopLinePos']
        if v>w-0.01 then
            w=v+0.01
        end
        if v<0 then v=0 end
        if v~=c['startLinePos'] or w~=c['stopLinePos'] then
            c['startLinePos']=v
            c['stopLinePos']=w
            model.writeInfo(c)
            model.setSizes()
            model.updatePluginRepresentation()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.upstreamMarginChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v>1 then v=1 end
        if v<0 then v=0 end
        if v~=c['upstreamMarginPos'] then
            c['upstreamMarginPos']=v
            model.writeInfo(c)
            model.setSizes()
            model.updatePluginRepresentation()
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.calibrationBallOffsetChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local i=1
    local t=c['calibrationBallOffset']
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0 then t[i]=0 end
            if t[i]>20 then t[i]=20 end
        end
        if i==2 then
            if t[i]<-2 then t[i]=2 end
            if t[i]>2 then t[i]=2 end
        end
        if i==3 then
            if t[i]<-0.5 then t[i]=-0.5 end
            if t[i]>0.5 then t[i]=0.5 end
        end
        i=i+1
    end
    c['calibrationBallOffset']=t
    model.writeInfo(c)
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    model.updatePluginRepresentation()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.inputChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.comboInput[newIndex+1][2]
    if newLoc>=0 then
        simBWF.forbidInputForTrackingWindowChainItems(newLoc)
    end
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT,newLoc)
    model.avoidCircularInput(-1)
    model.alignCalibrationBallsWithInputAndReturnRedBall()
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.clearCalibrationDataClick_callback()
    model.ext.clearCalibration()
end

function model.dlg.trainCalibrationBallsClick_callback()
    if model.getAssociatedRobotHandle()==-1 then
        sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,'Calibration','The tracking window is not associated with any robot.')
    else
        local data={}
        data.idRobot=model.getAssociatedRobotHandle()
        data.id=model.handle
        local reply=simBWF.query('trackingWindow_trainStart',data)
        if reply=='ok' or simBWF.isInTestMode() then
            local xml =[[
                <group layout="hbox" flat="true">
                     <label text="Move the gripper platform to the red ball, then click 'Red' below" id="1"/>
                </group>
                <group layout="hbox" flat="true">
                    <button text="Red"  style="* {min-width: 150px; background-color: #ff8888}" on-click="model.dlg.calibrationDlg.calibrationBallClick_callback" id="2" />
                    <button text="Green"  style="* {min-width: 150px; background-color: #88ff88}" enabled="false" on-click="model.dlg.calibrationDlg.calibrationBallClick_callback" id="3" />
                    <button text="Blue"  style="* {min-width: 150px; background-color: #8888ff}" enabled="false" on-click="model.dlg.calibrationDlg.calibrationBallClick_callback" id="4" />
                </group>
            ]]
            model.dlg.calibrationDlg.data={}
            model.dlg.calibrationDlg.data.ui=simBWF.createCustomUi(xml,"Calibration","center",true,"model.dlg.calibrationDlg.onCloseDlg",true,false,true)
            model.dlg.calibrationDlg.data.relativeBallPositions={}
        end
    end
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        simUI.setTitle(model.dlg.ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General">
                <group layout="form" flat="false">

                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

                <label text="Offset (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.offsetChange_callback" id="24"/>

                <label text="Size (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.sizeChange_callback" id="21"/>

                <label text="Upstream margin (mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.upstreamMarginChange_callback" id="61"/>

                <label text="Input"/>
                <combobox id="11" on-change="model.dlg.inputChange_callback">
                </combobox>

                <label text="Calibration ball offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.calibrationBallOffsetChange_callback" id="23"/>
            </group>

            <group layout="form" flat="false">

                <checkbox text="Line control" style="* {background-color: #ccffcc}" on-change="model.dlg.startStopLine_callback" id="50" />
                <label text=""/>
                
                <label text="Part type" style="* {background-color: #ccffcc}"/>
                <combobox id="53" on-change="model.dlg.partTypeChange_callback">
                </combobox>

                <label text="Stop line" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.stopLineChange_callback" id="51"/>

                <label text="Start line" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.startLineChange_callback" id="52"/>
            </group>


            </tab>
            <tab title="Online">
                <button text="Train calibration balls"  style="* {min-width: 300px;}" on-click="model.dlg.trainCalibrationBallsClick_callback" id="100" />
                <button text="Clear calibration data"  style="* {min-width: 300px;}" on-click="model.dlg.clearCalibrationDataClick_callback" id="101" />
            </tab>
            <tab title="More">
            <group layout="form" flat="false">

                <label text="Hidden during simulation" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.hidden_callback" id="3" />

                 <label text="Calibration balls hidden during simulation" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.calibrationBallsHidden_callback" id="1" />


                <label text="Visualize tracked items" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.showPoints_callback" id="5" />

                <label text="Create parts (online mode)" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.createParts_callback" id="2" />

                <label text="" style="* {margin-left: 175px;}"/>
                <label text="" style="* {margin-left: 175px;}"/>
            </group>
            </tab>
       </tabs>
        ]]
        --[[
                 <label text="Calibration balls always hidden" />
                <checkbox text="" on-change="calibrationBallsAlwaysHidden_callback" id="2" />
--]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos,false,nil,false,false,false)


        model.dlg.refresh()
        simUI.setCurrentTab(model.dlg.ui,77,model.dlg.mainTabIndex,true)
    end
end

function model.dlg.showDlg()
    if not model.dlg.ui then
        model.dlg.createDlg()
    end
end

function model.dlg.removeDlg()
    if model.dlg.ui then
        local x,y=simUI.getPosition(model.dlg.ui)
        model.dlg.previousDlgPos={x,y}
        model.dlg.mainTabIndex=simUI.getCurrentTab(model.dlg.ui,77)
        simUI.destroy(model.dlg.ui)
        model.dlg.ui=nil
    end
end

function model.dlg.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model.handle then
        model.dlg.showDlg()
    else
        model.dlg.removeDlg()
    end
end

function model.dlg.init()
    model.dlg.mainTabIndex=0
    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
end
