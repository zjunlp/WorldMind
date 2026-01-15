model.dlg={}
model.dlg.calibrationDlg={}

function model.dlg.calibrationDlg.onCloseDlg()
    local data={}
    data.idRobot=model.getAssociatedRobotHandle()
    data.id=model.handle
    simBWF.query('locationFrame_trainEnd',data)
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
        local reply,replyData=simBWF.query('locationFrame_train',data)
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
            local reply,replyData=simBWF.query('locationFrame_train',data)
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
            local reply,replyData=simBWF.query('locationFrame_train',data)
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
                        simBWF.query('locationFrame_trainEnd',data)
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
        simBWF.query('locationFrame_trainEnd',data)
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Calibration",weHaveAProblem)
        simUI.destroy(model.dlg.calibrationDlg.data.ui)
        model.dlg.calibrationDlg.data=nil
    end
end

-------------------------------------------------------
-------------------------------------------------------

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        local config=model.readInfo()
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,5,simStopped,true) -- simBWF.getReferencedObjectHandle(model,model.objRefIdx.PALLET)>=0,true)
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setCheckboxValue(model.dlg.ui,1,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],16)~=0),true)

        local pallets=simBWF.getAvailablePallets()
        local refPallet=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PALLET)
        local selected=simBWF.NONE_TEXT
        for i=1,#pallets,1 do
            if pallets[i][2]==refPallet then
                selected=pallets[i][1]
                break
            end
        end
        comboPallet=simBWF.populateCombobox(model.dlg.ui,2,pallets,{},selected,true,{{simBWF.NONE_TEXT,-1}})
        
        local refreshModeComboItems={
            {"Lua",0},
            {"Automatic",1},
            {"Vision",2}
        }
        simBWF.populateCombobox(model.dlg.ui,200,refreshModeComboItems,{},refreshModeComboItems[config['refreshMode']+1][1],false,nil)
        
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

function model.dlg.palletChange_callback(ui,id,newIndex)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PALLET,comboPallet[newIndex+1][2])
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updateEnabledDisabledItems()
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

function model.dlg.showPallet_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.refreshModeChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['refreshMode']=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.clearCalibrationDataClick_callback()
    model.ext.clearCalibration()
end

function model.dlg.trainCalibrationBallsClick_callback()
    if model.getAssociatedRobotHandle()==-1 then
        sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_ok,'Calibration','The location frame is not associated with any robot.')
    else
        local data={}
        data.idRobot=model.getAssociatedRobotHandle()
        data.id=model.handle
        local reply=simBWF.query('locationFrame_trainStart',data)
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
            model.dlg.calibrationDlg.data.relativeBallPositions={}
            model.dlg.calibrationDlg.data.ui=simBWF.createCustomUi(xml,"Calibration","center",true,"model.dlg.calibrationDlg.onCloseDlg",true,false,true)
        end
    end
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
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
                
                <label text="Refresh mode" style="* {background-color: #ccffcc}"/>
                <combobox id="200" on-change="model.dlg.refreshModeChange_callback"></combobox>
                
                <label text="Associated pallet" style="* {background-color: #ccffcc}"/>
                <combobox id="2" on-change="model.dlg.palletChange_callback"/>
            </group>
            </tab>
            <tab title="Online">
                <button text="Train calibration balls"  style="* {min-width: 300px;}" on-click="model.dlg.trainCalibrationBallsClick_callback" id="100" />
                <button text="Clear calibration data"  style="* {min-width: 300px;}" on-click="model.dlg.clearCalibrationDataClick_callback" id="101" />
            </tab>
            <tab title="More">
            <group layout="form" flat="false">
                 <label text="Hidden during simulation" />
                <checkbox text="" on-change="model.dlg.hidden_callback" id="1" />

                 <label text="Calibration balls hidden during simulation" />
                <checkbox text="" on-change="model.dlg.calibrationBallsHidden_callback" id="3" />
                
                <label text="Show associated pallet" />
                <checkbox text="" on-change="model.dlg.showPallet_callback" id="5" />

                <label text="Create parts (online mode)"/>
                <checkbox text="" on-change="model.dlg.createParts_callback" id="4" />
            </group>
            </tab>
       </tabs>
        ]]

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
