model.packMl={}

if simBWF.getVrepVersion()>30500 then

function model.packMl.editCode(title,field)
    local s="800 600"
    local p="100 100"
    if not model.packMl.editorPositions then
        model.packMl.editorPositions={}
        model.packMl.editorSizes={}
        model.packMl.editorHandles={}
    end
    if model.packMl.editorSizes[field] then
        s=model.packMl.editorSizes[field][1]..' '..model.packMl.editorSizes[field][2]
    end
    if model.packMl.editorPositions[field] then
        p=model.packMl.editorPositions[field][1]..' '..model.packMl.editorPositions[field][2]
    end
    local modalDlg=true
    local xml = '<editor title="'..title..'" '
    xml = xml..[[editable="true" searchable="true"
            tab-width="4" text-col="50 50 50" background-col="190 190 190"
            selection-col="128 128 255" size="]]..s..[[" position="]]..p..[["
            is-lua="true" keyword1-col="152 0 0" keyword2-col="220 80 20" modal="]]..tostring(modalDlg)..[[" on-close="model.packMl.]]..field..[[_editor_close_callback">]]
--    xml=xml..
    local result,data=simBWF.query('packml_getfunctions',{})
    if result~='ok' then
        -- data={{name='my.testFunc1',calltip='string ret=my.testFunc1(number arg1)'},{name='my.testFunc2',calltip='string ret=my.testFunc2(number arg1)'},{name='my.variable1'}}
        data={  {name='pml.proceed',calltip='bool success=pml.proceed()'},
                {name='pml.write',calltip='bool success=pml.write(string text)'},
                {name='pml.initializeRobot',calltip='bool success=pml.initializeRobot(string robotName)'},
                {name='pml.initializeAllRobots',calltip='bool success=pml.initializeAllRobots()'},
                {name='pml.startRobotProcess',calltip='bool success=pml.startRobotProcess(string robotName)'},
                {name='pml.startAllRobotProcess',calltip='bool success=pml.startAllRobotProcess()'},
                {name='pml.stopRobotProcess',calltip='bool success=pml.stopRobotProcess(string robotName)'},
                {name='pml.stopAllRobotProcess',calltip='bool success=pml.stopAllRobotProcess()'},
				{name='pml.clearRobot',calltip='bool success=pml.clearRobot(string robotName)'},
                {name='pml.clearAllRobots',calltip='bool success=pml.clearAllRobots()'},
                {name='pml.addEvent',calltip='bool success=pml.addEvent(string eventName)\neventName={\'inputChange\', \'robotStateChange\', \'robotAborted\'}'},
                {name='pml.getIOChanges',calltip='table changes, bool success=pml.getIOChanges(string iohubName)'},
                {name='pml.getRobotStateChanges',calltip='table changes, bool success=pml.getRobotStateChanges(string robotName)'},
                {name='pml.checkRobotState',calltip='bool isInState, bool success=pml.checkRobotState(string robotName, string robotState)\nrobotState={\'HaltOn/Off\', \'BusyOn/Off\', \'PoweredOn/Off\', \'DryRunOn/Off\'}'},
                {name='pml.setIO', calltip='bool success=pml.setIO(string iohubName, string ioName, bool isInState)'},
                {name='pml.configureIO', calltip='bool success=pml.configureIO(string iohubName, string ioName, string option, optionValue)\noption={\'initState\', \'haltState\', \'onCodes\', \'offCodes\', \'onDelay\', \'offDelay\', }\noptionValue={bool, bool, string, string, integer, integer}'},
                {name='pml.getRobotInfo',calltip='string mimerSerialNo, string mimerFirmwareVersion,\nstring mimerCoprocessorVersion, bool success=pml.getRobotInfo(string robotName)'},
                {name='pml.getRobotPosition',calltip='int x, int y, int z, bool success=pml.getRobotPosition(string robotName)'},
                {name='pml.abort',calltip='bool success = pml.abort()'},
                {name='pml.stop',calltip='bool success = pml.stop()'},
                {name='pml.hold',calltip='bool success = pml.hold()'},
                {name='pml.suspend',calltip='bool success = pml.suspend()'},
                {name='pml.sendInfo',calltip='bool success = pml.sendInfo(string infoMessage)'},
                {name='pml.sendWarning',calltip='bool success = pml.sendWarning(string warningMessage)'},
                {name='pml.sendError',calltip='bool success = pml.sendError(string errorMessage)'},
                {name='pml.isFrameDone',calltip='bool isDone, bool success = pml.isFrameDone(string frameName)'},
                {name='pml.refillFrame',calltip='bool success = pml.refillFrame(string frameName)'},
                {name='pml.clearTrackingObjects',calltip='bool success = pml.clearTrackingObjects(string frameOrBeltName)'},
                {name='pml.clearAllTrackingObjects',calltip='bool success = pml.clearAllTrackingObjects()'},
                {name='pml.setOutputBoxState',calltip='bool success = pml.setOutputBoxState(string outputBoxName, bool state) \nbool success = pml.setOutputBoxState(string outputBoxName, float frequency)'},
                {name='pml.getOutputBoxState',calltip='bool state, bool success = pml.getOutputBoxState(string outputBoxName)'},
                {name='pml.getInputBoxChange',calltip='bool changeStatus, bool success = pml.getInputBoxChange(string inputBoxName, bool state)'},
                {name='pml.getInputBoxState',calltip='bool state, bool success = pml.getInputBoxState(string inputBoxName)'},
                {name='pml.generateObjects',calltip='bool success = pml.generateObjects(string sensorDigitalName)'},
                {name='pml.setSoftSignal',calltip='bool success = pml.setSoftSignal(string softSignalName, bool state)'},
                {name='pml.getSoftSignal',calltip='bool state, bool success = pml.getSoftSignal(string softSignalName)'},
                {name='pml.checkWindowStopStartLines',calltip='bool stop, bool start, bool success = pml.checkWindowStopStartLines(string windowName)\nbool stop, bool start, bool success = pml.checkWindowStopStartLines(string windowName, string partTypeName)'},
                {name='pml.addWindowToLineControl',calltip='bool success = pml.addWindowToLineControl(string outputBoxName, string windowName)\nbool success = pml.addWindowToLineControl(string outputBoxName, string windowName, string partTypeName)'},
                {name='pml.clearLineControl',calltip='bool success = pml.clearLineControl(string outputBoxName)'},
                {name='pml.activateLineControl',calltip='bool success = pml.activateLineControl(string outputBoxName)'},
				{name='pml.activateAllLineControls',calltip='bool success = pml.activateAllLineControls()'},
                {name='pml.deactivateLineControl',calltip='bool success = pml.deactivateLineControl(string outputBoxName)'},
                {name='pml.deactivateAllLineControls',calltip='bool success = pml.deactivateAllLineControls()'},
                {name="pml.setVariable",calltip='bool success = pml.setVariable(string variableName, value)'},
                {name="pml.getVariable",calltip='value, bool success = pml.getVariable(string variableName)'},
                {name="pml.clearVariable",calltip='bool success = pml.clearVariable(string variableName)'},
                {name="pml.clearAllVariables",calltip='bool success = pml.clearAllVariables()'}
            }
    end
    xml=xml..'<keywords1>'
    for i=1,#data,1 do
        local name=data[i].name
        local ct=data[i].calltip
        if name and ct then
            xml=xml..'<item word="'..name..'" autocomplete="true" calltip="'..ct..'"/>'
        end
    end
    xml=xml..'</keywords1>'
    xml=xml..'<keywords2>'
    for i=1,#data,1 do
        local name=data[i].name
        local ct=data[i].calltip
        if name and not ct then
            xml=xml..'<item word="'..name..'" autocomplete="true"/>'
        end
    end
    xml=xml..'</keywords2>'
    
    xml=xml..'</editor>'

    local c=model.readInfo()
    local initialText=c['packMlCode'][field]
    model.packMl.editorHandles[field]=sim.textEditorOpen(initialText,xml)
end

function model.packMl.aborting_editor_close_callback()
    model.packMl.handleEditorClose("aborting")
end

function model.packMl.aborted_editor_close_callback()
    model.packMl.handleEditorClose("aborted")
end

function model.packMl.clearing_editor_close_callback()
    model.packMl.handleEditorClose("clearing")
end

function model.packMl.stopping_editor_close_callback()
    model.packMl.handleEditorClose("stopping")
end

function model.packMl.stopped_editor_close_callback()
    model.packMl.handleEditorClose("stopped")
end

function model.packMl.suspending_editor_close_callback()
    model.packMl.handleEditorClose("suspending")
end

function model.packMl.suspended_editor_close_callback()
    model.packMl.handleEditorClose("suspended")
end

function model.packMl.unsuspending_editor_close_callback()
    model.packMl.handleEditorClose("unsuspending")
end

function model.packMl.resetting_editor_close_callback()
    model.packMl.handleEditorClose("resetting")
end

function model.packMl.complete_editor_close_callback()
    model.packMl.handleEditorClose("complete")
end

function model.packMl.completing_editor_close_callback()
    model.packMl.handleEditorClose("completing")
end

function model.packMl.execute_editor_close_callback()
    model.packMl.handleEditorClose("execute")
end

function model.packMl.starting_editor_close_callback()
    model.packMl.handleEditorClose("starting")
end

function model.packMl.idle_editor_close_callback()
    model.packMl.handleEditorClose("idle")
end

function model.packMl.holding_editor_close_callback()
    model.packMl.handleEditorClose("holding")
end

function model.packMl.hold_editor_close_callback()
    model.packMl.handleEditorClose("hold")
end

function model.packMl.unholding_editor_close_callback()
    model.packMl.handleEditorClose("unholding")
end

function model.packMl.handleEditorClose(field)
    local modifiedCode,pos,size=sim.textEditorClose(model.packMl.editorHandles[field])
    model.packMl.editorPositions[field]=pos
    model.packMl.editorSizes[field]=size
    local c=model.readInfo()
    local originalCode=c['packMlCode'][field]
    if modifiedCode~=originalCode then
        if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,"PackML Code Change","Do you want to apply the changes?") then
            c['packMlCode'][field]=modifiedCode
            model.writeInfo(c)
            simBWF.markUndoPoint()
            model.updatePluginRepresentation_brApp()
        end
    end
    model.packMl.editorHandles[field]=nil
end

else
------ START OLD CODE (<BR V3.6.0) ---------------------------------------------------------------------------------
    function model.packMl.editCode(title,field)
        -- OLD FUNCTION, use newer one further above
        local s="800 600"
        local p="100 100"
        if not model.packMl.editorPositions then
            model.packMl.editorPositions={}
            model.packMl.editorSizes={}
            model.packMl.editorHandles={}
        end
        if model.packMl.editorSizes[field] then
            s=model.packMl.editorSizes[field][1]..' '..model.packMl.editorSizes[field][2]
        end
        if model.packMl.editorPositions[field] then
            p=model.packMl.editorPositions[field][1]..' '..model.packMl.editorPositions[field][2]
        end
        local xml = '<editor title="'..title..'" '
        xml = xml..[[editable="true" searchable="true"
                tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190"
                selectionColor="128 128 255" size="]]..s..[[" position="]]..p..[["
                useVrepKeywords="false" isLua="true">]]
        local result,data=simBWF.query('packml_getfunctions',{})
        if result~='ok' then
            data={  {name='pml.proceed',calltip='bool success=pml.proceed()'},
                    {name='pml.write',calltip='bool success=pml.write(string text)'},
                    {name='pml.initializeRobot',calltip='bool success=pml.initializeRobot(string robotName)'},
                    {name='pml.initializeAllRobots',calltip='bool success=pml.initializeAllRobots()'},
                    {name='pml.startRobotProcess',calltip='bool success=pml.startRobotProcess(string robotName)'},
                    {name='pml.startAllRobotProcess',calltip='bool success=pml.startAllRobotProcess()'},
                    {name='pml.stopRobotProcess',calltip='bool success=pml.stopRobotProcess(string robotName)'},
                    {name='pml.stopAllRobotProcess',calltip='bool success=pml.stopAllRobotProcess()'},
                    {name='pml.clearRobot',calltip='bool success=pml.clearRobot(string robotName)'},
                    {name='pml.clearAllRobots',calltip='bool success=pml.clearAllRobots()'},
                    {name='pml.addEvent',calltip='bool success=pml.addEvent(string eventName)\neventName={\'inputChange\', \'robotStateChange\', \'robotAborted\'}'},
                    {name='pml.getIOChanges',calltip='table changes, bool success=pml.getIOChanges(string iohubName)'},
                    {name='pml.getRobotStateChanges',calltip='table changes, bool success=pml.getRobotStateChanges(string robotName)'},
                    {name='pml.checkRobotState',calltip='bool isInState, bool success=pml.checkRobotState(string robotName, string robotState)\nrobotState={\'HaltOn/Off\', \'BusyOn/Off\', \'PoweredOn/Off\', \'DryRunOn/Off\'}'},
                    {name='pml.setIO', calltip='bool success=pml.setIO(string iohubName, string ioName, bool isInState)'},
                    {name='pml.configureIO', calltip='bool success=pml.configureIO(string iohubName, string ioName, string option, optionValue)\noption={\'initState\', \'haltState\', \'onCodes\', \'offCodes\', \'onDelay\', \'offDelay\', }\noptionValue={bool, bool, string, string, integer, integer}'},
                    {name='pml.getRobotInfo',calltip='string mimerSerialNo, string mimerFirmwareVersion,\nstring mimerCoprocessorVersion, bool success=pml.getRobotInfo(string robotName)'},
                    {name='pml.getRobotPosition',calltip='int x, int y, int z, bool success=pml.getRobotPosition(string robotName)'},
                    {name='pml.abort',calltip='bool success = pml.abort()'},
                    {name='pml.stop',calltip='bool success = pml.stop()'},
                    {name='pml.hold',calltip='bool success = pml.hold()'},
                    {name='pml.suspend',calltip='bool success = pml.suspend()'},
                    {name='pml.sendInfo',calltip='bool success = pml.sendInfo(string infoMessage)'},
                    {name='pml.sendWarning',calltip='bool success = pml.sendWarning(string warningMessage)'},
                    {name='pml.sendError',calltip='bool success = pml.sendError(string errorMessage)'},
                    {name='pml.isFrameDone',calltip='bool isDone, bool success = pml.isFrameDone(string frameName)'},
                    {name='pml.refillFrame',calltip='bool success = pml.refillFrame(string frameName)'},
                    {name='pml.clearTrackingObjects',calltip='bool success = pml.clearTrackingObjects(string frameOrBeltName)'},
                    {name='pml.clearAllTrackingObjects',calltip='bool success = pml.clearAllTrackingObjects()'},
                    {name='pml.setOutputBoxState',calltip='bool success = pml.setOutputBoxState(string outputBoxName, bool state) \nbool success = pml.setOutputBoxState(string outputBoxName, float frequency)'},
                    {name='pml.getOutputBoxState',calltip='bool state, bool success = pml.getOutputBoxState(string outputBoxName)'},
                    {name='pml.getInputBoxChange',calltip='bool changeStatus, bool success = pml.getInputBoxChange(string inputBoxName, bool state)'},
                    {name='pml.getInputBoxState',calltip='bool state, bool success = pml.getInputBoxState(string inputBoxName)'},
                    {name='pml.generateObjects',calltip='bool success = pml.generateObjects(string sensorDigitalName)'},
                    {name='pml.setSoftSignal',calltip='bool success = pml.setSoftSignal(string softSignalName, bool state)'},
                    {name='pml.getSoftSignal',calltip='bool state, bool success = pml.getSoftSignal(string softSignalName)'},
                    {name='pml.checkWindowStopStartLines',calltip='bool stop, bool start, bool success = pml.checkWindowStopStartLines(string windowName)\nbool stop, bool start, bool success = pml.checkWindowStopStartLines(string windowName, string partTypeName)'},
                    {name='pml.addWindowToLineControl',calltip='bool success = pml.addWindowToLineControl(string outputBoxName, string windowName)\nbool success = pml.addWindowToLineControl(string outputBoxName, string windowName, string partTypeName)'},
                    {name='pml.clearLineControl',calltip='bool success = pml.clearLineControl(string outputBoxName)'},
                    {name='pml.activateLineControl',calltip='bool success = pml.activateLineControl(string outputBoxName)'},
                    {name='pml.activateAllLineControls',calltip='bool success = pml.activateAllLineControls()'},
                    {name='pml.deactivateLineControl',calltip='bool success = pml.deactivateLineControl(string outputBoxName)'},
                    {name='pml.deactivateAllLineControls',calltip='bool success = pml.deactivateAllLineControls()'},
                    {name="pml.setVariable",calltip='bool success = pml.setVariable(string variableName, value)'},
                    {name="pml.getVariable",calltip='value, bool success = pml.getVariable(string variableName)'},
                    {name="pml.clearVariable",calltip='bool success = pml.clearVariable(string variableName)'},
                    {name="pml.clearAllVariables",calltip='bool success = pml.clearAllVariables()'}
                }
        end
        xml=xml..'<keywords1 color="152 0 0">'
        for i=1,#data,1 do
            local name=data[i].name
            local ct=data[i].calltip
            if name and ct then
                xml=xml..'<item word="'..name..'" autocomplete="true" calltip="'..ct..'"/>'
            end
        end
        xml=xml..'</keywords1>'
        xml=xml..'<keywords2 color="220 80 20">'
        for i=1,#data,1 do
            local name=data[i].name
            local ct=data[i].calltip
            if name and not ct then
                xml=xml..'<item word="'..name..'" autocomplete="true"/>'
            end
        end
        xml=xml..'</keywords2>'
        
        xml=xml..'</editor>'

        local c=model.readInfo()
        local initialText=c['packMlCode'][field]
        if simBWF.getVrepVersion()>30500 or (simBWF.getVrepVersion()==30500 and simBWF.getVrepRevision()>=15) then
            -- NON-MODAL OPERATION:
            model.packMl.editorHandles[field]=sim.openTextEditor(initialText,xml,'model.packMl.'..field..'_editor_close_callback')
        else
            -- MODAL OPERATION:
            local modifiedText
            modifiedText,model.packMl.packMlCodeDlgSize,model.packMl.packMlCodeDlgPos=sim.openTextEditor(initialText,xml)
            if modifiedText~=initialText then
                if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,"PackML Code Change","Do you want to apply the changes?") then
                    c['packMlCode'][field]=modifiedText
                    model.writeInfo(c)
                    simBWF.markUndoPoint()
                    model.updatePluginRepresentation_brApp()
                end
            end
        end
    end
    function model.packMl.aborting_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("aborting",code,pos,size)
    end
    function model.packMl.aborted_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("aborted",code,pos,size)
    end
    function model.packMl.clearing_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("clearing",code,pos,size)
    end
    function model.packMl.stopping_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("stopping",code,pos,size)
    end
    function model.packMl.stopped_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("stopped",code,pos,size)
    end
    function model.packMl.suspending_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("suspending",code,pos,size)
    end
    function model.packMl.suspended_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("suspended",code,pos,size)
    end
    function model.packMl.unsuspending_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("unsuspending",code,pos,size)
    end
    function model.packMl.resetting_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("resetting",code,pos,size)
    end
    function model.packMl.complete_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("complete",code,pos,size)
    end
    function model.packMl.completing_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("completing",code,pos,size)
    end
    function model.packMl.execute_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("execute",code,pos,size)
    end
    function model.packMl.starting_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("starting",code,pos,size)
    end
    function model.packMl.idle_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("idle",code,pos,size)
    end
    function model.packMl.holding_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("holding",code,pos,size)
    end
    function model.packMl.hold_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("hold",code,pos,size)
    end
    function model.packMl.unholding_editor_close_callback(code,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.handleEditorClose("unholding",code,pos,size)
    end
    function model.packMl.handleEditorClose(field,modifiedCode,pos,size)
        -- OLD FUNCTION, use newer one further above
        model.packMl.editorPositions[field]=pos
        model.packMl.editorSizes[field]=size
        local c=model.readInfo()
        local originalCode=c['packMlCode'][field]
        if modifiedCode~=originalCode then
            if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,"PackML Code Change","Do you want to apply the changes?") then
                c['packMlCode'][field]=modifiedCode
                model.writeInfo(c)
                simBWF.markUndoPoint()
                model.updatePluginRepresentation_brApp()
            end
        end
        model.packMl.editorHandles[field]=nil
    end
end -- END OLD CODE (<BR V3.6.0)----------------------------------------------------------------------------------

function model.packMl.aborting_callback()
    model.packMl.editCode("PackML 'Aborting' Code","aborting")
end

function model.packMl.aborted_callback()
    model.packMl.editCode("PackML 'Aborted' Code","aborted")
end

function model.packMl.clearing_callback()
    model.packMl.editCode("PackML 'Clearing' Code","clearing")
end

function model.packMl.stopping_callback()
    model.packMl.editCode("PackML 'Stopping' Code","stopping")
end

function model.packMl.stopped_callback()
    model.packMl.editCode("PackML 'Stopped' Code","stopped")
end

function model.packMl.suspending_callback()
    model.packMl.editCode("PackML 'Suspending' Code","suspending")
end

function model.packMl.suspended_callback()
    model.packMl.editCode("PackML 'Suspended' Code","suspended")
end

function model.packMl.unsuspending_callback()
    model.packMl.editCode("PackML 'Un-Suspending' Code","unsuspending")
end

function model.packMl.resetting_callback()
    model.packMl.editCode("PackML 'Resetting' Code","resetting")
end

function model.packMl.complete_callback()
    model.packMl.editCode("PackML 'Complete' Code","complete")
end

function model.packMl.completing_callback()
    model.packMl.editCode("PackML 'Completing' Code","completing")
end

function model.packMl.execute_callback()
    model.packMl.editCode("PackML 'Execute' Code","execute")
end

function model.packMl.starting_callback()
    model.packMl.editCode("PackML 'Starting' Code","starting")
end

function model.packMl.idle_callback()
    model.packMl.editCode("PackML 'Idle' Code","idle")
end

function model.packMl.holding_callback()
    model.packMl.editCode("PackML 'Holding' Code","holding")
end

function model.packMl.hold_callback()
    model.packMl.editCode("PackML 'Hold' Code","hold")
end

function model.packMl.unholding_callback()
    model.packMl.editCode("PackML 'Un-Holding' Code","unholding")
end


function model.packMl.onClose()
--    local x,y=simUI.getPosition(model.packMl.ui)
--    model.packMl.previousDlgPos={x,y}
    simUI.destroy(model.packMl.ui)
    model.packMl.ui=nil
end

function model.packMl.createDlg()
    local xml =[[
            <image geometry="0,0,1088,607" width="1088" height="607" id="1000"/>

            <button text="Aborting" geometry="932,502,100,40" on-click="model.packMl.aborting_callback" id="1" style="* {background-color: #66ff66}"/>
            <button text="Aborted" geometry="715,502,100,40" on-click="model.packMl.aborted_callback" id="2" style="* {background-color: #ffff66}"/>
            <button text="Clearing" geometry="497,502,100,40" on-click="model.packMl.clearing_callback" id="3" style="* {background-color: #66ff66}"/>
            <button text="Stopping" geometry="279,502,100,40" on-click="model.packMl.stopping_callback" id="4" style="* {background-color: #66ff66}"/>
            <button text="Stopped" geometry="61,502,100,40" on-click="model.packMl.stopped_callback" id="5" style="* {background-color: #ffff66}"/>

            <button text="Suspending" geometry="715,308,100,40" on-click="model.packMl.suspending_callback" id="6" style="* {background-color: #66ff66}"/>
            <button text="Suspended" geometry="497,308,100,40" on-click="model.packMl.suspended_callback" id="7" style="* {background-color: #ffff66}"/>
            <button text="Un-Suspending" geometry="279,308,100,40" on-click="model.packMl.unsuspending_callback" id="8" style="* {background-color: #66ff66}"/>
            <button text="Resetting" geometry="61,308,100,40" on-click="model.packMl.resetting_callback" id="9" style="* {background-color: #66ff66}"/>

            <button text="Complete" geometry="932,186,100,40" on-click="model.packMl.complete_callback" id="10" style="* {background-color: #ffff66}"/>
            <button text="Completing" geometry="715,186,100,40" on-click="model.packMl.completing_callback" id="11" style="* {background-color: #66ff66}"/>
            <button text="Execute" geometry="497,186,100,40" on-click="model.packMl.execute_callback" id="12" style="* {background-color: #66aaff}"/>
            <button text="Starting" geometry="279,186,100,40" on-click="model.packMl.starting_callback" id="13" style="* {background-color: #66ff66}"/>
            <button text="Idle" geometry="61,186,100,40" on-click="model.packMl.idle_callback" id="14" style="* {background-color: #ffff66}"/>

            <button text="Holding" geometry="715,65,100,40" on-click="model.packMl.holding_callback" id="15" style="* {background-color: #66ff66}"/>
            <button text="Hold" geometry="497,65,100,40" on-click="model.packMl.hold_callback" id="16" style="* {background-color: #ffff66}"/>
            <button text="Un-Holding" geometry="279,65,100,40" on-click="model.packMl.unholding_callback" id="17" style="* {background-color: #66ff66}"/>
            ]]
    model.packMl.ui=simBWF.createCustomUi(xml,'PackML',model.packMl.previousDlgPos,true,'model.packMl.onClose',true,false,false,'layout="none"',{1088,607})
    print("creating packml UI")
    local img=sim.loadImage(0,sim.getStringParameter(sim.stringparam_application_path).."/BlueWorkforce/resources/packML-edit.png")
    simUI.setImageData(model.packMl.ui,1000,img,1088,607)
end

