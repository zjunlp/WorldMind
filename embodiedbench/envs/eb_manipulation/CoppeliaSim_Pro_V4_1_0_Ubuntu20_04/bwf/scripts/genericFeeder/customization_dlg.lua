model.dlg={}

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.size
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0 then t[i]=0 end
            if t[i]>2 then t[i]=2 end
        end
        if i==2 then
            if t[i]<0 then t[i]=0 end
            if t[i]>2 then t[i]=2 end
        end
        if i==3 then
            if t[i]<0 then t[i]=0 end
            if t[i]>2 then t[i]=2 end
        end
        i=i+1
    end
    c.size=t
    model.writeInfo(c)
    model.setModelSize()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.frequencyChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>10 then v=10 end
        if v~=c['frequency'] then
            simBWF.markUndoPoint()
            c['frequency']=v
            model.writeInfo(c)
        end
    end
    simUI.setEditValue(ui,2,simBWF.format("%.2f",c['frequency']),true)
end

function model.dlg.conveyorDistanceChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>2 then v=2 end
        if v~=c['conveyorDist'] then
            simBWF.markUndoPoint()
            c['conveyorDist']=v
            model.writeInfo(c)
        end
    end
    simUI.setEditValue(ui,62,simBWF.format("%.0f",c['conveyorDist']/0.001),true)
end

function model.dlg.maxProductionCntChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0 then v=0 end
        if v>1000000000 then v=1000000000 end
        v=math.floor(v)
        if v~=c['maxProductionCnt'] then
            c['maxProductionCnt']=v
            model.writeInfo(c)
            simBWF.markUndoPoint()
        end
    else
        if c['maxProductionCnt']~=0 then
            c['maxProductionCnt']=0
            model.writeInfo(c)
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.sizeScalingClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['sizeScaling']=id-70
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.dropAlgorithmClick_callback()

    local s="800 600"
    local p="100 100"
    if model.dlg.algoDlgSize then
        s=model.dlg.algoDlgSize[1]..' '..model.dlg.algoDlgSize[2]
    end
    if model.dlg.algoDlgPos then
        p=model.dlg.algoDlgPos[1]..' '..model.dlg.algoDlgPos[2]
    end
    local xml = [[
        <editor title="Feeder Algorithm" editable="true" searchable="true"
            tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190"
            selectionColor="128 128 255" size="]]..s..[[" position="]]..p..[["
            useVrepKeywords="true" isLua="true">
            <keywords1 color="152 0 0" >
            </keywords1>
            <keywords2 color="220 80 20" >
            </keywords2>
        </editor>
    ]]

    local c=model.readInfo()
    local initialText=c['algorithm']
    local modifiedText
    modifiedText,model.dlg.algoDlgSize,model.dlg.algoDlgPos=sim.openTextEditor(initialText,xml)
    c['algorithm']=modifiedText
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.sensorComboChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboSensor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.SENSOR,sens)
    simBWF.markUndoPoint()
end

function model.dlg.conveyorComboChange_callback(ui,id,newIndex)
    local conv=model.dlg.comboConveyor[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR,conv)
    simBWF.markUndoPoint()
end

function model.dlg.loadTheString()
    local f=loadstring("local counter=0\n".."return "..model.dlg.theString)
    return f
end

function model.dlg.distributionDlg(title,fieldName,tempComment)
    local prop=model.readInfo()
    local s="800 400"
    local p="200 200"
    if model.dlg.distributionDlgSize then
        s=model.dlg.distributionDlgSize[1]..' '..model.dlg.distributionDlgSize[2]
    end
    if model.dlg.distributionDlgPos then
        p=model.dlg.distributionDlgPos[1]..' '..model.dlg.distributionDlgPos[2]
    end
    local xml = [[ <editor title="]]..title..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            



    local initialText=prop[fieldName]
    if tempComment then
        initialText=initialText.."--[[tmpRem\n\n"..tempComment.."\n\n--]]"
    end
    local modifiedText
    modifiedText,model.dlg.distributionDlgSize,model.dlg.distributionDlgPos=sim.openTextEditor(initialText,xml)
    model.dlg.theString='{'..modifiedText..'}' -- variable needs to be global here
    local bla,f=pcall(model.dlg.loadTheString)
    local success=false
    if f then
        local res,err=xpcall(f,function(err) return debug.traceback(err) end)
        if res then
            modifiedText=simBWF.removeTmpRem(modifiedText)
            prop[fieldName]=modifiedText
            simBWF.markUndoPoint()
            model.writeInfo(prop)
            success=true
        end
    end
    if not success then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Input Error',"The distribution string is ill-formated.")
    end
end

function model.dlg.partDistribution_callback(ui,id,newVal)
    local parts=simBWF.getAllPartsFromPartRepository()
    local tmpTxt="a) There are currently no parts available in the part repository.\n\n"
    if parts and #parts>0 then
        tmpTxt="a) Following parts are currently available in the part repository:"
        local lcnt=0
        for i=1,#parts,1 do
            if lcnt~=0 then
                tmpTxt=tmpTxt..", "
            end
            if lcnt==4 then
                lcnt=0
            end
            if lcnt==0 then
                tmpTxt=tmpTxt.."\n   "
            end
            tmpTxt=tmpTxt.."'"..parts[i][1].."'"
            lcnt=lcnt+1
        end
        tmpTxt=tmpTxt.."\n\n"
    end
    tmpTxt=tmpTxt..[[
b) Usage:
   {partialProportion1,<partName>},{partialProportion2,<partName>}

c) Example:
   {2.1,'BOX1'},{1.5,'BOX2'} --> out of (2.1+1.5) parts, 2.1 will be 'BOX1', and 1.5 will be 'BOX2' (statistically)

d) You may use the variable 'counter' as in following example:
   {counter%2,'BOX1'},{(counter+1)%2,'BOX2'} --> parts alternate between 'BOX1' and 'BOX2']]
    model.dlg.distributionDlg('Part Distribution','partDistribution',tmpTxt)
end
--[=[
function model.dlg.destinationDistribution_callback(ui,id,newVal)
    local destinations=simBWF.getAllPossiblePartDestinations()
    local tmpTxt="a) There are currently no destinations available.\n\n"
    if #destinations>0 then
        tmpTxt="a) Following destinations are currently available in this scene:"
        local lcnt=0
        for i=1,#destinations,1 do
            if lcnt~=0 then
                tmpTxt=tmpTxt..", "
            end
            if lcnt==4 then
                lcnt=0
            end
            if lcnt==0 then
                tmpTxt=tmpTxt.."\n   "
            end
            tmpTxt=tmpTxt.."'"..destinations[i].."'"
            lcnt=lcnt+1
        end
        tmpTxt=tmpTxt.."\n\n"
    end
    tmpTxt=tmpTxt..[[
b) Usage:
   {partialProportion1,<destinationName>},{partialProportion2,<destinationName>}

c) Example:
   {2.1,'TRAY'},{1.5,'PLATE'} --> out of (2.1+1.5) destinations, 2.1 will be 'TRAY', and 1.5 will be 'PLATE' (statistically)

d) You may use the variable 'counter' as in following example:
   {counter%2,'TRAY'},{(counter+1)%2,'PLATE'} --> destinations alternate between 'TRAY' and 'PLATE'
   
e) You may use '<DEFAULT>', to refer to the default destination for a given part, as in following example:
   {1,'<DEFAULT>'},{1,'TRAY'} --> out of 2 destinations, one will be the default (for the part), the other one will be 'TRAY' (statistically)]]
   
    model.dlg.distributionDlg('Destination Distribution','destinationDistribution',tmpTxt)
end
--]=]
function model.dlg.shiftDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<proportionalShiftVector>},{partialProportion2,<proportionalShiftVector>}, where <proportionalShiftVector> is {shiftAlongX,shiftAlongY,shiftAlongZ} (im meters)

b) Example:
   {2.1,{0,0,0}},{1.5,{1,0,0}} --> out of (2.1+1.5) items, 2.1 will be shifted by {0,0,0}*feederDimension, and 1.5 will be shifted by {1,0,0}*feederDimension (statistically)

c) You may use math expressions, as in following example:
   {1,{math.random(),math.random(),math.random()}} --> items are random 3D vectors

d) You may use the variable 'counter' as in following example:
   {counter%2,{0,0,0}},{(counter+1)%2,{1,0,0}} --> items alternate between {0,0,0} and {1,0,0}]]
    model.dlg.distributionDlg('Relative Shift Distribution','shiftDistribution',tmpTxt)
end

function model.dlg.labelDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<bitCodedValue>},{partialProportion2,<bitCodedValue>}, where <bitCodedValue>:
    1 is for the first label (usually large top label), 2 and 4 is for the 2nd and 3rd labels (usually small side labels)

b) Example:
   {2.1,0},{1.5,1+2+4} --> out of (2.1+1.5) items, 2.1 will have no labels, and 1.5 will have 3 labels (if the part is configured to have labels)

c) You may use math expressions, as in following example:
   {1,math.floor(math.random()+0.5)} --> random items will have a top label

d) You may use the variable 'counter' as in following example:
   {counter%2,0},{(counter+1)%2,1+2+4} --> items will alternatively have no labels, or 3 labels]]
    model.dlg.distributionDlg('Label Count Distribution','labelDistribution',tmpTxt)
end

function model.dlg.rotationDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<eulerAngles>},{partialProportion2,<eulerAngles>}, where <eulerAngles> is {rotationAngleAroundXAxis,rotationAngleAroundYAxis,rotationAngleAroundZAxis} (in radians)

b) Example:
   {2.1,{0,0,0}},{1.5,{0,0,math.pi/2}} --> out of (2.1+1.5) items, 2.1 will not be rotated, and 1.5 will be rotated by 90 degrees around their Z axis (statistically)

c) You may use math expressions, as in following example:
   {1,{0,0,math.random()*2*math.pi}} --> items are randomly rotated around their Z axis

d) You may use the variable 'counter' as in following example:
   {counter%2,{0,0,0}},{(counter+1)%2,{0,0,math.pi/2}} --> items alternate between no rotation, and rotation of 90 degrees around their Z axis]]
    model.dlg.distributionDlg('Rotation Distribution','rotationDistribution',tmpTxt)
end

function model.dlg.weightDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<mass>},{partialProportion2,<mass>} (mass in Kg)

b) Example:
   {2.1,0.1},{1.5,0.2} --> out of (2.1+1.5) items, 2.1 will have a mass of 0.1Kg, and 1.5 will have a mass of 0.2Kg (statistically)

c) You may use math expressions, as in following example:
   {1,0.1+0.9*math.random()} --> items have a random mass between 0.1Kg and 1Kg

d) You may use the variable 'counter' as in following example:
   {counter%2,0.1},{(counter+1)%2,0.2} --> item masses alternate between 0.1Kg and 0.2Kg
   
e) You may use '<DEFAULT>', to refer to the default value, as in following example:
   {1,'<DEFAULT>'},{1,0.1} --> out of 2 items, one will have its default mass, the other one will have a mass of 0.1Kg (statistically)]]
    model.dlg.distributionDlg('Weight Distribution','weightDistribution',tmpTxt)
end

function model.dlg.isoSizeScalingDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Usage:
   {partialProportion1,<scaling>},{partialProportion2,<scaling>} (valid scaling values are between 0.1 and 10)

b) Example:
   {2.1,0.5},{1.5,1} --> out of (2.1+1.5) items, 2.1 will have a size scaled down by 50%, and 1.5 will have a default size (statistically)

c) You may use math expressions, as in following example:
   {1,0.1+0.9*math.random()} --> items have a random scaling between 0.1 and 1

d) You may use the variable 'counter' as in following example:
   {counter%2,0.5},{(counter+1)%2,1} --> every other item is scaled by 50%]]
    model.dlg.distributionDlg('Isometric Size Scaling Distribution','isoSizeScalingDistribution',tmpTxt)
end

function model.dlg.nonIsoSizeScalingDistribution_callback(ui,id,newVal)
    local tmpTxt=[[
a) Not all parts can be scaled non-isometrically!

b) Usage:
   {partialProportion1,<scaling>},{partialProportion2,<scaling>}, where <scaling> is {xScaling,yScaling,zScaling} (valid values are between 0.1 and 10)

c) Example:
   {2.1,{1,1,2}},{1.5,{1,1,1}} --> out of (2.1+1.5) items, 2.1 will be scaled by a factor 2 along the Z-axis (statistically)

d) You may use math expressions, as in following example:
   {1,{1,1,0.1+0.9*math.random()}} --> items have a random Z-axis scaling between 0.1 and 1

e) You may use the variable 'counter' as in following example:
   {counter%2,{1,1,0.5}},{(counter+1)%2,{1,1,1}} --> every other item is scaled by 50% along its Z-axis]]
    model.dlg.distributionDlg('Non-isometric Size Scaling Distribution','nonIsoSizeScalingDistribution',tmpTxt)
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        local config=model.readInfo()
        local freq=sim.boolAnd32(config['bitCoded'],4+8+16)==0
        local sens=sim.boolAnd32(config['bitCoded'],4+8+16)==4
        local user=sim.boolAnd32(config['bitCoded'],4+8+16)==8
        local conv=sim.boolAnd32(config['bitCoded'],4+8+16)==12
        simUI.setEnabled(model.dlg.ui,1365,enabled,true)
        simUI.setEnabled(model.dlg.ui,60,enabled,true)
--        simUI.setEnabled(model.dlg.ui,49,enabled,true)
        simUI.setEnabled(model.dlg.ui,2,enabled and freq,true)
        simUI.setEnabled(model.dlg.ui,999,enabled and sens,true)
        simUI.setEnabled(model.dlg.ui,3,enabled and user,true)
        simUI.setEnabled(model.dlg.ui,998,enabled and conv,true)
        simUI.setEnabled(model.dlg.ui,62,enabled and conv,true)
        simUI.setEnabled(model.dlg.ui,63,enabled)
        simUI.setEnabled(model.dlg.ui,1000,enabled,true)
        simUI.setEnabled(model.dlg.ui,1001,enabled,true)
        simUI.setEnabled(model.dlg.ui,1002,enabled,true)
        simUI.setEnabled(model.dlg.ui,1003,enabled,true)
        simUI.setEnabled(model.dlg.ui,1004,enabled,true)
        simUI.setEnabled(model.dlg.ui,1005,enabled,true)
        simUI.setEnabled(model.dlg.ui,11,enabled,true)
        simUI.setEnabled(model.dlg.ui,12,enabled,true)
        simUI.setEnabled(model.dlg.ui,13,enabled,true)
        simUI.setEnabled(model.dlg.ui,14,enabled,true)
        simUI.setEnabled(model.dlg.ui,30,enabled,true)
      --  simUI.setEnabled(model.dlg.ui,40,enabled,true)
        simUI.setEnabled(model.dlg.ui,50,enabled,true)
        simUI.setEnabled(model.dlg.ui,17,enabled,true)
        simUI.setEnabled(model.dlg.ui,73,config['sizeScaling']==1,true)
        simUI.setEnabled(model.dlg.ui,74,config['sizeScaling']==2,true)

        simUI.setEnabled(model.dlg.ui,54,enabled,true) -- config tab

        simUI.setEnabled(model.dlg.ui,100,enabled,true) -- stop trigger
        simUI.setEnabled(model.dlg.ui,101,enabled,true) -- restart trigger
    end
end

function model.dlg.refresh()
    if model.dlg.ui then
    
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        local config=model.readInfo()
        local loc=model.getAvailableConveyors()
        model.dlg.comboConveyor=simBWF.populateCombobox(model.dlg.ui,998,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR)),true,{{simBWF.NONE_TEXT,-1}})
        loc=model.getAvailableSensors()
        model.dlg.comboSensor=simBWF.populateCombobox(model.dlg.ui,999,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.SENSOR)),true,{{simBWF.NONE_TEXT,-1}})

        model.dlg.updateStartStopTriggerComboboxes()
        
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,60,simBWF.format("%.0f , %.0f , %.0f",config.size[1]*1000,config.size[2]*1000,config.size[3]*1000),true)
        simUI.setEditValue(model.dlg.ui,62,simBWF.format("%.0f",config['conveyorDist']/0.001),true)
        simUI.setEditValue(model.dlg.ui,2,simBWF.format("%.2f",config['frequency']),true)
        if config['maxProductionCnt']==0 then
            simUI.setEditValue(model.dlg.ui,63,"unlimited",true)
        else
            simUI.setEditValue(model.dlg.ui,63,simBWF.format("%.0f",config['maxProductionCnt']),true)
        end
        simUI.setCheckboxValue(model.dlg.ui,30,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,40,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],2)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,50,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],128)~=0),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1000,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==0),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1001,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==4),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1002,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==8),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1003,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==12),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1004,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==16),true)
        simUI.setRadiobuttonValue(model.dlg.ui,1005,simBWF.getRadiobuttonValFromBool(sim.boolAnd32(config['bitCoded'],4+8+16)==20),true)
        
        simUI.setRadiobuttonValue(model.dlg.ui,70,simBWF.getRadiobuttonValFromBool(config['sizeScaling']==0),true)
        simUI.setRadiobuttonValue(model.dlg.ui,71,simBWF.getRadiobuttonValFromBool(config['sizeScaling']==1),true)
        simUI.setRadiobuttonValue(model.dlg.ui,72,simBWF.getRadiobuttonValFromBool(config['sizeScaling']==2),true)
        
        
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
        model.dlg.updateEnabledDisabledItems()
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
end

function model.dlg.enabled_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
end

function model.dlg.showStatisticsClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
end

function model.dlg.triggerTypeClick_callback(ui,id)
    local c=model.readInfo()
    local w={4+8+16,8+16,4+16,16,4+8,8}
    local v=w[id-1000+1]
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4+8+16)-v
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
    model.dlg.updateEnabledDisabledItems()
end

function model.dlg.triggerStopChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboStopTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,sens)
    if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)==sens then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,-1)
    end
    simBWF.markUndoPoint()
    model.dlg.updateStartStopTriggerComboboxes()
end

function model.dlg.triggerStartChange_callback(ui,id,newIndex)
    local sens=model.dlg.comboStartTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL,sens)
    if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)==sens then
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL,-1)
    end
    simBWF.markUndoPoint()
    model.dlg.updateStartStopTriggerComboboxes()
end

function model.dlg.updateStartStopTriggerComboboxes()
    local c=model.readInfo()
    local loc=model.getAvailableSensors()
    model.dlg.comboStopTrigger=simBWF.populateCombobox(model.dlg.ui,100,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STOPSIGNAL)),true,{{simBWF.NONE_TEXT,-1}})
    model.dlg.comboStartTrigger=simBWF.populateCombobox(model.dlg.ui,101,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.STARTSIGNAL)),true,{{simBWF.NONE_TEXT,-1}})
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="77">
    <tab title="General" layout="form">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>
                
               <label text="Enabled" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.enabled_callback" id="40" />
                
                <label text="Distribution stretch (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.sizeChange_callback" id="60"/>

                <radiobutton text="Time trigger (parts/s)" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1000" />
                <edit on-editing-finished="model.dlg.frequencyChange_callback" id="2"/>

                <radiobutton text="Sensor triggered" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1001" />
                <combobox id="999" on-change="model.dlg.sensorComboChange_callback">
                </combobox>

                <radiobutton text="Conveyor belt triggered" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1003" />
                <combobox id="998" on-change="model.dlg.conveyorComboChange_callback">
                </combobox>

                <label text="Distance for trigger (mm)" style="* {margin-left: 20px; background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.conveyorDistanceChange_callback" id="62"/>

                <radiobutton text="User defined trigger" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1002" />
                <button text="Edit trigger algorithm"  on-click="model.dlg.dropAlgorithmClick_callback" id="3" />

                <radiobutton text="Multi-feeder triggered" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1004" />
                <label text=""/>

                <radiobutton text="Manual trigger" on-click="model.dlg.triggerTypeClick_callback" style="* {background-color: #ccffcc}" id="1005" />
                <label text=""/>
                
                <label text="Max. production count" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.maxProductionCntChange_callback" id="63"/>
                
    </tab>
    <tab title="Configuration" layout="form" id="54">
                <label text="Part distribution" style="* {background-color: #ccffcc}"/>
                <button text="Edit"  on-click="model.dlg.partDistribution_callback" id="11" />


                <label text="Relative position distribution" style="* {background-color: #ccffcc}"/>
                <button text="Edit"  on-click="model.dlg.shiftDistribution_callback" id="13" />

                <label text="Rotation distribution" style="* {background-color: #ccffcc}"/>
                <button text="Edit"  on-click="model.dlg.rotationDistribution_callback" id="14" />

                <label text="Weight distribution" style="* {background-color: #ccffcc}"/>
                <button text="Edit"  on-click="model.dlg.weightDistribution_callback" id="12" />

                <label text="Label distribution" style="* {background-color: #ccffcc}"/>
                <button text="Edit"  on-click="model.dlg.labelDistribution_callback" id="17" />

                <radiobutton text="No size scaling" style="* {background-color: #ccffcc}" on-click="model.dlg.sizeScalingClick_callback" id="70" />
                <label text=""/>

                <radiobutton text="Isometric size scaling distribution" style="* {background-color: #ccffcc}" on-click="model.dlg.sizeScalingClick_callback" id="71" />
                <button text="Edit"  on-click="model.dlg.isoSizeScalingDistribution_callback" id="73" />

                <radiobutton text="Non-isometric size scaling distribution" style="* {background-color: #ccffcc}" on-click="model.dlg.sizeScalingClick_callback" id="72" />
                <button text="Edit"  on-click="model.dlg.nonIsoSizeScalingDistribution_callback" id="74" />

                <label text="" style="* {margin-left: 190px;}"/>
                <label text="" style="* {margin-left: 190px;}"/>
    </tab>
    <tab title="More" layout="form">
                <label text="Stop on trigger" style="* {background-color: #ccffcc}"/>
                <combobox id="100" on-change="model.dlg.triggerStopChange_callback">
                </combobox>

                <label text="Restart on trigger" style="* {background-color: #ccffcc}"/>
                <combobox id="101" on-change="model.dlg.triggerStartChange_callback">
                </combobox>

                <label text="Hidden during simulation" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.hidden_callback" id="30" />

                <label text="Show statistics" style="* {background-color: #ccffcc}"/>
                 <checkbox text="" checked="false" on-change="model.dlg.showStatisticsClick_callback" id="50"/>
    </tab>
    </tabs>
        ]]
        
        
--                <label text="Destination distribution"/>
--                <button text="Edit"  on-click="model.dlg.destinationDistribution_callback"  id="49"/>
        
        

        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

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
    model.dlg.previousDlgPos,model.dlg.algoDlgSize,model.dlg.algoDlgPos,model.dlg.distributionDlgSize,model.dlg.distributionDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos,model.dlg.algoDlgSize,model.dlg.algoDlgPos,model.dlg.distributionDlgSize,model.dlg.distributionDlgPos)
end
