model.dlg={}

function model.dlg.sizeAChange_callback(ui,id,newVal)
    model.setArmLength(200+newVal*50,nil)
    model.dlg.refresh()
end

function model.dlg.sizeBChange_callback(ui,id,newVal)
    model.setArmLength(nil,400+newVal*50)
    model.dlg.refresh()
end

function model.dlg.frameHeightChange_callback(ui,id,newVal)
    local c=model.readInfo()
    c['frameHeightInMM']=newVal*50
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.adjustRobot()
    model.dlg.refresh()
end

function model.dlg.velocityChange_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        newValue=newValue/1000
        if newValue~=c['maxVel'] then
            c['maxVel']=newValue
            model.writeInfo(c)
            simBWF.markUndoPoint()
            model.adjustMaxVelocityMaxAcceleration()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.accelerationChange_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        newValue=newValue/1000
        if newValue~=c['maxAccel'] then
            c['maxAccel']=newValue
            model.writeInfo(c)
            simBWF.markUndoPoint()
            model.adjustMaxVelocityMaxAcceleration()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.visualizeWorkspaceClick_callback(uiHandle,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],256)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-256
    end
    model.writeInfo(c)

    if newVal>0 then
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,1)
        model.workspaceUpdateRequest=sim.getSystemTimeInMs(-1) -- to trigger recomputation
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWs,sim.objintparam_visibility_layer,0)
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.visualizeWorkspaceSimClick_callback(uiHandle,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],512)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-512
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.visualizeWsBoxClick_callback(uiHandle,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    model.writeInfo(c)

    if newVal>0 then
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Parameter(model.handles.ragnarWsBox,sim.objintparam_visibility_layer,0)
    end
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.visualizeWsBoxSimClick_callback(uiHandle,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.enabledClicked_callback(ui,id,newVal)
    local c=model.readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.hideHousingClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],8)
    if newVal==0 then
        c['jobBitCoded']=c['jobBitCoded']-8
    end
    model.writeInfo(c)
    model.hideHousing(newVal~=0)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.hideFrameClick_callback(ui,id,newVal)
    -- Hiding the frame, but it can still be present!
    local c=model.readInfo()
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],16)
    local s=2 -- hidden
    if newVal==0 then
        c['jobBitCoded']=c['jobBitCoded']-16
        s=1 -- visible
    end
    model.writeInfo(c)
    if c['frameType']==C.FRAMETYPELIST[1] then
        s=0 -- not present
    end
    model.setFrameState(s)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.attachPartClicked_callback(ui,id,newVal)
    local c=model.readInfo()
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],1024)
    if newVal==0 then
        c['jobBitCoded']=c['jobBitCoded']-1024
    end
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.ignorePartDestinationsClicked_callback(ui,id,newVal)
    local c=model.readInfo()
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],4096)
    if newVal==0 then
        c['jobBitCoded']=c['jobBitCoded']-4096
    end
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.pickWithoutTargetClicked_callback(ui,id,newVal)
    local c=model.readInfo()
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],2048)
    if newVal==0 then
        c['jobBitCoded']=c['jobBitCoded']-2048
    end
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.aliasComboChange_callback(uiHandle,id,newValue)
    local newAlias=model.dlg.comboAlias[newValue+1][1]
    local c=model.readInfo()
    c['robotAlias']=newAlias
    model.writeInfo(c)
    simBWF.markUndoPoint()
--    model.dlg.updateAliasCombobox()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.motorTypeChange_callback(uiHandle,id,newIndex)
    local newType=model.dlg.motorType_comboboxItems[newIndex+1][2]
    local c=model.readInfo()
    c['motorType']=newType
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.dlg.updateMotorTypeCombobox()
    model.adjustMaxVelocityMaxAcceleration()
    model.dlg.refresh()
    model.updatePluginRepresentation()
end

function model.dlg.updateMotorTypeCombobox()
    local c=model.readInfo()
    local loc={}
    for i=1,#C.MOTORTYPELIST,1 do
        loc[i]={C.MOTORTYPES[C.MOTORTYPELIST[i]].text,C.MOTORTYPELIST[i]}
    end
    model.dlg.motorType_comboboxItems=simBWF.populateCombobox(model.dlg.ui,95,loc,{},loc[c.motorType+1][1],false,{})
end

function model.dlg.exteriorTypeChange_callback(uiHandle,id,newIndex)
    local newType=model.dlg.exteriorType_comboboxItems[newIndex+1][2]
    local c=model.readInfo()
    c['exteriorType']=newType
    model.writeInfo(c)

    local col1={0.2,0.24,0.29} -- 'default'
    local col2={0.7,0.7,0.7} -- 'default'
    if newType==1 then
        col1={0.2,0.4,0.58} -- 'wash-down'
        col2={0.85,0.85,1}
    end
    if newType==2 then
        col1={0.83,0.85,0.86} -- 'hygienic'
        col2={0.85,0.85,1}
    end
    local s=sim.getObjectsInTree(model.handle,sim.object_shape_type)
    for i=1,#s,1 do
        sim.setShapeColor(s[i],'DARK_BLUE',sim.colorcomponent_ambient_diffuse,col1)
        sim.setShapeColor(s[i],'LIGHT_BLUE',sim.colorcomponent_ambient_diffuse,col2)
    end

    simBWF.markUndoPoint()
    model.dlg.updateExteriorTypeCombobox()
    model.updatePluginRepresentation()
end

function model.dlg.updateExteriorTypeCombobox()
    local c=model.readInfo()
    local loc={}
    for i=1,#C.EXTERIORTYPELIST,1 do
        loc[i]={C.EXTERIORTYPES[C.EXTERIORTYPELIST[i]].text,C.EXTERIORTYPELIST[i]}
    end
    model.dlg.exteriorType_comboboxItems=simBWF.populateCombobox(model.dlg.ui,96,loc,{},loc[c.exteriorType+1][1],false,{})
end

function model.dlg.frameTypeChange_callback(uiHandle,id,newIndex)
    local newType=model.dlg.frameType_comboboxItems[newIndex+1][2]
    local c=model.readInfo()
    c['frameType']=newType
    c['jobBitCoded']=sim.boolOr32(c['jobBitCoded'],16)-16 -- reset the 'hide' flag (frame could be present but hidden)

    local s=0
    if newType~=C.FRAMETYPELIST[1] then
        s=1
    end

    model.writeInfo(c)

    model.setFrameState(s)

    simBWF.markUndoPoint()
    model.dlg.updateFrameTypeCombobox()
    model.dlg.refresh()
    model.updatePluginRepresentation()
end

function model.dlg.updateFrameTypeCombobox()
    local c=model.readInfo()
    local loc={}
    for i=1,#C.FRAMETYPELIST,1 do
        loc[i]={C.FRAMETYPES[C.FRAMETYPELIST[i]].text,C.FRAMETYPELIST[i]}
    end
    model.dlg.frameType_comboboxItems=simBWF.populateCombobox(model.dlg.ui,97,loc,{},loc[c.frameType+1][1],false,{})
end

function model.dlg.frameDoorChange_callback(uiHandle,id,newIndex)
    local state=model.dlg.frameDoor_comboboxItems[newIndex+1][2]
    local c=model.readInfo()
    c['frameDoor']=state
    model.writeInfo(c)

    model.setFrameDoorState(state)

    simBWF.markUndoPoint()
    model.dlg.updateFrameDoorCombobox()
    model.dlg.refresh()
    model.updatePluginRepresentation()
end

function model.dlg.updateFrameDoorCombobox()
    local c=model.readInfo()
    local loc={}
    for i=1,#C.FRAMEDOORSTATELIST,1 do
        loc[i]={C.FRAMEDOORSTATES[C.FRAMEDOORSTATELIST[i]].text,C.FRAMEDOORSTATELIST[i]}
    end
    model.dlg.frameDoor_comboboxItems=simBWF.populateCombobox(model.dlg.ui,98,loc,{},loc[c.frameDoor+1][1],false,{})
end

function model.dlg.simBufferSize_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>10000 then newValue=10000 end
        if c['connectionBufferSize'][1]~=newValue then
            c['connectionBufferSize'][1]=newValue
            simBWF.markUndoPoint()
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end

function model.dlg.simShowRobotPlotClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c.showPlot[1]=not c.showPlot[1]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.simShowTrajectoryClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c.showTrajectory[1]=not c.showTrajectory[1]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.showClearanceClick_callback(ui,id,newVal)
    local c=model.readInfo()
    local ind=1
    if id~=1307 then
        ind=2
    end
    c.clearance[ind]=not c.clearance[ind]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.clearanceWithPlatformClick_callback(ui,id,newVal)
    local c=model.readInfo()
    local ind=1
    if id~=1308 then
        ind=2
    end
    c.clearanceWithPlatform[ind]=not c.clearanceWithPlatform[ind]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.clearanceWarning_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        newValue=newValue/1000
        if newValue<0.001 then newValue=0 end
        if newValue>1 then newValue=1 end
        if c['clearanceWarning'][1]~=newValue then
            c['clearanceWarning'][1]=newValue
            simBWF.markUndoPoint()
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end

function model.dlg.simVisualizeUpdateFrequChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['visualizeUpdateFrequ'][1]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end


function model.dlg.realBufferSize_callback(uiHandle,id,newValue)
    local c=model.readInfo()
    newValue=tonumber(newValue)
    if newValue then
        if newValue<1 then newValue=1 end
        if newValue>10000 then newValue=10000 end
        if c['connectionBufferSize'][2]~=newValue then
            c['connectionBufferSize'][2]=newValue
            simBWF.markUndoPoint()
            model.writeInfo(c)
        end
    end
    model.dlg.refresh()
end

function model.dlg.realShowRobotPlotClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c.showPlot[2]=not c.showPlot[2]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.realShowTrajectoryClick_callback(ui,id,newVal)
    local c=model.readInfo()
    c.showTrajectory[2]=not c.showTrajectory[2]
    simBWF.markUndoPoint()
    model.writeInfo(c)
    model.dlg.refresh()
end

function model.dlg.realVisualizeUpdateFrequChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['visualizeUpdateFrequ'][2]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.pickTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.pickTrackingWindows_comboboxItems[id-610][newIndex+1][2]
    -- Make sure no other has the same item:
    for i=1,C.CIC,1 do
        if i~=id-610 then
            if newLoc~=-1 and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)==newLoc then
                model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1),-1)
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1,-1)
            end
        end
    end
    -- Clear calibration data of previous item:
    local itm=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+id-610-1)
    if itm>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",itm)
    end
    -- Clear calibration data of current item:
    if newLoc>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",newLoc)
    end
    -- Set the item:
    model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+id-610-1),newLoc)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+id-610-1,newLoc)
    -- Make sure other Ragnar robots do not reference this item:
    if newLoc>=0 then
        local allRagnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
        for i=1,#allRagnars,1 do
            local m=allRagnars[i]
            if m~=model.handle then
                for j=1,C.CIC,1 do
                    local item=simBWF.getReferencedObjectHandle(m,model.objRefIdx.PICKTRACKINGWINDOW1+j-1)
                    if item==newLoc then
                        simBWF.setReferencedObjectHandle(m,model.objRefIdx.PICKTRACKINGWINDOW1+j-1,-1) -- the item was same. We set it to -1
                    end
                end
            end
        end
    end

    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updatePickTrackingWindowComboboxes()
end

function model.dlg.placeTrackingWindowChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.placeTrackingWindows_comboboxItems[id-620][newIndex+1][2]
    -- Make sure no other has the same item:
    for i=1,C.CIC,1 do
        if i~=id-620 then
            if newLoc~=-1 and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)==newLoc then
                model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1),-1)
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1,-1)
            end
        end
    end
    -- Clear calibration data of previous item:
    local itm=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+id-620-1)
    if itm>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",itm)
    end
    -- Clear calibration data of current item:
    if newLoc>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",newLoc)
    end
    -- Set the item:
    model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+id-620-1),newLoc)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+id-620-1,newLoc)
    -- Make sure other Ragnar robots do not reference this item:
    if newLoc>=0 then
        local allRagnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
        for i=1,#allRagnars,1 do
            local m=allRagnars[i]
            if m~=model.handle then
                for j=1,C.CIC,1 do
                    local item=simBWF.getReferencedObjectHandle(m,model.objRefIdx.PLACETRACKINGWINDOW1+j-1)
                    if item==newLoc then
                        simBWF.setReferencedObjectHandle(m,model.objRefIdx.PLACETRACKINGWINDOW1+j-1,-1) -- the item was same. We set it to -1
                    end
                end
            end
        end
    end

    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updatePlaceTrackingWindowComboboxes()
end


function model.dlg.pickFrameChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.pickFrames_comboboxItems[id-600][newIndex+1][2]
    -- Make sure no other has the same item:
    for i=1,C.CIC,1 do
        if i~=id-600 then
            if newLoc~=-1 and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)==newLoc then
                model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1),-1)
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1,-1)
            end
        end
    end
    -- Clear calibration data of previous item:
    local itm=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+id-600-1)
    if itm>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",itm)
    end
    -- Clear calibration data of current item:
    if newLoc>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",newLoc)
    end
    -- Set the item:
    model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+id-600-1),newLoc)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+id-600-1,newLoc)
    -- Make sure other Ragnar robots do not reference this item:
    if newLoc>=0 then
        local allRagnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
        for i=1,#allRagnars,1 do
            local m=allRagnars[i]
            if m~=model.handle then
                for j=1,C.CIC,1 do
                    local item=simBWF.getReferencedObjectHandle(m,model.objRefIdx.PICKFRAME1+j-1)
                    if item==newLoc then
                        simBWF.setReferencedObjectHandle(m,model.objRefIdx.PICKFRAME1+j-1,-1) -- the item was same. We set it to -1
                    end
                end
            end
        end
    end

    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updatePickFrameComboboxes()
end

function model.dlg.placeFrameChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.placeFrames_comboboxItems[id-500][newIndex+1][2]
    -- Make sure no other has the same item:
    for i=1,C.CIC,1 do
        if i~=id-500 then
            if newLoc~=-1 and simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)==newLoc then
                model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1),-1)
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1,-1)
            end
        end
    end
    -- Clear calibration data of previous item:
    local itm=simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+id-500-1)
    if itm>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",itm)
    end
    -- Clear calibration data of current item:
    if newLoc>=0 then
        simBWF.callCustomizationScriptFunction("model.ext.clearCalibration",newLoc)
    end
    -- Set the item:
    model.attachOrDetachReferencedItem(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+id-500-1),newLoc)
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+id-500-1,newLoc)
    -- Make sure other Ragnar robots do not reference this item:
    if newLoc>=0 then
        local allRagnars=sim.getObjectsWithTag(simBWF.modelTags.RAGNAR,true)
        for i=1,#allRagnars,1 do
            local m=allRagnars[i]
            if m~=model.handle then
                for j=1,C.CIC,1 do
                    local item=simBWF.getReferencedObjectHandle(m,model.objRefIdx.PLACEFRAME1+j-1)
                    if item==newLoc then
                        simBWF.setReferencedObjectHandle(m,model.objRefIdx.PLACEFRAME1+j-1,-1) -- the item was same. We set it to -1
                    end
                end
            end
        end
    end

    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updatePlaceFrameComboboxes()
end

function model.dlg.updatePickTrackingWindowComboboxes()
    model.dlg.pickTrackingWindows_comboboxItems={}
    local loc=model.getAvailableTrackingWindows(true)
    for i=1,C.ECIC_PiW,1 do
        model.dlg.pickTrackingWindows_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,611+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKTRACKINGWINDOW1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updatePlaceTrackingWindowComboboxes()
    model.dlg.placeTrackingWindows_comboboxItems={}
    local loc=model.getAvailableTrackingWindows(false)
    for i=1,C.ECIC_PlW,1 do
        model.dlg.placeTrackingWindows_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,621+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACETRACKINGWINDOW1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updatePickFrameComboboxes()
    model.dlg.pickFrames_comboboxItems={}
    local loc=model.getAvailableFrames(true)
    for i=1,C.ECIC_PiL,1 do
        model.dlg.pickFrames_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,601+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PICKFRAME1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updatePlaceFrameComboboxes()
    model.dlg.placeFrames_comboboxItems={}
    local loc=model.getAvailableFrames(false)
    for i=1,C.ECIC_PlL,1 do
        model.dlg.placeFrames_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,501+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.PLACEFRAME1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

--[[
function model.dlg.conveyorChange_callback(ui,id,newIndex)
    local newLoc=model.dlg.conveyor_comboboxItems[id-1200][newIndex+1][2]
    -- Make sure no other has the same item:
    for i=1,2,1 do
        if i~=id-1200 then
            if simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1+i-1)==newLoc then
                simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1+i-1,-1)
            end
        end
    end
    -- Set the item:
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1+id-1200-1,newLoc)
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updateConveyorComboboxes()
end

function model.dlg.updateConveyorComboboxes()
    model.dlg.conveyor_comboboxItems={}
    local loc=model.getAvailableConveyors()
    for i=1,2,1 do
        model.dlg.conveyor_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,1201+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.CONVEYOR1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end
--]]

function model.dlg.updateAliasCombobox()
    local c=model.readInfo()
    local resp,data
    if simBWF.isInTestMode() then
        resp='ok'
        data={}
        data.aliases={'testID-1','testID-2'}
    else
        resp,data=simBWF.query('get_ragnarAliases')
        if resp~='ok' then
            data.aliases={}
        end
    end

    local selected=c['robotAlias']
    local isKnown=false
    local items={}
    for i=1,#data.aliases,1 do
        if data.aliases[i]==selected then
            isKnown=true
        end
        items[#items+1]={data.aliases[i],i}
    end
    if not isKnown then
        table.insert(items,1,{selected,#items+1})
    end
    if selected~=simBWF.NONE_TEXT then
        table.insert(items,1,{simBWF.NONE_TEXT,#items+1})
    end
    model.dlg.comboAlias=simBWF.populateCombobox(model.dlg.ui,1200,items,{},selected,false,{})
--    model.updatePluginRepresentation()
end

function model.dlg.identificationAndRenaming_identification_callback(ui,id,newVal)
    local c=model.readInfo()
    local data={}
    data.alias=c.robotAlias
    simBWF.query('identify_ragnarFromAlias',data)
    data={}
    data.deviceId=c.deviceId
    simBWF.query('identify_device',data)
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2,simStopped,true)
        simUI.setEnabled(model.dlg.ui,92,simStopped,true)
        simUI.setEnabled(model.dlg.ui,94,simStopped,true)
        simUI.setEnabled(model.dlg.ui,95,simStopped,true)
        simUI.setEnabled(model.dlg.ui,96,simStopped,true)
        simUI.setEnabled(model.dlg.ui,97,simStopped,true)
        simUI.setEnabled(model.dlg.ui,98,simStopped and c.frameType==1,true)
        simUI.setEnabled(model.dlg.ui,611,simStopped,true)
        simUI.setEnabled(model.dlg.ui,612,simStopped,true)
        simUI.setEnabled(model.dlg.ui,621,simStopped,true)
        simUI.setEnabled(model.dlg.ui,622,simStopped,true)
        simUI.setEnabled(model.dlg.ui,501,simStopped,true)
        simUI.setEnabled(model.dlg.ui,502,simStopped,true)
        simUI.setEnabled(model.dlg.ui,503,simStopped,true)
        simUI.setEnabled(model.dlg.ui,504,simStopped,true)
        simUI.setEnabled(model.dlg.ui,601,simStopped,true)
        simUI.setEnabled(model.dlg.ui,602,simStopped,true)
        simUI.setEnabled(model.dlg.ui,603,simStopped,true)
        simUI.setEnabled(model.dlg.ui,604,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2001,simStopped,true)
        simUI.setEnabled(model.dlg.ui,2002,simStopped,true)


        simUI.setEnabled(model.dlg.ui,3,simStopped,true)
        simUI.setEnabled(model.dlg.ui,305,simStopped and sim.boolAnd32(c.bitCoded,256)>0,true)

        simUI.setEnabled(model.dlg.ui,3002,simStopped,true)
        simUI.setEnabled(model.dlg.ui,3003,simStopped and sim.boolAnd32(c.bitCoded,1)>0,true)
        simUI.setEnabled(model.dlg.ui,3000,simStopped and sim.boolAnd32(c.bitCoded,1)>0,true)
        simUI.setEnabled(model.dlg.ui,3001,simStopped and sim.boolAnd32(c.bitCoded,1)>0,true)

        simUI.setEnabled(model.dlg.ui,312,c.frameType~=0,true)


        local online=simBWF.isSystemOnline()
        simUI.setEnabled(model.dlg.ui,1200,simStopped,true)
--        simUI.setEnabled(model.dlg.ui,1201,simStopped,true)
--        simUI.setEnabled(model.dlg.ui,1202,simStopped,true)
        simUI.setEnabled(model.dlg.ui,1303,simStopped,true)
        simUI.setEnabled(model.dlg.ui,1203,simStopped,true)

        for i=1,8,1 do
            simUI.setEnabled(model.dlg.ui,700+i,simStopped,true) -- inputs
            simUI.setEnabled(model.dlg.ui,710+i,simStopped,true) -- outputs
        end


        local runningOnline=not simStopped and online
        local runningSim=not simStopped and not online

        simUI.setEnabled(model.dlg.ui,1304,runningSim or simStopped,true)
        simUI.setEnabled(model.dlg.ui,1305,runningSim or simStopped,true)
        simUI.setEnabled(model.dlg.ui,1306,runningSim or simStopped,true)

        simUI.setEnabled(model.dlg.ui,1204,runningOnline or simStopped,true)
        simUI.setEnabled(model.dlg.ui,1205,runningOnline or simStopped,true)
        simUI.setEnabled(model.dlg.ui,1206,runningOnline or simStopped,true)

        simUI.setEnabled(model.dlg.ui,1300,simStopped and ( c.robotAlias~=simBWF.NONE_TEXT or c.deviceId~=simBWF.NONE_TEXT ),true)

        simUI.setEnabled(model.dlg.ui,1307,runningSim or simStopped,true)
        simUI.setEnabled(model.dlg.ui,1308,(runningSim or simStopped) and c.clearance[1],true)
        simUI.setEnabled(model.dlg.ui,1310,(runningSim or simStopped) and c.clearance[1],true)

    end
end

function model.dlg.refresh()
    if model.dlg.ui then
        local c=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setSliderValue(model.dlg.ui,2,(c['primaryArmLengthInMM']-200)/50,true)
        simUI.setSliderValue(model.dlg.ui,92,(c['secondaryArmLengthInMM']-400)/50,true)
        simUI.setSliderValue(model.dlg.ui,94,c['frameHeightInMM']/50,true)

        simUI.setCheckboxValue(model.dlg.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],256)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,305,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],512)~=0),true)

        simUI.setCheckboxValue(model.dlg.ui,3002,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,3003,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],4)~=0),true)

        for i=1,2,1 do
            local wsBoxPt=c.wsBox[i]
            simUI.setEditValue(model.dlg.ui,3000+i-1,simBWF.format("%.0f , %.0f , %.0f",wsBoxPt[1]*1000,wsBoxPt[2]*1000,wsBoxPt[3]*1000),true)
        end


        for i=1,2,1 do
            local coord=c.waitLocAfterPickOrPlace[i]
            simUI.setEditValue(model.dlg.ui,3004+i-1,simBWF.format("%.0f , %.0f , %.0f",coord[1]*1000,coord[2]*1000,coord[3]*1000),true)
        end



        simUI.setCheckboxValue(model.dlg.ui,1000,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],64)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,311,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['jobBitCoded'],8)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,312,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['jobBitCoded'],16)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,2000,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['jobBitCoded'],1024)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,2001,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['jobBitCoded'],2048)~=0),true)
        simUI.setCheckboxValue(model.dlg.ui,2002,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['jobBitCoded'],4096)~=0),true)

        simUI.setCheckboxValue(model.dlg.ui,1304,simBWF.getCheckboxValFromBool(c.showPlot[1]),true)
        simUI.setCheckboxValue(model.dlg.ui,1306,simBWF.getCheckboxValFromBool(c.showTrajectory[1]),true)
        simUI.setCheckboxValue(model.dlg.ui,1204,simBWF.getCheckboxValFromBool(c.showPlot[2]),true)
        simUI.setCheckboxValue(model.dlg.ui,1206,simBWF.getCheckboxValFromBool(c.showTrajectory[2]),true)

        simUI.setCheckboxValue(model.dlg.ui,1307,simBWF.getCheckboxValFromBool(c.clearance[1]),true)
        simUI.setCheckboxValue(model.dlg.ui,1308,simBWF.getCheckboxValFromBool(c.clearanceWithPlatform[1]),true)
        if c['clearanceWarning'][1]<0.001 then
            simUI.setEditValue(model.dlg.ui,1310,simBWF.NONE_TEXT,true)
        else
            simUI.setEditValue(model.dlg.ui,1310,simBWF.format("%i",c['clearanceWarning'][1]*1000),true)
        end

        simUI.setEditValue(model.dlg.ui,1303,simBWF.format("%i",c['connectionBufferSize'][1]),true)
        simUI.setEditValue(model.dlg.ui,1203,simBWF.format("%i",c['connectionBufferSize'][2]),true)
        simUI.setLabelText(model.dlg.ui,1,'Primary arm length: '..simBWF.format("%.0f",c['primaryArmLengthInMM'])..' mm')
        simUI.setLabelText(model.dlg.ui,91,'Secondary arm length: '..simBWF.format("%.0f",c['secondaryArmLengthInMM'])..' mm')
        simUI.setLabelText(model.dlg.ui,93,'Reference point Z pos.: '..simBWF.format("%.0f",c['frameHeightInMM'])..' mm')
        simUI.setEditValue(model.dlg.ui,10,simBWF.format("%.0f",c['maxVel']*1000),true)
        simUI.setEditValue(model.dlg.ui,11,simBWF.format("%.0f",c['maxAccel']*1000),true)

        model.dlg.updatePickTrackingWindowComboboxes()
        model.dlg.updatePlaceTrackingWindowComboboxes()
        model.dlg.updatePickFrameComboboxes()
        model.dlg.updatePlaceFrameComboboxes()
        -- model.dlg.updateAliasCombobox()
        model.dlg.updateMotorTypeCombobox()
        model.dlg.updateExteriorTypeCombobox()
        model.dlg.updateFrameTypeCombobox()
        model.dlg.updateFrameDoorCombobox()

        model.dlg.updateConveyorInputBoxComboboxes()
        model.dlg.updateInputBoxComboboxes()
        model.dlg.updateOutputBoxComboboxes()

--        model.dlg.updateConveyorComboboxes()

        local updateFrequComboItems={
            {"every 50 ms",0},
            {"every 200 ms",1},
            {"every 1000 ms",2}
        }
        simBWF.populateCombobox(model.dlg.ui,1305,updateFrequComboItems,{},updateFrequComboItems[c['visualizeUpdateFrequ'][1]+1][1],false,nil)
        simBWF.populateCombobox(model.dlg.ui,1205,updateFrequComboItems,{},updateFrequComboItems[c['visualizeUpdateFrequ'][2]+1][1],false,nil)

        model.dlg.updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
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

function model.dlg.wsBox_change(ui,id,newValue)
    local tolU=0
    local tolL=0.05
    local index=id-3000+1
    if index==1 then
        tolU=0.05
        tolL=0
    end
    local c=model.readInfo()
    local i=1
    local t={0,0,0}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001

        if index==1 then
            if t[i]>c.wsBox[2][i]-tolU then t[i]=c.wsBox[2][i]-tolU end
        else
            if t[i]<c.wsBox[1][i]+tolL then t[i]=c.wsBox[1][i]+tolL end
        end

        if t[i]>1-tolU then t[i]=1-tolU end
        if t[i]<-1+tolL then t[i]=-1+tolL end
        i=i+1
    end
    c.wsBox[index]={t[1],t[2],t[3]}
    model.writeInfo(c)
    model.adjustWsBox()
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.waitingLocationAfterPickOrPlace_change(ui,id,newValue)
    local index=id-3004+1
    local c=model.readInfo()
    local i=1
    local t={0,0,0}
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001

        if t[i]>0.5 then t[i]=0.5 end
        if t[i]<-0.5 then t[i]=-0.5 end

        i=i+1
    end
    c.waitLocAfterPickOrPlace[index]={t[1],t[2],t[3]}
    model.writeInfo(c)
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.updateConveyorInputBoxComboboxes()
    model.dlg.conveyorInputBox_comboboxItems={}
    local loc=model.getAvailableConveyors()
    for i=1,2,1 do
        model.dlg.conveyorInputBox_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,701+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updateInputBoxComboboxes()
    model.dlg.inputBox_comboboxItems={}
    local loc=model.getAvailableInputBoxes()
    for i=3,8,1 do
        model.dlg.inputBox_comboboxItems[i-2]=simBWF.populateCombobox(model.dlg.ui,701+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.updateOutputBoxComboboxes()
    model.dlg.outputBox_comboboxItems={}
    local loc=model.getAvailableOutputBoxes()
    for i=1,8,1 do
        model.dlg.outputBox_comboboxItems[i]=simBWF.populateCombobox(model.dlg.ui,711+i-1,loc,{},simBWF.getObjectAltNameOrNone(simBWF.getReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+i-1)),true,{{simBWF.NONE_TEXT,-1}})
    end
end

function model.dlg.inputChange_callback(ui,id,newIndex)
    local idd=id-700
    if idd<3 then
        -- Input 1&2 (conveyors)
        local newLoc=model.dlg.conveyorInputBox_comboboxItems[idd][newIndex+1][2]
        if newLoc~=-1 then
            simBWF.disconnectInputOrOutputBox(newLoc)
        end
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+idd-1,newLoc)
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        model.dlg.updateConveyorInputBoxComboboxes()
    else
        -- Input 3-8 (input boxes)
        local newLoc=model.dlg.inputBox_comboboxItems[idd-2][newIndex+1][2]
        if newLoc~=-1 then
            simBWF.disconnectInputOrOutputBox(newLoc)
        end
        simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.INPUT1+idd-1,newLoc)
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        model.dlg.updateInputBoxComboboxes()
    end
end

function model.dlg.outputChange_callback(ui,id,newIndex)
    local idd=id-700
    -- Outputs 1-8 (output boxes)
    local newLoc=model.dlg.outputBox_comboboxItems[idd-10][newIndex+1][2]
    if newLoc~=-1 then
        simBWF.disconnectInputOrOutputBox(newLoc)
    end
    simBWF.setReferencedObjectHandle(model.handle,model.objRefIdx.OUTPUT1+idd-11,newLoc)
    simBWF.markUndoPoint()
    model.updatePluginRepresentation()
    model.dlg.updateOutputBoxComboboxes()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="78">
    <tab title="General">
            <group layout="form" flat="false">
                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

                <label text="Enabled"/>
                <checkbox text="" on-change="model.dlg.enabledClicked_callback" id="1000"/>

                <label text="Maximum speed (mm/s)" style="* {background-color: #ccffcc}" />
                <edit on-editing-finished="model.dlg.velocityChange_callback" id="10"/>

                <label text="Maximum acceleration (mm/s^2)"  style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.accelerationChange_callback" id="11"/>
            </group>
            <group layout="form" flat="false">
                <label text="Workspace" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Show actual workspace"/>
                <checkbox text="" checked="false" on-change="model.dlg.visualizeWorkspaceClick_callback" id="3"/>

                <label text="Show actual workspace also when running"/>
                <checkbox text="" checked="false" on-change="model.dlg.visualizeWorkspaceSimClick_callback" id="305"/>

                <label text="Show workspace box"/>
                <checkbox text="" checked="false" on-change="model.dlg.visualizeWsBoxClick_callback" id="3002"/>

                <label text="Show workspace box also when running"/>
                <checkbox text="" checked="false" on-change="model.dlg.visualizeWsBoxSimClick_callback" id="3003"/>

                <label text="workspace box min. coordinate (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.wsBox_change" id="3000"/>

                <label text="workspace box max. coordinate (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.wsBox_change" id="3001"/>

            </group>
    </tab>
    <tab title="Pick/Place">

            <group layout="form" flat="false">
                <label text="Waiting locations" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="After pick (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.waitingLocationAfterPickOrPlace_change" id="3004"/>

                <label text="After place (X, Y, Z, in mm)" style="* {background-color: #ccffcc}"/>
                <edit on-editing-finished="model.dlg.waitingLocationAfterPickOrPlace_change" id="3005"/>

            </group>

            <group layout="form" flat="false">
                <label text="Other" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Pick without target in sight" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.pickWithoutTargetClicked_callback" id="2001"/>

                <label text="Attach part to target" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.attachPartClicked_callback" id="2000"/>

                <label text="Ignore part destinations" style="* {background-color: #ccffcc}"/>
                <checkbox text="" on-change="model.dlg.ignorePartDestinationsClicked_callback" id="2002"/>
            </group>
    </tab>
    <tab title="Configuration">
            <group layout="form" flat="false">
                <label text="Pick" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Tracking window 1"/>
                <combobox id="611" on-change="model.dlg.pickTrackingWindowChange_callback">
                </combobox>

                <label text="Tracking window 2"/>
                <combobox id="612" on-change="model.dlg.pickTrackingWindowChange_callback">
                </combobox>

                <label text="Location frame 1"/>
                <combobox id="601" on-change="model.dlg.pickFrameChange_callback">
                </combobox>

                <label text="Location frame 2"/>
                <combobox id="602" on-change="model.dlg.pickFrameChange_callback">
                </combobox>

                <label text="Location frame 3"/>
                <combobox id="603" on-change="model.dlg.pickFrameChange_callback">
                </combobox>

                <label text="Location frame 4"/>
                <combobox id="604" on-change="model.dlg.pickFrameChange_callback">
                </combobox>
            </group>

            <group layout="form" flat="false">
                <label text="Place" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Tracking window 1"/>
                <combobox id="621" on-change="model.dlg.placeTrackingWindowChange_callback">
                </combobox>

                <label text="Tracking window 2"/>
                <combobox id="622" on-change="model.dlg.placeTrackingWindowChange_callback">
                </combobox>

                <label text="Location frame 1"/>
                <combobox id="501" on-change="model.dlg.placeFrameChange_callback">
                </combobox>

                <label text="Location frame 2"/>
                <combobox id="502" on-change="model.dlg.placeFrameChange_callback">
                </combobox>

                <label text="Location frame 3"/>
                <combobox id="503" on-change="model.dlg.placeFrameChange_callback">
                </combobox>

                <label text="Location frame 4"/>
                <combobox id="504" on-change="model.dlg.placeFrameChange_callback">
                </combobox>
            </group>
    </tab>
    <tab title="Robot">
            <group layout="form" flat="false">
                <label text="Type" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Primary arm length" id="1"/>
                <hslider tick-position="above" tick-interval="1" minimum="0" maximum="7" on-change="model.dlg.sizeAChange_callback" id="2"/>

                <label text="Secondary arm length" id="91"/>
                <hslider tick-position="above" tick-interval="1" minimum="0" maximum="17" on-change="model.dlg.sizeBChange_callback" id="92"/>

                <label text="Robot Z position" id="93"/>
                <hslider tick-position="above" tick-interval="1" minimum="24" maximum="38" on-change="model.dlg.frameHeightChange_callback" id="94"/>

                <label text="Motor type"/>
                <combobox id="95" on-change="model.dlg.motorTypeChange_callback"></combobox>

                <label text="Exterior type"/>
                <combobox id="96" on-change="model.dlg.exteriorTypeChange_callback"></combobox>

                <label text="Frame type"/>
                <combobox id="97" on-change="model.dlg.frameTypeChange_callback"></combobox>

                <label text="Frame Door"/>
                <combobox id="98" on-change="model.dlg.frameDoorChange_callback"></combobox>
            </group>

    </tab>
    <tab title="Simulation">
            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Buffer size (states)"/>
                <edit on-editing-finished="model.dlg.simBufferSize_callback" id="1303"/>

                <label text="Show robot plot"/>
                <checkbox text="" checked="false" on-change="model.dlg.simShowRobotPlotClick_callback" id="1304"/>

                <label text="Show trajectory"/>
                <checkbox text="" checked="false" on-change="model.dlg.simShowTrajectoryClick_callback" id="1306"/>

                <label text="Show robot clearance plot"/>
                <checkbox text="" checked="false" on-change="model.dlg.showClearanceClick_callback" id="1307"/>

                <label text="Include platform & gripper"/>
                <checkbox text="" checked="false" on-change="model.dlg.clearanceWithPlatformClick_callback" id="1308"/>

                <label text="Clearance warning (mm)"/>
                <edit on-editing-finished="model.dlg.clearanceWarning_callback" id="1310"/>

                <label text="Update frequency"/>
                <combobox id="1305" on-change="model.dlg.simVisualizeUpdateFrequChange_callback"></combobox>
            </group>
            <group layout="form" flat="true">
                <label text="" style="* {margin-left: 200px;}"/>
                <label text="" style="* {margin-left: 200px;}"/>
            </group>
    </tab>
    <tab title="Online">
            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Buffer size (states)"/>
                <edit on-editing-finished="model.dlg.realBufferSize_callback" id="1203"/>

                <label text="Show robot plot"/>
                <checkbox text="" checked="false" on-change="model.dlg.realShowRobotPlotClick_callback" id="1204"/>

                <label text="Show trajectory"/>
                <checkbox text="" checked="false" on-change="model.dlg.realShowTrajectoryClick_callback" id="1206"/>

                <label text="Update frequency"/>
                <combobox id="1205" on-change="model.dlg.realVisualizeUpdateFrequChange_callback"></combobox>
            </group>
            <group layout="form" flat="false">
                <label text="Real robot specific" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Robot serial"/>
                <combobox id="1200" on-change="model.dlg.aliasComboChange_callback"> </combobox>

                <label text=""/>
                <button text="Identify"  on-click="model.dlg.identificationAndRenaming_identification_callback" id="1300" />

            </group>
    </tab>
    <tab title="Input Connections">
            <group layout="form" flat="false">

                <label text="Conveyor1" />
                <combobox on-change="model.dlg.inputChange_callback" id="701" />

                <label text="Conveyor2" />
                <combobox on-change="model.dlg.inputChange_callback" id="702" />

                <label text="Input3" />
                <combobox on-change="model.dlg.inputChange_callback" id="703" />

                <label text="Input4" />
                <combobox on-change="model.dlg.inputChange_callback" id="704" />

                <label text="Input5" />
                <combobox on-change="model.dlg.inputChange_callback" id="705" />

                <label text="Input6" />
                <combobox on-change="model.dlg.inputChange_callback" id="706" />

                <label text="Input7" />
                <combobox on-change="model.dlg.inputChange_callback" id="707" />

                <label text="Input8" />
                <combobox on-change="model.dlg.inputChange_callback" id="708" />
            </group>
    </tab>
    <tab title="Output Connections">
            <group layout="form" flat="false">
                <label text="Output1" />
                <combobox on-change="model.dlg.outputChange_callback" id="711" />

                <label text="Output2" />
                <combobox on-change="model.dlg.outputChange_callback" id="712" />

                <label text="Output3" />
                <combobox on-change="model.dlg.outputChange_callback" id="713" />

                <label text="Output4" />
                <combobox on-change="model.dlg.outputChange_callback" id="714" />

                <label text="Output5" />
                <combobox on-change="model.dlg.outputChange_callback" id="715" />

                <label text="Output6" />
                <combobox on-change="model.dlg.outputChange_callback" id="716" />

                <label text="Output7" />
                <combobox on-change="model.dlg.outputChange_callback" id="717" />

                <label text="Output8" />
                <combobox on-change="model.dlg.outputChange_callback" id="718" />
            </group>
    </tab>
    <tab title="More">
            <group layout="form" flat="false">
                <label text="Hide robot housing" style="* {background-color: #ccffcc}"/>
                <checkbox text="" checked="false" on-change="model.dlg.hideHousingClick_callback" id="311"/>

                <label text="Hide frame" style="* {background-color: #ccffcc}"/>
                <checkbox text="" checked="false" on-change="model.dlg.hideFrameClick_callback" id="312"/>

            </group>
    </tab>
    </tabs>
        ]]

--[[
                <label text="Conveyor 1"/>
                <combobox id="1201" on-change="model.dlg.conveyorChange_callback"> </combobox>

                <label text="Conveyor 2"/>
                <combobox id="1202" on-change="model.dlg.conveyorChange_callback"> </combobox>
--]]

        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
        model.dlg.updateAliasCombobox()
        model.dlg.refresh()

        simUI.setCurrentTab(model.dlg.ui,78,model.dlg.mainTabIndex,true)
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
        model.dlg.mainTabIndex=simUI.getCurrentTab(model.dlg.ui,78)
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
