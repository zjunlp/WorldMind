json=require("dkjson")
http = require("socket.http")
ltn12 = require("ltn12")

function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function createPalletPointsIfNeeded(objectHandle)
    local data=readPartInfo(objectHandle)
    if #data['palletPoints']==0 then
        data['palletPoints']=simBWF.generatePalletPoints(data)
    end
    writePartInfo(objectHandle,data)
end

function updatePalletPoints(objectHandle)
    local data=readPartInfo(objectHandle)
    if data['palletPattern']~=5 then
        data['palletPoints']={} -- remove them
        writePartInfo(objectHandle,data)
        createPalletPointsIfNeeded(objectHandle)
    end
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='repository'
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- all free for now
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.OLDPARTREPO)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    getDefaultInfoForNonExistingFields(data)
    return data
end

function writeInfo(data)
    if data then
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDPARTREPO,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.OLDPARTREPO,'')
    end
end

function readPartInfo(handle)
    local data=simBWF.readPartInfoV0(handle)

    -- Additional fields here:
--    if not data['palletPoints'] then
--        data['palletPoints']={}
--    end

    return data
end

function writePartInfo(handle,data)
    return simBWF.writePartInfo(handle,data)
end

function getPartTable()
    local l=sim.getObjectsInTree(originalPartHolder,sim.handle_all,1+2)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.PART)
        if data then
            data=sim.unpackTable(data)
            retL[#retL+1]={data['name']..'   ('..sim.getObjectName(l[i])..')',l[i]}
        end
    end
    return retL
end

function displayPartProperties()
    if #parts>0 then
        local h=parts[partIndex+1][2]
        local prop=readPartInfo(h)
        simUI.setEditValue(ui1,5,prop['name'],true)
        simUI.setEditValue(ui1,6,prop['destination'],true)
        simUI.setCheckboxValue(ui1,41,simBWF.getCheckboxValFromBool(sim.boolAnd32(prop['bitCoded'],1)~=0),true)
        simUI.setCheckboxValue(ui1,42,simBWF.getCheckboxValFromBool(sim.boolAnd32(prop['bitCoded'],2)~=0),true)
    end
end

function comboboxChange_callback(ui,id,newIndex)
    partIndex=newIndex
    displayPartProperties()
end

function getSpacelessString(str)
    return string.gsub(str," ","_")
end

function partName_callback(ui,id,newVal)
    if #parts>0 then
        local h=parts[partIndex+1][2]
        local prop=readPartInfo(h)
        newVal=getSpacelessString(newVal)
        if prop['name']~=newVal and #newVal>0 then
            local allNames=getAllPartNameMap()
            if allNames[newVal] then
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Duplicate naming',"A part named '"..newVal.."' already exists.")
            else
                prop['name']=newVal
                simBWF.markUndoPoint()
                writePartInfo(h,prop)
                local partTable=getPartTable()
                parts,partIndex=simBWF.populateCombobox(ui1,4,partTable,nil,newVal..'   ('..sim.getObjectName(h)..')',true,nil)
            end
        end
        displayPartProperties()
    end
end

function defaultDestination_callback(ui,id,newVal)
    if #parts>0 then
        local h=parts[partIndex+1][2]
        local prop=readPartInfo(h)
        newVal=getSpacelessString(newVal)
        if  #newVal>0 then
            prop['destination']=newVal
            writePartInfo(h,prop)
            simBWF.markUndoPoint()
        end
        displayPartProperties()
    end
end

function updateEnabledDisabledItemsDlg1()
    if ui1 then
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        local config=readInfo()

        if #parts<=0 then
            enabled=false
        end

        simUI.setEnabled(ui1,4,enabled,true)
        simUI.setEnabled(ui1,5,enabled,true)
        simUI.setEnabled(ui1,6,enabled,true)
        simUI.setEnabled(ui1,41,enabled,true)
        simUI.setEnabled(ui1,42,enabled,true)
        simUI.setEnabled(ui1,18,enabled,true)
        simUI.setEnabled(ui1,53,enabled,true)
        simUI.setEnabled(ui1,56,enabled,true)

    end
end

function invisiblePart_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writePartInfo(h,c)
end

function invisibleToOtherParts_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    c['bitCoded']=sim.boolOr32(c['bitCoded'],2)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writePartInfo(h,c)
end

function setPalletDlgItemContent()
    if palletUi then
		local h=parts[partIndex+1][2]
        local config=readPartInfo(h)
        local sel=simBWF.getSelectedEditWidget(palletUi)
        local pattern=config['palletPattern']
        simUI.setRadiobuttonValue(palletUi,101,simBWF.getRadiobuttonValFromBool(pattern==0),true)
        simUI.setRadiobuttonValue(palletUi,103,simBWF.getRadiobuttonValFromBool(pattern==2),true)
        simUI.setRadiobuttonValue(palletUi,104,simBWF.getRadiobuttonValFromBool(pattern==3),true)
        simUI.setRadiobuttonValue(palletUi,105,simBWF.getRadiobuttonValFromBool(pattern==4),true)
        simUI.setRadiobuttonValue(palletUi,106,simBWF.getRadiobuttonValFromBool(pattern==5),true)
        local circular=config['circularPatternData3']
        local off=circular[1]
        simUI.setEditValue(palletUi,3004,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,3000,simBWF.format("%.0f",circular[2]/0.001),true) -- radius
        simUI.setEditValue(palletUi,3001,simBWF.format("%.0f",circular[3]),true) -- count
        simUI.setEditValue(palletUi,3002,simBWF.format("%.0f",180*circular[4]/math.pi),true) -- angle off
        simUI.setCheckboxValue(palletUi,3003,simBWF.getCheckboxValFromBool(circular[5]),true) --center
        simUI.setEditValue(palletUi,3005,simBWF.format("%.0f",circular[6]),true) -- layers
        simUI.setEditValue(palletUi,3006,simBWF.format("%.0f",circular[7]/0.001),true) -- layer step

        local lin=config['linePatternData']
        off=lin[1]
        simUI.setEditValue(palletUi,4000,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,4001,simBWF.format("%.0f",lin[2]),true) -- rows
        simUI.setEditValue(palletUi,4002,simBWF.format("%.0f",lin[3]/0.001),true) -- row step
        simUI.setEditValue(palletUi,4003,simBWF.format("%.0f",lin[4]),true) -- cols
        simUI.setEditValue(palletUi,4004,simBWF.format("%.0f",lin[5]/0.001),true) -- col step
        simUI.setEditValue(palletUi,4005,simBWF.format("%.0f",lin[6]),true) -- layers
        simUI.setEditValue(palletUi,4006,simBWF.format("%.0f",lin[7]/0.001),true) -- layer step

        local honey=config['honeycombPatternData']
        off=honey[1]
        simUI.setEditValue(palletUi,5000,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true) --offset
        simUI.setEditValue(palletUi,5001,simBWF.format("%.0f",honey[2]),true) -- rows
        simUI.setEditValue(palletUi,5002,simBWF.format("%.0f",honey[3]/0.001),true) -- row step
        simUI.setEditValue(palletUi,5003,simBWF.format("%.0f",honey[4]),true) -- cols
        simUI.setEditValue(palletUi,5004,simBWF.format("%.0f",honey[5]/0.001),true) -- col step
        simUI.setEditValue(palletUi,5005,simBWF.format("%.0f",honey[6]),true) -- layers
        simUI.setEditValue(palletUi,5006,simBWF.format("%.0f",honey[7]/0.001),true) -- layer step
        simUI.setCheckboxValue(palletUi,5007,simBWF.getCheckboxValFromBool(honey[8]),true) -- firstRowOdd

        simUI.setEnabled(palletUi,201,(pattern==0),true)
        simUI.setEnabled(palletUi,203,(pattern==2),true)
        simUI.setEnabled(palletUi,204,(pattern==3),true)
        simUI.setEnabled(palletUi,205,(pattern==4),true)
        simUI.setEnabled(palletUi,206,(pattern==5),true)
        simBWF.setSelectedEditWidget(palletUi,sel)
    end
end

function removePart_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local p=sim.getModelProperty(h)
    if sim.boolAnd32(p,sim.modelproperty_not_model)>0 then
        sim.removeObject(h)
    else
        sim.removeModel(h)
    end
    simBWF.markUndoPoint()
    partIndex=-1
    removeDlg1() -- triggers a refresh
end

function onVisualizeCloseClicked()
    if visualizeData then
        local x,y=simUI.getPosition(visualizeData.ui)
        previousVisualizeDlgPos={x,y}
        simUI.destroy(visualizeData.ui)
        sim.removeObject(visualizeData.sensor)
        sim.removeCollection(visualizeData.collection)
        visualizeData=nil
    end
end

function updateVisualizeImage()
    if visualizeData then
    
        sim.setObjectPosition(visualizeData.sensor,visualizeData.part,{0,0,0})
        sim.setObjectOrientation(visualizeData.sensor,-1,{90*math.pi/180,0,0})
        local m=sim.getObjectMatrix(visualizeData.sensor,-1)
        m[4]=m[4]-visualizeData.params[1]*m[3]
        m[8]=m[8]-visualizeData.params[1]*m[7]
        m[12]=m[12]-visualizeData.params[1]*m[11]
        m=sim.rotateAroundAxis(m,{1,0,0},sim.getObjectPosition(visualizeData.part,-1),visualizeData.params[3])
        m=sim.rotateAroundAxis(m,{0,0,1},sim.getObjectPosition(visualizeData.part,-1),visualizeData.params[2])
        sim.setObjectMatrix(visualizeData.sensor,-1,m)
        sim.setModelProperty(visualizeData.part,0)
        sim.handleVisionSensor(visualizeData.sensor)
        sim.setModelProperty(visualizeData.part,sim.modelproperty_not_visible+sim.modelproperty_not_renderable+sim.modelproperty_not_showasinsidemodel)
        local img,x,y=sim.getVisionSensorCharImage(visualizeData.sensor)
        simUI.setImageData(visualizeData.ui,1,img,x,y)
    end
end

function onVisualizeZoomInClicked()
    if visualizeData.params[1]>0.03 then 
        visualizeData.params[1]=visualizeData.params[1]-0.04
    end
end

function onVisualizeZoomOutClicked()
    if visualizeData.params[1]<1 then 
        visualizeData.params[1]=visualizeData.params[1]+0.04
    end
end

function onVisualizeRotLeftClicked()
    visualizeData.params[2]=visualizeData.params[2]-0.1745
end

function onVisualizeRotRightClicked()
    visualizeData.params[2]=visualizeData.params[2]+0.1745
end

function onVisualizeRotUpClicked()
    if visualizeData.params[3]<1.3964 then 
        visualizeData.params[3]=visualizeData.params[3]+0.1745
    end
end

function onVisualizeRotDownClicked()
    if visualizeData.params[3]>0.169 then 
        visualizeData.params[3]=visualizeData.params[3]-0.1745
    end
end

function visualizePart_callback(ui,id,newVal)
    if not visualizeData then
        local xml =[[
            <image width="512" height="512" id="1"/>
            
            <group layout="hbox" flat="true">
                <group layout="vbox" flat="true">
                    <button text="Zoom in"  on-click="onVisualizeZoomInClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330"/>
                    <button text="Zoom out"  on-click="onVisualizeZoomOutClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330" />
                </group>
                <group layout="hbox" flat="true">
                    <button text="Rotate left"  on-click="onVisualizeRotLeftClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330" />
                    <button text="Rotate right"  on-click="onVisualizeRotRightClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330" />
                </group>
                <group layout="vbox" flat="true">
                    <button text="Rotate up"  on-click="onVisualizeRotUpClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330" />
                    <button text="Rotate down"  on-click="onVisualizeRotDownClicked" autoRepeat="true" autoRepeatDelay="500" autoRepeatInterval="330" />
                </group>
            </group>
            <button text="OK"  on-click="onVisualizeCloseClicked" />
        ]]
        visualizeData={}
        visualizeData.ui=simBWF.createCustomUi(xml,"Part and pallet visualization",previousVisualizeDlgPos and previousVisualizeDlgPos or 'center',true,'onVisualizeCloseClicked',true--[[,activate,additionalUiAttribute--]])
        visualizeData.sensor=sim.createVisionSensor(1+2+128,{512,512,0,0},{0.001,5,60*math.pi/180,0.1,0.1,0.1,0.4,0.5,0.5,0,0})
        sim.setObjectInt32Parameter(visualizeData.sensor,sim.objintparam_visibility_layer ,0)
        local p=sim.boolOr32(sim.getObjectProperty(visualizeData.sensor),sim.objectproperty_dontshowasinsidemodel)
        sim.setObjectProperty(visualizeData.sensor,p)

        local part=parts[partIndex+1][2]
        local info=readPartInfo(part)
        if sim.boolAnd32(sim.getModelProperty(part),sim.modelproperty_not_model)>0 then
            part=sim.copyPasteObjects({part},0)
            part=part[1]
            sim.setObjectInt32Parameter(part,sim.objintparam_visibility_layer,1)
            sim.setObjectSpecialProperty(part,sim.objectspecialproperty_renderable+sim.objectspecialproperty_detectable_all)
            local prop=sim.boolOr32(sim.getObjectProperty(part),sim.objectproperty_dontshowasinsidemodel)-sim.objectproperty_dontshowasinsidemodel
            sim.setObjectProperty(part,prop)
            sim.setModelProperty(part,sim.modelproperty_not_visible+sim.modelproperty_not_renderable+sim.modelproperty_not_showasinsidemodel) -- makes it a model
        else
            part=sim.copyPasteObjects({part},1)
            part=part[1]
            sim.setModelProperty(part,sim.modelproperty_not_visible+sim.modelproperty_not_renderable+sim.modelproperty_not_showasinsidemodel)
        end
        visualizeData.part=part
        visualizeData.collection=sim.createCollection('',0)
        sim.addObjectToCollection(visualizeData.collection,part,sim.handle_tree,0)
        sim.setObjectInt32Parameter(visualizeData.sensor,sim.visionintparam_entity_to_render,visualizeData.collection)
        sim.setObjectParent(visualizeData.part,functionalPartHolder,true)
        sim.setObjectParent(visualizeData.sensor,visualizeData.part,true)
        visualizeData.params={0.35,math.pi/4,math.pi/4}
        
        if #info['palletPoints']>0 then
            -- Check the detection point (for the z-position of the pallet):
            local res,bbMax=sim.getObjectFloatParameter(visualizeData.part,sim.objfloatparam_modelbbox_max_z)
            sim.setObjectPosition(proxSensor,visualizeData.part,{0,0,bbMax*1.001})
            sim.setObjectOrientation(proxSensor,visualizeData.part,{math.pi,0,0})
            local shapes=sim.getObjectsInTree(visualizeData.part,sim.object_shape_type,0)
            local zMin=1
            for i=1,#shapes,1 do
                if sim.boolAnd32(sim.getObjectSpecialProperty(shapes[i]),sim.objectspecialproperty_detectable_all)>0 then
                    local r,dist=sim.checkProximitySensor(proxSensor,shapes[i])
                    if r>0 and dist<zMin then
                        zMin=dist
                    end
                end
            end
            
            -- Now the pallet:
            for i=1,#info['palletPoints'],1 do
                local plpt=info['palletPoints'][i]
                local h=sim.createPureShape(0,4+16,{0.01,0.01,0.01},0.1,nil)
                sim.setShapeColor(h,nil,sim.colorcomponent_ambient_diffuse,{1,0,1})
                sim.setObjectSpecialProperty(h,sim.objectspecialproperty_renderable)
                sim.setObjectPosition(h,visualizeData.part,{plpt['pos'][1],plpt['pos'][2],plpt['pos'][3]+bbMax*1.001-zMin})
                sim.setObjectParent(h,visualizeData.part,true)
            end
        end


        updateVisualizeImage()
    end
end

function palletCreation_callback(ui,id,newVal)
    createPalletDlg()
end

function circularPattern_offsetChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['circularPatternData3'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end

function circularPattern_radiusChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.5 then v=0.5 end
        if v~=c['circularPatternData3'][2] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][2]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_angleOffsetChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        if v<-359 then v=-359 end
        if v>359 then v=359 end
        v=v*math.pi/180
        if v~=c['circularPatternData3'][4] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][4]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_countChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>40 then v=40 end
        if v~=c['circularPatternData3'][3] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][3]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_layersChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['circularPatternData3'][6] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][6]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_layerStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['circularPatternData3'][7] then
            simBWF.markUndoPoint()
            c['circularPatternData3'][7]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function circularPattern_centerChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    c['circularPatternData3'][5]=(newVal~=0)
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end


function linePattern_offsetChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['linePatternData'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end

function linePattern_rowsChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][2] then
            simBWF.markUndoPoint()
            c['linePatternData'][2]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_rowStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][3] then
            simBWF.markUndoPoint()
            c['linePatternData'][3]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_colsChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][4] then
            simBWF.markUndoPoint()
            c['linePatternData'][4]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_colStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][5] then
            simBWF.markUndoPoint()
            c['linePatternData'][5]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_layersChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['linePatternData'][6] then
            simBWF.markUndoPoint()
            c['linePatternData'][6]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function linePattern_layerStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['linePatternData'][7] then
            simBWF.markUndoPoint()
            c['linePatternData'][7]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end




function honeyPattern_offsetChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local i=1
    local t={0,0,0}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if t[i]>0.2 then t[i]=0.2 end
        if t[i]<-0.2 then t[i]=-0.2 end
        i=i+1
    end
    c['honeycombPatternData'][1]={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end

function honeyPattern_rowsChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][2] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][2]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_rowStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][3] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][3]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_colsChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<2 then v=2 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][4] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][4]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_colStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][5] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][5]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_layersChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<1 then v=1 end
        if v>10 then v=10 end
        if v~=c['honeycombPatternData'][6] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][6]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_layerStepChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.01 then v=0.01 end
        if v>0.2 then v=0.2 end
        if v~=c['honeycombPatternData'][7] then
            simBWF.markUndoPoint()
            c['honeycombPatternData'][7]=v
            writePartInfo(h,c)
        end
    end
    setPalletDlgItemContent()
end

function honeyPattern_rowIsOddChange_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    c['honeycombPatternData'][8]=(newVal~=0)
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end

function editPatternItems_callback(ui,id,newVal)
    local h=parts[partIndex+1][2]
    local prop=readPartInfo(h)
    local s="600 400"
    local p="200 200"
    if customPalletDlgSize then
        s=customPalletDlgSize[1]..' '..customPalletDlgSize[2]
    end
    if customPalletDlgPos then
        p=customPalletDlgPos[1]..' '..customPalletDlgPos[2]
    end
    local xml = [[ <editor title="Pallet points" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            
    local initialText=simBWF.palletPointsToString(prop['palletPoints'])

    initialText=initialText.."\n\n--[[".."\n\nFormat as in following example:\n\n"..[[
{{pt1X,pt1Y,pt1Z},{pt1Alpha,pt1Beta,pt1Gamma},pt1Layer},
{{pt2X,pt2Y,pt2Z},{pt2Alpha,pt2Beta,pt2Gamma},pt2Layer}]].."\n\n--]]"

    local modifiedText
    while true do
        modifiedText,customPalletDlgSize,customPalletDlgPos=sim.openTextEditor(initialText,xml)
        local newPalletPoints=simBWF.stringToPalletPoints(modifiedText)
        if newPalletPoints then
            if not simBWF.arePalletPointsSame_posOrientAndLayer(newPalletPoints,prop['palletPoints']) then
                prop['palletPoints']=newPalletPoints
                writePartInfo(h,prop)
                simBWF.markUndoPoint()
            end
            break
        else
            if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_yesno,'Input Error',"The input is not formated correctly. Do you wish to discard the changes?") then
                break
            end
            initialText=modifiedText
        end
    end
end

function importPallet_callback(ui,id,newVal)
    local file=sim.fileDialog(sim.filedlg_type_load,'Loading pallet items','','','pallet items','txt')
    if file then
        local newPalletPoints=simBWF.readPalletFromFile(file)
        if newPalletPoints then
            local h=parts[partIndex+1][2]
            local prop=readPartInfo(h)
            prop['palletPoints']=newPalletPoints
            writePartInfo(h,prop)
            simBWF.markUndoPoint()
        else
            sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'File Read Error',"The specified file could not be read.")
        end
    end
end

function patternTypeClick_callback(ui,id)
    local h=parts[partIndex+1][2]
    local c=readPartInfo(h)
    local changed=(c['palletPattern']~=id-101)
    c['palletPattern']=id-101
--    if c['palletPattern']==5 and changed then
--        c['palletPoints']={} -- clear the pallet points when we select 'imported'
--    end
    simBWF.markUndoPoint()
    writePartInfo(h,c)
    setPalletDlgItemContent()
end

function onPalletCloseClicked()
    if palletUi then
        local x,y=simUI.getPosition(palletUi)
        previousPalletDlgPos={x,y}
        simUI.destroy(palletUi)
        palletUi=nil
        local h=parts[partIndex+1][2]
        updatePalletPoints(h)
    end
end

function createPalletDlg()
    if not palletUi then
        local xml =[[
    <tabs id="77">
            <tab title="None">
            <radiobutton text="Do not create a pallet" on-click="patternTypeClick_callback" id="101" />
            <group layout="form" flat="true" id="201">
            </group>
                <label text="" style="* {margin-left: 380px;}"/>
            </tab>

            <tab title="Circular type">
            <radiobutton text="Create a pallet with items arranged in a circular pattern" on-click="patternTypeClick_callback" id="103" />
            <group layout="form" flat="true"  id="203">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="circularPattern_offsetChange_callback" id="3004"/>

                <label text="Items on circumference"/>
                <edit on-editing-finished="circularPattern_countChange_callback" id="3001"/>

                <label text="Angle offset (deg)"/>
                <edit on-editing-finished="circularPattern_angleOffsetChange_callback" id="3002"/>

                <label text="Radius (mm)"/>
                <edit on-editing-finished="circularPattern_radiusChange_callback" id="3000"/>

                <label text="Center in use"/>
                <checkbox text="" on-change="circularPattern_centerChange_callback" id="3003" />

                <label text="Layers"/>
                <edit on-editing-finished="circularPattern_layersChange_callback" id="3005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="circularPattern_layerStepChange_callback" id="3006"/>
            </group>
            </tab>

            <tab title="Line type">
            <radiobutton text="Create a pallet with items arranged in a rectangular pattern" on-click="patternTypeClick_callback" id="104" />
            <group layout="form" flat="true"  id="204">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="linePattern_offsetChange_callback" id="4000"/>

                <label text="Rows"/>
                <edit on-editing-finished="linePattern_rowsChange_callback" id="4001"/>

                <label text="Row step (mm)"/>
                <edit on-editing-finished="linePattern_rowStepChange_callback" id="4002"/>

                <label text="Columns"/>
                <edit on-editing-finished="linePattern_colsChange_callback" id="4003"/>

                <label text="Columns step (mm)"/>
                <edit on-editing-finished="linePattern_colStepChange_callback" id="4004"/>

                <label text="Layers"/>
                <edit on-editing-finished="linePattern_layersChange_callback" id="4005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="linePattern_layerStepChange_callback" id="4006"/>
            </group>
            </tab>

            <tab title="Honeycomb type">
            <radiobutton text="Create a pallet with items arranged in a honeycomb pattern" on-click="patternTypeClick_callback" id="105" />
            <group layout="form" flat="true"  id="205">
                <label text="Offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="honeyPattern_offsetChange_callback" id="5000"/>

                <label text="Rows (longest)"/>
                <edit on-editing-finished="honeyPattern_rowsChange_callback" id="5001"/>

                <label text="Row step (mm)"/>
                <edit on-editing-finished="honeyPattern_rowStepChange_callback" id="5002"/>

                <label text="Columns"/>
                <edit on-editing-finished="honeyPattern_colsChange_callback" id="5003"/>

                <label text="Columns step (mm)"/>
                <edit on-editing-finished="honeyPattern_colStepChange_callback" id="5004"/>

                <label text="Layers"/>
                <edit on-editing-finished="honeyPattern_layersChange_callback" id="5005"/>

                <label text="Layer step (mm)"/>
                <edit on-editing-finished="honeyPattern_layerStepChange_callback" id="5006"/>

                <label text="1st row is odd"/>
                <checkbox text="" on-change="honeyPattern_rowIsOddChange_callback" id="5007" />
            </group>
            </tab>

            <tab title="Custom/imported">
            <radiobutton text="Create a pallet with items arranged in a customized pattern" on-click="patternTypeClick_callback" id="106" />
            <group layout="vbox" flat="true"  id="206">
                <button text="Edit pallet items"  on-click="editPatternItems_callback"  id="6000"/>
                <button text="Import pallet items"  on-click="importPallet_callback"  id="6001"/>
                <label text="" style="* {margin-left: 380px;}"/>
            </group>
            </tab>

            </tabs>
        ]]

        palletUi=simBWF.createCustomUi(xml,"Pallet Creation",'center',true,'onPalletCloseClicked',true--[[,resizable,activate,additionalUiAttribute--]])

        setPalletDlgItemContent()
        local h=parts[partIndex+1][2]
        local c=readPartInfo(h)
        local pattern=c['palletPattern']
        local pat={}
        pat[0]=0
        pat[2]=1
        pat[3]=2
        pat[4]=3
        pat[5]=4
        simUI.setCurrentTab(palletUi,77,pat[pattern],true)
    end
end

function createDlg1()
    if (not ui1) and simBWF.canOpenPropertyDialog() then
        local xml =[[
                <combobox id="4" on-change="comboboxChange_callback"> </combobox>

            <group layout="form" flat="true">

                <label text="Name"/>
                <edit on-editing-finished="partName_callback" id="5"/>

                <label text="Default destination"/>
                <edit on-editing-finished="defaultDestination_callback" id="6"/>

                <label text="Pallet creation"/>
                <button text="Adjust"  on-click="palletCreation_callback" id="18" />

                <label text="Invisible"/>
                <checkbox text="" on-change="invisiblePart_callback" id="41" />

                <label text="Invisible to other parts"/>
                <checkbox text="" on-change="invisibleToOtherParts_callback" id="42" />
            </group>

            <group layout="hbox" flat="true">
                <button text="Visualize part and pallet"  on-click="visualizePart_callback" id="56" />
                <button text="Remove part"  on-click="removePart_callback" id="53" />
            </group>
        ]]


        ui1=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlg1Pos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

        local previousItemName=nil
        if parts and #parts>0 and (partIndex>=0) then
            previousItemName=parts[partIndex+1][1]..'   ('..sim.getObjectName(parts[partIndex+1][2])..')'
        end
        local partTable=getPartTable()
        
        parts,partIndex=simBWF.populateCombobox(ui1,4,partTable,nil,previousItemName,true,nil)

        displayPartProperties()
        updateEnabledDisabledItemsDlg1()
    end
end

function showDlg1()
    if not ui1 then
        createDlg1()
    end
end

function removeDlg1()
    if ui1 then
        local x,y=simUI.getPosition(ui1)
        previousDlg1Pos={x,y}
        simUI.destroy(ui1)
        ui1=nil
    end
end

function getPotentialNewParts()
    local p=sim.getObjectsInTree(model,sim.handle_all,1+2)
    local i=1
    while i<=#p do
        if p[i]==functionalPartHolder then
            table.remove(p,i)
        else
            i=i+1
        end
    end
    return p
end

function getAllPartNameMap()
    local allNames={}
    local parts=sim.getObjectsInTree(originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        local info=readPartInfo(parts[i])
        local nm=info['name']
        allNames[nm]=parts[i]
    end
    return allNames
end

function resolveDuplicateNames()
    local allNames={}
    local parts=sim.getObjectsInTree(originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        local info=readPartInfo(parts[i])
        local nm=info['name']
        if nm=='<partName>' then
            nm=sim.getObjectName(parts[i])
        end
        while allNames[nm] do
            nm=nm..'_COPY'
        end
        allNames[nm]=true
        info['name']=nm
        writePartInfo(parts[i],info)
    end
end

function sysCall_init()
    partToEdit=-1
    lastT=sim.getSystemTimeInMs(-1)
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    originalPartHolder=sim.getObjectHandle('partRepository_modelParts')
    functionalPartHolder=sim.getObjectHandle('partRepository_functional')
    proxSensor=sim.getObjectHandle('partRepository_sensor')
	

    
    -- Following because of a bug in V-REP V3.3.3 and before:
    local p=sim.boolOr32(sim.getModelProperty(originalPartHolder),sim.modelproperty_scripts_inactive)
    if sim.getInt32Parameter(sim.intparam_program_version)>30303 then
        sim.setModelProperty(originalPartHolder,p)
    else
        sim.setModelProperty(originalPartHolder,p-sim.modelproperty_scripts_inactive)
    end
    
    -- Following for backward compatibility:
    local parts=sim.getObjectsInTree(originalPartHolder,sim.handle_all,1+2)
    for i=1,#parts,1 do
        createPalletPointsIfNeeded(parts[i])
    end
    
    -- Following for backward compatibility:
    resolveDuplicateNames()

    sim.setIntegerSignal('__brUndoPointCounter__',0)
    previousUndoPointCounter=0
    undoPointStayedSameCounter=-1
    
    previousPalletDlgPos,algoDlgSize,algoDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")

    -- Allow only one part repository per scene:
    local objs=sim.getObjectsWithTag(simBWF.modelTags.OLDPARTREPO,true)
    if #objs>1 then
        sim.removeModel(model)
        sim.removeObjectFromSelection(sim.handle_all)
        objs=sim.getObjectsWithTag(simBWF.modelTags.OLDPARTREPO,true)
        sim.addObjectToSelection(sim.handle_single,objs[1])
    else
        updatePluginRepresentation()
    end
end

showOrHideUi1IfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg1()
    else
        removeDlg1()
    end
end

removeAssociatedCustomizationScriptIfAvailable=function(h)
    local sh=sim.getCustomizationScriptAssociatedWithObject(h)
    if sh>0 then
        sim.removeScript(sh)
    end
end

checkPotentialNewParts=function(potentialParts)
    local retVal=false -- true means update the part list in the dialog (i.e. rebuild the dialog's part combo)
    local functionType=0 -- 0=question, 1=make parts, 2=make orphans
    for modC=1,#potentialParts,1 do
        local h=potentialParts[modC]
        local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
        if not data then
            -- This is not yet flagged as part
            simBWF.markUndoPoint()
            if functionType==0 then
                local msg="Detected new children of object '"..sim.getObjectName(model).."'. Objects attached to that object should be repository parts. Do you wish to turn those new objects into repository parts? If you click 'no', then those new objects will be made orphan. If you click 'yes', then those new objects will be adjusted appropriately. Only shapes or models can be turned into repository parts." 
                local ret=sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Part Definition',msg)
                if ret==sim.msgbox_return_yes then
                    functionType=1
                else
                    functionType=2
                end
            end
            if functionType==1 then
                -- We want to accept it as a part
                local allNames=getAllPartNameMap()
                data=readPartInfo(h)
                local nm=sim.getObjectName(h)
                while true do
                    if not allNames[nm] then
                        data['name']=nm -- ok, that name doesn't exist yet!
                        break
                    end
                    nm=nm..'_COPY'
                end
                writePartInfo(h,data) -- attach the XYZ_FEEDERPART_INFO tag
                sim.setObjectPosition(h,model,{0,0,0}) -- keep the orientation as it is

                if sim.boolAnd32(sim.getModelProperty(h),sim.modelproperty_not_model)>0 then
                    -- Shape
                    local p=sim.boolOr32(sim.getObjectProperty(h),sim.objectproperty_dontshowasinsidemodel)
                    sim.setObjectProperty(h,p)
                else
                    -- Model
                    local p=sim.boolOr32(sim.getModelProperty(h),sim.modelproperty_not_showasinsidemodel)
                    sim.setModelProperty(h,p)
                end
                createPalletPointsIfNeeded(h)
                removeAssociatedCustomizationScriptIfAvailable(h)
                sim.setObjectParent(h,originalPartHolder,true)
                retVal=true
            end
            if functionType==2 then
                -- We reject it as a part
                sim.setObjectParent(h,-1,true)
            end
        else
            -- This is already flagged as part
            data=readPartInfo(h)
            
            local allNames=getAllPartNameMap()
            local nm=data['name']
            while true do
                if not allNames[nm] then
                    data['name']=nm -- ok, that name doesn't exist yet!
                    break
                end
                nm=nm..'_COPY'
            end
            writePartInfo(h,data) -- append additional tags that were maybe missing previously
            -- just in case we are adding an item that was already tagged previously
            sim.setObjectPosition(h,model,{0,0,0}) -- keep the orientation as it is
            -- Make the model static, non-respondable, non-collidable, non-measurable, non-visible, etc.
            if sim.boolAnd32(sim.getModelProperty(h),sim.modelproperty_not_model)>0 then
                -- Shape
                local p=sim.boolOr32(sim.getObjectProperty(h),sim.objectproperty_dontshowasinsidemodel)
                sim.setObjectProperty(h,p)
            else
                -- Model
                local p=sim.boolOr32(sim.getModelProperty(h),sim.modelproperty_not_showasinsidemodel)
                sim.setModelProperty(h,p)
            end
            createPalletPointsIfNeeded(h)
            removeAssociatedCustomizationScriptIfAvailable(h)
            sim.setObjectParent(h,originalPartHolder,true)
            retVal=true
        end
    end
    return retVal
end

function sysCall_nonSimulation()
    showOrHideUi1IfNeeded()
    updateVisualizeImage()
    -- Following is the central part where we set undo points:
    ---------------------------------
    local cnt=sim.getIntegerSignal('__brUndoPointCounter__')
    if cnt~=previousUndoPointCounter then
        undoPointStayedSameCounter=8
        previousUndoPointCounter=cnt
    end
    if undoPointStayedSameCounter>0 then
        undoPointStayedSameCounter=undoPointStayedSameCounter-1
    else
        if undoPointStayedSameCounter==0 then
            sim.announceSceneContentChange() -- to have an undo point
            undoPointStayedSameCounter=-1
        end
    end
    ---------------------------------
    
    if sim.getSystemTimeInMs(lastT)>3000 then
        lastT=sim.getSystemTimeInMs(-1)
        local potentialNewParts=getPotentialNewParts()
        if #potentialNewParts>0 then
            if checkPotentialNewParts(potentialNewParts) then
                removeDlg1() -- we need to update the dialog with the new parts
            end
        end
    end

    pricingRequest_executeIfNeeded()
end

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,1)
end

function sysCall_beforeSimulation()
    sim.setObjectInt32Parameter(model,sim.objintparam_visibility_layer,0)
    removeDlg1()
end

function sysCall_beforeInstanceSwitch()
    removeDlg1()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sendPricingRequest(payload)
    local path = "http://service.blueworkforce.com/public_html/generate/report"
    local response_body = { }
    http.TIMEOUT=5 -- default is 60

    local res, code, response_headers, response_status_line = http.request
    {
        url = path,
        method = "POST",
        headers =
        {
          ["Content-Type"] = "application/json",
          ["Content-Length"] = payload:len()
        },
        source = ltn12.source.string(payload),
        sink = ltn12.sink.table(response_body)
    }
    return res,code,response_status_line,table.concat(response_body)
end

function pricingRequest_executeIfNeeded()
    if pricingRequest then
        if pricingRequest.counter>0 then
            pricingRequest.counter=pricingRequest.counter-1
        else
            local res,code,response_status_line,data=sendPricingRequest(pricingRequest.payload)
--            sim.auxiliaryConsoleClose(pricingRequest.requestAuxConsole)
            if res and code==200 then
                local aux=sim.auxiliaryConsoleOpen('Pricing reply',500,4,{600,100},{800,800},nil,{0.95,1,0.95})
                sim.auxiliaryConsolePrint(aux,data)
            else
                -- code contains the error msg if res is nil. Otherwise, it contains a status code
                local msg="Failed to retrieve the pricing information.\n"
                if not res then
                    msg=msg.."Status code is: "..code
                else
                    msg=msg.."Error message is: "..res
                end
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Pricing inquiry",msg)
            end
            simUI.destroy(pricingRequest.ui)
            pricingRequest=nil
        end
    end
end

function pricing_callback()
    if not pricingRequest then
        local objects={}
        local tags={simBWF.modelTags.RAGNAR,simBWF.modelTags.RAGNARGRIPPER,simBWF.modelTags.OLDLOCATION,simBWF.modelTags.TRACKINGWINDOW,"XYZ_STATICPICKWINDOW_INFO","XYZ_DETECTIONWINDOW_INFO",simBWF.modelTags.CONVEYOR}
        for i=1,#tags,1 do
            local obj=sim.getObjectsWithTag(tags[i],true)
            for j=1,#obj,1 do
                local ob=sim.callScriptFunction('ext_getItemData_pricing@'..sim.getObjectName(obj[j]),sim.scripttype_customizationscript)
                objects[#objects+1]=ob
            end
        end

    
        pricingRequest={}
        pricingRequest.payload=json.encode(objects,{indent=true})

--[=[        
        -- Testing:
        pricingRequest.payload = [[ {   "version":1,	
                                        "robot":"Ragnar",
                                        "gripper":"fcm",
                                        "frame":"experimental",
                                        "exterior":"wd",
                                        "motors":"standard",  
                                        "primary_arms":"250", 
                                        "secondary_arms":"500", 
                                        "software":"load sharing"
                                    } ]]
--]=]        
        pricingRequest.requestAuxConsole=sim.auxiliaryConsoleOpen('Pricing request',500,4,{100,100},{800,800},nil,{1,0.95,0.95})
        sim.auxiliaryConsolePrint(pricingRequest.requestAuxConsole,pricingRequest.payload)
        local xml =[[
                <label text="Please wait a few seconds..."  style="* {qproperty-alignment: AlignCenter; min-width: 300px; min-height: 100px;}"/>
        ]]
        pricingRequest.ui=simBWF.createCustomUi(xml,'Pricing request','center',false,nil,true,false,false)
        pricingRequest.counter=3
    end
end


function sysCall_br() 
 --   pricing_callback()
end


function sysCall_cleanup()
    removeDlg1()
    removeFromPluginRepresentation()
    if sim.isHandleValid(model)==1 then
        -- The associated model might already have been destroyed
        simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousPalletDlgPos,algoDlgSize,algoDlgPos,previousDlg1Pos)
    end
end
