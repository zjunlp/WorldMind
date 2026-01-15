function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function setObjectSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    if z then
        sim.scaleObject(h,x/sx,y/sy,z/sz)
    else
        sim.scaleObject(h,x/sx,y/sy,1) -- for the labels
    end
end

function setCuboidMassAndInertia(h,sizeX,sizeY,sizeZ,mass,inertiaFact)
    local inertiaFact=1
    local transf=sim.getObjectMatrix(h,-1)
    local inertia={(sizeY*sizeY+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeX*sizeX+sizeZ*sizeZ)*mass*inertiaFact/12,0,0,0,(sizeY*sizeY+sizeX*sizeX)*mass*inertiaFact/12}
    sim.setShapeMassAndInertia(h,mass,inertia,{0,0,0},transf)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='shippingBox'
    end
    if not info['width'] then
        info['width']=0.3
    end
    if not info['length'] then
        info['length']=0.3
    end
    if not info['height'] then
        info['height']=0.3
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- 1,2: free, 4:textured
    end
    if not info['mass'] then
        info['mass']=0.5
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_SHIPPINGBOX_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_SHIPPINGBOX_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_SHIPPINGBOX_INFO','')
    end
end

function readPartInfo()
    local data=simBWF.readPartInfoV0(model)
    -- Additional part data (label specific):
    if not data['labelData'] then
        data['labelData']={}
    end
    if not data['labelData']['bitCoded'] then
        data['labelData']['bitCoded']=0 -- 1,2,4: free, 8=label1,16=label2,32=label3,64=largeLabel1,128=largeLabel2,256=largeLabel3
    end
    if not data['labelData']['placementCode'] then
        data['labelData']['placementCode']={'{0,0,0},{0,0,0}','{0,0,0},{0,0,0}','{0,0,0},{0,0,0}'}
    end
    if not data['labelData']['smallLabelSize'] then
        data['labelData']['smallLabelSize']={0.075,0.038}
    end
    if not data['labelData']['largeLabelSize'] then
        data['labelData']['largeLabelSize']={0.075,0.1125}
    end
    if not data['labelData']['boxSize'] then
        data['labelData']['boxSize']={0.1,0.1,0.1}
    end
    return data
end

function writePartInfo(data)
    return simBWF.writePartInfo(model,data)
end

function setColor(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
end

function getColor()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    return rgb[1],rgb[2],rgb[3]
end

function updateModel()
    local c=readInfo()
    local partInfo=readPartInfo()
    local partLabelC=partInfo['labelData']
    local w=c['width']
    local l=c['length']
    local h=c['height']
    local bitCText=c['bitCoded']
    local bitC=partLabelC['bitCoded']
    local mass=c['mass']
    local boxSize={w,l,h}
    local smallLabelSize=partLabelC['smallLabelSize']
    local largeLabelSize=partLabelC['largeLabelSize']

    partLabelC['boxSize']={w,l,h}
    partInfo['labelData']=partLabelC
    writePartInfo(partInfo)

    setObjectSize(model,w,l,h)
    setCuboidMassAndInertia(model,w,l,h,mass)

    setObjectSize(smallLabel,smallLabelSize[1],smallLabelSize[2])
    -- Scale also the texture:
    sim.setObjectFloatParameter(smallLabel,sim.shapefloatparam_texture_scaling_y,0.11*smallLabelSize[1]/0.075)
    sim.setObjectFloatParameter(smallLabel,sim.shapefloatparam_texture_scaling_x,0.11*smallLabelSize[2]/0.0375)
    sim.setObjectFloatParameter(smallLabel,sim.shapefloatparam_texture_y,0.037*smallLabelSize[2]/0.0375)

    setObjectSize(largeLabel,largeLabelSize[1],largeLabelSize[2])
    -- Scale also the texture:
    sim.setObjectFloatParameter(largeLabel,sim.shapefloatparam_texture_scaling_y,0.11*largeLabelSize[1]/0.075)
    sim.setObjectFloatParameter(largeLabel,sim.shapefloatparam_texture_scaling_x,0.11*largeLabelSize[2]/0.1125)

    local textureId=sim.getShapeTextureId(texture)

    if sim.boolAnd32(bitCText,4)>0 then
        -- textured
        sim.setShapeTexture(model,textureId,sim.texturemap_cube,4+8,{0.3,0.3})
    else
        -- without texture
        sim.setShapeTexture(model,-1,sim.texturemap_cube,4+8,{0.3,0.3})
    end
    
    -- Now the labels:

    -- Remove the current labels:
    local objs=sim.getObjectsInTree(model,sim.handle_all,1)
    for i=1,#objs,1 do
        local h=objs[i]
        if h~=texture and h~=smallLabel and h~=largeLabel then
            sim.removeObject(h)
        end
    end
    
    -- Now process the 3 potential labels:
    for labelInd=1,3,1 do
        if sim.boolAnd32(bitC,8*(2^(labelInd-1)))>0 then
            local useLargeLabel=(sim.boolAnd32(bitC,64*(2^(labelInd-1)))>0)
            local labelSize=smallLabelSize
            local modelLabelHandle=smallLabel
            if useLargeLabel then
                labelSize=largeLabelSize
                modelLabelHandle=largeLabel
            end
            local h=sim.copyPasteObjects({modelLabelHandle},0)[1]
            sim.setObjectParent(h,model,true)
            sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,255) -- make it visible
            sim.setObjectSpecialProperty(h,sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable) -- make it collidable, measurable, detectable, etc.
            local code=partLabelC['placementCode'][labelInd]
            local toExecute='local boxSizeX='..boxSize[1]..'\n'
            toExecute=toExecute..'local boxSizeY='..boxSize[2]..'\n'
            toExecute=toExecute..'local boxSizeZ='..boxSize[3]..'\n'
            toExecute=toExecute..'local labelSizeX='..labelSize[1]..'\n'
            toExecute=toExecute..'local labelSizeY='..labelSize[2]..'\n'
            toExecute=toExecute..'local labelRadius='..(0.5*math.sqrt(labelSize[1]*labelSize[1]+labelSize[2]*labelSize[2]))..'\n'

            toExecute=toExecute..'return {'..code..'}'
            local res,theTable=sim.executeLuaCode(toExecute)
            sim.setObjectPosition(h,model,theTable[1])
            sim.setObjectOrientation(h,model,theTable[2])
            local labelData={}
            labelData['labelIndex']=labelInd
            sim.writeCustomDataBlock(h,'XYZ_PARTLABEL_INFO',sim.packTable(labelData))
        end
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local partInfo=readPartInfo()
        local partConf=partInfo['labelData']
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['height']/0.001),true)

        local size=partConf['smallLabelSize']
        simUI.setEditValue(ui,50,simBWF.format("%.0f , %.0f",size[1]*1000,size[2]*1000),true)
        local size=partConf['largeLabelSize']
        simUI.setEditValue(ui,51,simBWF.format("%.0f , %.0f",size[1]*1000,size[2]*1000),true)

        simUI.setCheckboxValue(ui,40,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],8)~=0),true)
        simUI.setCheckboxValue(ui,43,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],16)~=0),true)
        simUI.setCheckboxValue(ui,46,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],32)~=0),true)

        simUI.setCheckboxValue(ui,41,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],64)~=0),true)
        simUI.setCheckboxValue(ui,44,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],128)~=0),true)
        simUI.setCheckboxValue(ui,47,simBWF.getCheckboxValFromBool(sim.boolAnd32(partConf['bitCoded'],256)~=0),true)
        simUI.setCheckboxValue(ui,888,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],4)~=0),true)
        simUI.setEditValue(ui,20,simBWF.format("%.2f",config['mass']),true)
        local red,green,blue=getColor()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)

        simUI.setEnabled(ui,41,sim.boolAnd32(partConf['bitCoded'],8)~=0,true)
        simUI.setEnabled(ui,42,sim.boolAnd32(partConf['bitCoded'],8)~=0,true)

        simUI.setEnabled(ui,44,sim.boolAnd32(partConf['bitCoded'],16)~=0,true)
        simUI.setEnabled(ui,45,sim.boolAnd32(partConf['bitCoded'],16)~=0,true)

        simUI.setEnabled(ui,47,sim.boolAnd32(partConf['bitCoded'],32)~=0,true)
        simUI.setEnabled(ui,48,sim.boolAnd32(partConf['bitCoded'],32)~=0,true)

        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function widthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>2 then v=2 end
        if v~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function lengthChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>2 then v=2 end
        if v~=c['length'] then
            simBWF.markUndoPoint()
            c['length']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.05 then v=0.05 end
        if v>2 then v=2 end
        if v~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function massChange_callback(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        if v<0.01 then v=0.01 end
        if v>10 then v=10 end
        if v~=c['mass'] then
            simBWF.markUndoPoint()
            c['mass']=v
            writeInfo(c)
            updateModel()
        end
    end
    setDlgItemContent()
end

function redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(newVal/100,g,b)
end

function greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(r,newVal/100,b)
end

function blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b=getColor()
    setColor(r,g,newVal/100)
end

function texture_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],4)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    updateModel()
    setDlgItemContent()
end

function label1_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],8)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-8
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function label2_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],16)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-16
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function label3_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],32)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-32
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function largeLabel1_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function largeLabel2_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],128)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-128
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function largeLabel3_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    c['bitCoded']=sim.boolOr32(c['bitCoded'],256)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-256
    end
    inf['labelData']=c
    writePartInfo(inf)
    updateModel()
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function smallLabelSizeChange_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    local i=1
    local t={0.01,0.01}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=10 end
        t[i]=t[i]*0.001
        if t[i]>0.3 then t[i]=0.3 end
        if t[i]<0.01 then t[i]=0.01 end
        i=i+1
    end
    local ov=c['smallLabelSize']
    if ov[1]~=t[1] or ov[2]~=t[2] then
        c['smallLabelSize']={t[1],t[2]}
        inf['labelData']=c
        writePartInfo(inf)
        updateModel()
    end
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function largeLabelSizeChange_callback(ui,id,newVal)
    local inf=readPartInfo()
    local c=inf['labelData']
    local i=1
    local t={0.01,0.01}
    for token in (newVal..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=10 end
        t[i]=t[i]*0.001
        if t[i]>0.3 then t[i]=0.3 end
        if t[i]<0.01 then t[i]=0.01 end
        i=i+1
    end
    local ov=c['largeLabelSize']
    if ov[1]~=t[1] or ov[2]~=t[2] then
        c['largeLabelSize']={t[1],t[2]}
        inf['labelData']=c
        writePartInfo(inf)
        updateModel()
    end
    setDlgItemContent()
    simBWF.markUndoPoint()
end

function placementDlg(title,fieldIndex,tempComment)
    local inf=readPartInfo()
    local prop=inf['labelData']['placementCode']
    local s="800 400"
    local p="200 200"
    if distributionDlgSize then
        s=distributionDlgSize[1]..' '..distributionDlgSize[2]
    end
    if distributionDlgPos then
        p=distributionDlgPos[1]..' '..distributionDlgPos[2]
    end
    local xml = [[ <editor title="]]..title..[[" size="]]..s..[[" position="]]..p..[[" tabWidth="4" textColor="50 50 50" backgroundColor="190 190 190" selectionColor="128 128 255" useVrepKeywords="true" isLua="true"> <keywords1 color="152 0 0" > </keywords1> <keywords2 color="220 80 20" > </keywords2> </editor> ]]            



    local initialText=prop[fieldIndex]
    if tempComment then
        initialText=initialText.."--[[tmpRem\n\n"..tempComment.."\n\n--]]"
    end
    local modifiedText
    modifiedText,distributionDlgSize,distributionDlgPos=sim.openTextEditor(initialText,xml)

    local toExecute="local boxSizeX,boxSizeY,boxSizeZ,labelSizeX,labelSizeY,labelRadius=1,1,1,1,1,1\n return {"..modifiedText.."}"
    local res,theTable=sim.executeLuaCode(toExecute)
    local success=false
    if theTable then
        if type(theTable)=='table' and #theTable>=2 then
            if type(theTable[1])=='table' and type(theTable[2])=='table' and #theTable[1]>=3 and #theTable[2]>=3 then
                success=true
                for i=1,3,1 do
                    if type(theTable[1][i])~='number' or type(theTable[2][i])~='number' then
                        success=false
                        break
                    end
                end
                if success then
                    modifiedText=simBWF.removeTmpRem(modifiedText)
                    prop[fieldIndex]=modifiedText
                    inf['labelData']['placementCode']=prop
                    writePartInfo(inf)
                    updateModel()
                    simBWF.markUndoPoint()
                end
            end
        end
    end
    if not success then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,'Input Error',"The placement code is ill-formated.")
    end
end

function getPlacementHelpText()
    local txt=[[
a) Usage:
   <labelPosition>,<labelOrientation> where <labelPosition> is {posX,posY,posZ} (in meters) and <labelOrientation> is {rotAroundX,rotAroundY,rotAroundZ} (in radians) 

b) You may use the variables 'boxSizeX', 'boxSizeY', 'boxSizeZ', 'labelSizeX', 'labelSizeY' and 'labelRadius'

c) Example:
   {0,0,boxSizeZ*0.5+0.001},{0,0,0}

d) Placement is relative to the box's reference frame]]
    return txt
end

function placementLabel1_callback(ui,id,newVal)
    local tmpTxt=getPlacementHelpText()
    placementDlg('Label 1 placement',1,tmpTxt)
end

function placementLabel2_callback(ui,id,newVal)
    local tmpTxt=getPlacementHelpText()
    placementDlg('Label 2 placement',2,tmpTxt)
end

function placementLabel3_callback(ui,id,newVal)
    local tmpTxt=getPlacementHelpText()
    placementDlg('Label 3 placement',3,tmpTxt)
end

function onCloseClicked()
    if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Finalizing the box',"By closing this customization dialog you won't be able to customize the box anymore. Do you want to proceed?") then
        finalizeModel=true
        sim.removeScript(sim.handle_self)
    end
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange_callback" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange_callback" id="2"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange_callback" id="3"/>

                <label text="Mass (Kg)"/>
                <edit on-editing-finished="massChange_callback" id="20"/>


                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
            </tab>
            <tab title="Labels">
                <group layout="form" flat="true">
                    <label text="Small label size (X, Y, in mm)"/>
                    <edit on-editing-finished="smallLabelSizeChange_callback" id="50"/>
                    <label text="Large label size (X, Y, in mm)"/>
                    <edit on-editing-finished="largeLabelSizeChange_callback" id="51"/>
                </group>

                <group layout="grid" flat="true">
                    <checkbox text="Label 1" on-change="label1_callback" id="40" />
                    <checkbox text="Large label" on-change="largeLabel1_callback" id="41" />
                    <button text="Label placement code"  on-click="placementLabel1_callback" id="42" />
                    <br />
                    <checkbox text="Label 2" on-change="label2_callback" id="43" />
                    <checkbox text="Large label" on-change="largeLabel2_callback" id="44" />
                    <button text="Label placement code"  on-click="placementLabel2_callback" id="45" />
                    <br />
                    <checkbox text="Label 3" on-change="label3_callback" id="46" />
                    <checkbox text="Large label" on-change="largeLabel3_callback" id="47" />
                    <button text="Label placement code"  on-click="placementLabel3_callback" id="48" />
                </group>
            </tab>
            <tab title="Colors/Texture" layout="form">
                    <label text="Textured"/>
                    <checkbox text="" on-change="texture_callback" id="888" />
                    
                    <label text="Red"/>
                    <hslider minimum="0" maximum="100" on-change="redChange" id="30"/>
                    <label text="Green"/>
                    <hslider minimum="0" maximum="100" on-change="greenChange" id="31"/>
                    <label text="Blue"/>
                    <hslider minimum="0" maximum="100" on-change="blueChange" id="32"/>
            </tab>
        </tabs>
        ]]

        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,true,'onCloseClicked'--[[,modal,resizable,activate,additionalUiAttribute--]])

        setDlgItemContent()
        simUI.setCurrentTab(ui,77,dlgMainTabIndex,true)
        
    end
end

function showDlg()
    if not ui then
        createDlg()
    end
end

function removeDlg()
    if ui then
        local x,y=simUI.getPosition(ui)
        previousDlgPos={x,y}
        dlgMainTabIndex=simUI.getCurrentTab(ui,77)
        simUI.destroy(ui)
        ui=nil
    end
end

function sysCall_init()
    dlgMainTabIndex=0
    model=sim.getObjectAssociatedWithScript(sim.handle_self)
    _MODELVERSION_=0
    _CODEVERSION_=0
    local _info=readInfo()
    simBWF.checkIfCodeAndModelMatch(model,_CODEVERSION_,_info['version'])
    writeInfo(_info)
    local data=readPartInfo()
    if data['name']=='<partName>' then
        data['name']='SHIPPINGBOX'
    end
    writePartInfo(data)
    texture=sim.getObjectHandle('genericShippingBox_texture')
    smallLabel=sim.getObjectHandle('genericShippingBox_smallLabel')
    largeLabel=sim.getObjectHandle('genericShippingBox_largeLabel')
    
    updatePluginRepresentation()
    previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos=simBWF.readSessionPersistentObjectData(model,"dlgPosAndSize")
end

showOrHideUiIfNeeded=function()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model then
        showDlg()
    else
        removeDlg()
    end
end

function sysCall_nonSimulation()
    showOrHideUiIfNeeded()
end


function sysCall_afterSimulation()
    showOrHideUiIfNeeded()
end

function sysCall_beforeSimulation()
    removeDlg()
end

function sysCall_beforeInstanceSwitch()
    removeDlg()
    removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    updatePluginRepresentation()
end

function sysCall_cleanup()
    removeDlg()
    removeFromPluginRepresentation()
    local repo,modelHolder=simBWF.getPartRepositoryHandles()
    if (repo and (sim.getObjectParent(model)==modelHolder)) or finalizeModel then
        -- This means the box is part of the part repository or that we want to finalize the model (i.e. won't be customizable anymore)
        local c=readInfo()
        sim.writeCustomDataBlock(model,'XYZ_SHIPPINGBOX_INFO','')
        sim.removeObject(texture)
        sim.removeObject(smallLabel)
        sim.removeObject(largeLabel)
    end
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end
