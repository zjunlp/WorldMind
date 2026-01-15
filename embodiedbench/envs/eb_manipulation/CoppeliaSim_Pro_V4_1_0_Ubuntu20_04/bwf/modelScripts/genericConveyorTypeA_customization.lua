function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function ext_getItemData_pricing()
    local c=readInfo()
    local obj={}
    obj.name=sim.getObjectName(model)
    obj.type='conveyor'
    obj.conveyorType='default'
    obj.brVersion=0
    obj.length=c.length*1000 -- in mm here
    obj.width=c.width*1000 -- in mm here
    return obj
end

function setShapeSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='A'
    end
    if not info['velocity'] then
        info['velocity']=0.1
    end
    if not info['acceleration'] then
        info['acceleration']=0.01
    end
    if not info['length'] then
        info['length']=1
    end
    if not info['width'] then
        info['width']=0.3
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['borderHeight'] then
        info['borderHeight']=0.2
    end
    if not info['bitCoded'] then
        info['bitCoded']=1+2+4+8
    end
    if not info['wallThickness'] then
        info['wallThickness']=0.005
    end
    if not info['stopRequests'] then
        info['stopRequests']={}
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,simBWF.modelTags.CONVEYOR)
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
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,simBWF.modelTags.CONVEYOR,'')
    end
end
function setColor(red,green,blue,spec)
    sim.setShapeColor(middleParts[2],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(middleParts[2],nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(endParts[1],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(endParts[1],nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(endParts[2],nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(endParts[2],nil,sim.colorcomponent_specular,{spec,spec,spec})
end

function getColor()
    local r,rgb=sim.getShapeColor(middleParts[2],nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(middleParts[2],nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function updateConveyor()
    local conf=readInfo()
    local length=conf['length']
    local width=conf['width']
    local height=conf['height']
    local borderHeight=conf['borderHeight']
    local bitCoded=conf['bitCoded']
    local wt=conf['wallThickness']
    local re=sim.boolAnd32(bitCoded,16)==0
---[[
    sim.setObjectPosition(rotJoints[1],model,{0,-length*0.5,-height*0.5})
    sim.setObjectPosition(rotJoints[2],model,{0,length*0.5,-height*0.5})

    setShapeSize(middleParts[1],width,length,height)
    setShapeSize(middleParts[2],width,length,0.001)
    setShapeSize(middleParts[3],width,length,height)
    sim.setObjectPosition(middleParts[1],model,{0,0,-height*0.5})
    sim.setObjectPosition(middleParts[2],model,{0,0,-0.0005})
    sim.setObjectPosition(middleParts[3],model,{0,0,-height*0.5})

    setShapeSize(endParts[1],width,0.083148*height/0.2,0.044443*height/0.2)
    sim.setObjectPosition(endParts[1],model,{0,-length*0.5-0.5*0.083148*height/0.2,-0.044443*height*0.5/0.2})

    setShapeSize(endParts[2],width,0.083148*height/0.2,0.044443*height/0.2)
    sim.setObjectPosition(endParts[2],model,{0,length*0.5+0.5*0.083148*height/0.2,-0.044443*height*0.5/0.2})

    setShapeSize(endParts[3],width,height*0.5,height)
    sim.setObjectPosition(endParts[3],model,{0,-length*0.5-0.25*height,-height*0.5})

    setShapeSize(endParts[4],width,height*0.5,height)
    sim.setObjectPosition(endParts[4],model,{0,length*0.5+0.25*height,-height*0.5})

    for i=5,6,1 do
        setShapeSize(endParts[i],height,height,width)
    end

    setShapeSize(sides[1],wt,length,height+2*borderHeight)
    setShapeSize(sides[2],wt,length,height+2*borderHeight)
    setShapeSize(sides[4],width+2*wt,height*0.5+1*borderHeight,height+2*borderHeight)
    setShapeSize(sides[3],width+2*wt,height*0.5+1*borderHeight,height+2*borderHeight)
    sim.setObjectPosition(sides[4],model,{0,-(length+height*0.5+borderHeight)*0.5,-height*0.5})
    sim.setObjectPosition(sides[3],model,{0,(length+height*0.5+borderHeight)*0.5,-height*0.5})
    sim.setObjectPosition(sides[1],model,{-(width+wt)*0.5,0,-height*0.5})
    sim.setObjectPosition(sides[2],model,{(width+wt)*0.5,0,-height*0.5})

    if re then
        sim.setObjectInt32Parameter(endParts[1],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(endParts[2],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(endParts[3],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(endParts[4],sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Parameter(endParts[5],sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(endParts[6],sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(endParts[5],sim.shapeintparam_respondable,1)
        sim.setObjectInt32Parameter(endParts[6],sim.shapeintparam_respondable,1)
    else
        sim.setObjectInt32Parameter(endParts[1],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[2],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[3],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[4],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[5],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[6],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(endParts[5],sim.shapeintparam_respondable,0)
        sim.setObjectInt32Parameter(endParts[6],sim.shapeintparam_respondable,0)
    end

    if sim.boolAnd32(bitCoded,1)~=0 then
        sim.setObjectInt32Parameter(sides[1],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(sides[1],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(sides[1],0)
        sim.setObjectProperty(sides[1],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(sides[1],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(sides[1],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(sides[1],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(sides[1],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,2)~=0 then
        sim.setObjectInt32Parameter(sides[2],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(sides[2],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(sides[2],0)
        sim.setObjectProperty(sides[2],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(sides[2],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(sides[2],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(sides[2],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(sides[2],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,4)~=0 or (not re) then
        sim.setObjectInt32Parameter(sides[3],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(sides[3],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(sides[3],0)
        sim.setObjectProperty(sides[3],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(sides[3],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(sides[3],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(sides[3],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(sides[3],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end
    if sim.boolAnd32(bitCoded,8)~=0 or (not re) then
        sim.setObjectInt32Parameter(sides[4],sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Parameter(sides[4],sim.shapeintparam_respondable,0)
        sim.setObjectSpecialProperty(sides[4],0)
        sim.setObjectProperty(sides[4],sim.objectproperty_dontshowasinsidemodel)
    else
        sim.setObjectInt32Parameter(sides[4],sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(sides[4],sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(sides[4],sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectProperty(sides[4],sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    end

    if sim.boolAnd32(bitCoded,32)==0 then
        local textureID=sim.getShapeTextureId(textureHolder)
        sim.setShapeTexture(middleParts[2],textureID,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endParts[1],textureID,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endParts[2],textureID,sim.texturemap_plane,12,{0.04,0.04})
    else
        sim.setShapeTexture(middleParts[2],-1,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endParts[1],-1,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endParts[2],-1,sim.texturemap_plane,12,{0.04,0.04})
    end
--]]    
end

function getAvailableSensors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        local data=sim.readCustomDataBlock(l[i],'XYZ_BINARYSENSOR_INFO')
        if data then
            retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
        end
    end
    return retL
end

function getAvailableMasterConveyors()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    local retL={}
    for i=1,#l,1 do
        if l[i]~=model then
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.CONVEYOR)
            if data then
                retL[#retL+1]={sim.getObjectName(l[i]),l[i]}
            end
        end
    end
    return retL
end

function lengthChange(ui,id,newVal)
    local conf=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.01 then l=0.01 end
        if l>5 then l=5 end
        if l~=conf['length'] then
            simBWF.markUndoPoint()
            conf['length']=l
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,2,simBWF.format("%.0f",conf['length']/0.001),true)
end

function widthChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.01 then w=0.01 end
        if w>2 then w=2 end
        if w~=conf['width'] then
            simBWF.markUndoPoint()
            conf['width']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,4,simBWF.format("%.0f",conf['width']/0.001),true)
end

function heightChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.01 then w=0.01 end
        if w>1 then w=1 end
        if w~=conf['height'] then
            simBWF.markUndoPoint()
            conf['height']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,20,simBWF.format("%.0f",conf['height']/0.001),true)
end

function borderHeightChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.005 then w=0.005 end
        if w>0.2 then w=0.2 end
        if w~=conf['borderHeight'] then
            simBWF.markUndoPoint()
            conf['borderHeight']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,21,simBWF.format("%.0f",conf['borderHeight']/0.001),true)
end

function wallThicknessChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.001 then w=0.001 end
        if w>0.02 then w=0.02 end
        if w~=conf['wallThickness'] then
            simBWF.markUndoPoint()
            conf['wallThickness']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,26,simBWF.format("%.0f",conf['wallThickness']/0.001),true)
end

function leftSideOpenClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],1)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
    updateConveyor()
end

function rightSideOpenClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],2)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-2
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
    updateConveyor()
end

function frontSideOpenClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],4)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-4
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
    updateConveyor()
end

function backSideOpenClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],8)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-8
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
    updateConveyor()
end

function triggerStopChange_callback(ui,id,newIndex)
    local sens=comboStopTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,1,sens)
    if simBWF.getReferencedObjectHandle(model,2)==sens then
        simBWF.setReferencedObjectHandle(model,2,-1)
    end
    simBWF.markUndoPoint()
    updateStartStopTriggerComboboxes()
end

function triggerStartChange_callback(ui,id,newIndex)
    local sens=comboStartTrigger[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,2,sens)
    if simBWF.getReferencedObjectHandle(model,1)==sens then
        simBWF.setReferencedObjectHandle(model,1,-1)
    end
    simBWF.markUndoPoint()
    updateStartStopTriggerComboboxes()
end

function masterChange_callback(ui,id,newIndex)
    local sens=comboMaster[newIndex+1][2]
    simBWF.setReferencedObjectHandle(model,3,sens) -- master
    updateMasterCombobox()
    updateEnabledDisabledItems()
    simBWF.markUndoPoint()
end

function updateStartStopTriggerComboboxes()
    local c=readInfo()
    local loc=getAvailableSensors()
    comboStopTrigger=simBWF.populateCombobox(ui,100,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,1)),true,{{simBWF.NONE_TEXT,-1}})
    comboStartTrigger=simBWF.populateCombobox(ui,101,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,2)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateMasterCombobox()
    local c=readInfo()
    local loc=getAvailableMasterConveyors()
    comboMaster=simBWF.populateCombobox(ui,102,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,3)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateEnabledDisabledItems()
    if ui then
        local c=readInfo()
        local re=sim.boolAnd32(c['bitCoded'],16)==0
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,2,simStopped,true)
        simUI.setEnabled(ui,4,simStopped,true)
        simUI.setEnabled(ui,5,simStopped,true)
        simUI.setEnabled(ui,6,simStopped,true)
        simUI.setEnabled(ui,7,simStopped,true)
        simUI.setEnabled(ui,8,simStopped,true)
        simUI.setEnabled(ui,20,simStopped,true)
        simUI.setEnabled(ui,21,simStopped,true)
        simUI.setEnabled(ui,22,simStopped,true)
        simUI.setEnabled(ui,23,simStopped,true)
        simUI.setEnabled(ui,24,simStopped and re,true)
        simUI.setEnabled(ui,25,simStopped and re,true)
        simUI.setEnabled(ui,26,simStopped,true)
        simUI.setEnabled(ui,27,simStopped,true)
        simUI.setEnabled(ui,28,simStopped,true)

        simUI.setEnabled(ui,1000,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- enable
        simUI.setEnabled(ui,10,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- vel
        simUI.setEnabled(ui,12,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- accel
        
        simUI.setEnabled(ui,100,simStopped,true) -- stop trigger
        simUI.setEnabled(ui,101,simStopped,true) -- restart trigger
        simUI.setEnabled(ui,102,simStopped,true) -- master
    end
end

function roundedEndsClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],16)
    if newVal~=0 then
        conf['bitCoded']=conf['bitCoded']-16
    end
    writeInfo(conf)
    simBWF.markUndoPoint()
    updateConveyor()
    updateEnabledDisabledItems()
end

function texturedClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],32)
    if newVal~=0 then
        conf['bitCoded']=conf['bitCoded']-32
    end
    writeInfo(conf)
    simBWF.markUndoPoint()
    updateConveyor()
end

function enabledClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],64)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-64
    end
    writeInfo(conf)
    simBWF.markUndoPoint()
end

function redChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(newVal/100,g,b,s)
end

function greenChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,newVal/100,b,s)
end

function blueChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,newVal/100,s)
end

function specularChange(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor()
    setColor(r,g,b,newVal/100)
end

function speedChange(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<-0.5 then v=-0.5 end
        if v>0.5 then v=0.5 end
        if v~=c['velocity'] then
            simBWF.markUndoPoint()
            c['velocity']=v
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,10,simBWF.format("%.0f",c['velocity']/0.001),true)
end

function accelerationChange(ui,id,newVal)
    local c=readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.001 then v=0.001 end
        if v>1 then v=1 end
        if v~=c['acceleration'] then
            simBWF.markUndoPoint()
            c['acceleration']=v
            writeInfo(c)
        end
    end
    simUI.setEditValue(ui,12,simBWF.format("%.0f",c['acceleration']/0.001),true)
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml = [[
    <tabs id="77">
    <tab title="General" layout="form">
                <label text="Enabled"/>
                <checkbox text="" on-change="enabledClicked" id="1000"/>

                <label text="Speed (mm/s)"/>
                <edit on-editing-finished="speedChange" id="10"/>

                <label text="Acceleration (mm/s^2)"/>
                <edit on-editing-finished="accelerationChange" id="12"/>

                <label text="Master conveyor"/>
                <combobox id="102" on-change="masterChange_callback">
                </combobox>

                <label text="Stop on trigger"/>
                <combobox id="100" on-change="triggerStopChange_callback">
                </combobox>

                <label text="Restart on trigger"/>
                <combobox id="101" on-change="triggerStartChange_callback">
                </combobox>

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>

    </tab>
    <tab title="Dimensions" layout="form">
                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange" id="2"/>

                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange" id="4"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange" id="20"/>

                <label text="Border height (mm)"/>
                <edit on-editing-finished="borderHeightChange" id="21"/>

                <label text="Wall thickness (mm)"/>
                <edit on-editing-finished="wallThicknessChange" id="26"/>
    </tab>
    <tab title="Color" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange" id="5"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange" id="6"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange" id="7"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="specularChange" id="8"/>
    </tab>
    <tab title="More" layout="form">
                <label text="Textured"/>
                <checkbox text="" on-change="texturedClicked" id="28"/>

                <label text="Left side open"/>
                <checkbox text="" on-change="leftSideOpenClicked" id="22"/>

                <label text="Right side open"/>
                <checkbox text="" on-change="rightSideOpenClicked" id="23"/>

                <label text="Rounded ends"/>
                <checkbox text="" on-change="roundedEndsClicked" id="27"/>

                <label text="Front side open"/>
                <checkbox text="" on-change="frontSideOpenClicked" id="24"/>

                <label text="Back side open"/>
                <checkbox text="" on-change="backSideOpenClicked" id="25"/>
    </tab>
    </tabs>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

        local red,green,blue,spec=getColor()
        local config=readInfo()

        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['borderHeight']/0.001),true)
        simUI.setEditValue(ui,26,simBWF.format("%.0f",config['wallThickness']/0.001),true)

        simUI.setCheckboxValue(ui,22,(sim.boolAnd32(config['bitCoded'],1)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,23,(sim.boolAnd32(config['bitCoded'],2)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,24,(sim.boolAnd32(config['bitCoded'],4)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,25,(sim.boolAnd32(config['bitCoded'],8)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,27,(sim.boolAnd32(config['bitCoded'],16)==0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,28,(sim.boolAnd32(config['bitCoded'],32)==0) and 2 or 0,true)

        simUI.setCheckboxValue(ui,1000,(sim.boolAnd32(config['bitCoded'],64)~=0) and 2 or 0,true)
        simUI.setEditValue(ui,10,simBWF.format("%.0f",config['velocity']/0.001),true)
        simUI.setEditValue(ui,12,simBWF.format("%.0f",config['acceleration']/0.001),true)
--        simUI.setEditValue(ui,14,simBWF.format("%.0f",config[3]/0.001),true)

        simUI.setSliderValue(ui,5,red*100,true)
        simUI.setSliderValue(ui,6,green*100,true)
        simUI.setSliderValue(ui,7,blue*100,true)
        simUI.setSliderValue(ui,8,spec*100,true)

        updateStartStopTriggerComboboxes()
        updateMasterCombobox()

        updateEnabledDisabledItems()
        simUI.setCurrentTab(ui,77,dlgMainTabIndex,true)

--]]       
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
    -- Following for backward compatibility:
    if _info['stopTrigger'] then
        simBWF.setReferencedObjectHandle(model,1,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['stopTrigger']))
        _info['stopTrigger']=nil
    end
    if _info['startTrigger'] then
        simBWF.setReferencedObjectHandle(model,2,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['startTrigger']))
        _info['startTrigger']=nil
    end
    if _info['masterConveyor'] then
        simBWF.setReferencedObjectHandle(model,3,sim.getObjectHandle_noErrorNoSuffixAdjustment(_info['masterConveyor']))
        _info['masterConveyor']=nil
    end
    ----------------------------------------
    writeInfo(_info)
    
    rotJoints={}
    rotJoints[1]=sim.getObjectHandle('genericConveyorTypeA_jointB')
    rotJoints[2]=sim.getObjectHandle('genericConveyorTypeA_jointC')

    middleParts={}
    middleParts[1]=sim.getObjectHandle('genericConveyorTypeA_sides')
    middleParts[2]=sim.getObjectHandle('genericConveyorTypeA_textureA')
    middleParts[3]=sim.getObjectHandle('genericConveyorTypeA_forwarderA')
    
    endParts={}
    endParts[1]=sim.getObjectHandle('genericConveyorTypeA_textureB')
    endParts[2]=sim.getObjectHandle('genericConveyorTypeA_textureC')
    endParts[3]=sim.getObjectHandle('genericConveyorTypeA_B')
    endParts[4]=sim.getObjectHandle('genericConveyorTypeA_C')
    endParts[5]=sim.getObjectHandle('genericConveyorTypeA_forwarderB')
    endParts[6]=sim.getObjectHandle('genericConveyorTypeA_forwarderC')

    sides={}
    sides[1]=sim.getObjectHandle('genericConveyorTypeA_leftSide')
    sides[2]=sim.getObjectHandle('genericConveyorTypeA_rightSide')
    sides[3]=sim.getObjectHandle('genericConveyorTypeA_frontSide')
    sides[4]=sim.getObjectHandle('genericConveyorTypeA_backSide')

    textureHolder=sim.getObjectHandle('genericConveyorTypeA_textureHolder')

	
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

function sysCall_sensing()
    if simJustStarted then
        updateEnabledDisabledItems()
    end
    simJustStarted=nil
    showOrHideUiIfNeeded()
end

function sysCall_suspend()
    showOrHideUiIfNeeded()
end

function sysCall_afterSimulation()
    updateEnabledDisabledItems()
    local conf=readInfo()
    conf['encoderDistance']=0
    conf['stopRequests']={}
    writeInfo(conf)
end

function sysCall_beforeSimulation()
    simJustStarted=true
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
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end