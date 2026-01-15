function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='A90'
    end
    if not info['velocity'] then
        info['velocity']=0.1
    end
    if not info['acceleration'] then
        info['acceleration']=0.01
    end
    if not info['outerRadius'] then
        info['outerRadius']=1
    end
    if not info['width'] then
        info['width']=0.3
    end
    if not info['padHeight'] then
        info['padHeight']=0.04
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['padSpacing'] then
        info['padSpacing']=0.2
    end
    if not info['padThickness'] then
        info['padThickness']=0.01
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
    sim.setShapeColor(pad,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(pad,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(endPad1,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(endPad1,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(endPad2,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(endPad2,nil,sim.colorcomponent_specular,{spec,spec,spec})

    i=0
    while true do
        local h=sim.getObjectChild(path,i)
        if h>=0 then
            local ch=sim.getObjectChild(h,0)
            if ch>=0 then
                sim.setShapeColor(ch,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
                sim.setShapeColor(ch,nil,sim.colorcomponent_specular,{spec,spec,spec})
            end
            i=i+1
        else
            break
        end
    end
end

function getColor()
    local r,rgb=sim.getShapeColor(pad,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(pad,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
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

function getActualPadSpacing()
    local conf=readInfo()
    local l=sim.getPathLength(path)
    local cnt=math.floor(l/conf['padSpacing'])+1
    local dx=l/cnt
    return dx
end

function updateConveyor()
    local conf=readInfo()
    local length=conf['outerRadius']
    local outerRadius=length
    local width=conf['width']
    local innerRadius=outerRadius-width
    local wallHeight=conf['padHeight']
    local padHeight=0.01
    local padSpacing=conf['padSpacing']
    local bitCoded=conf['bitCoded']
    local padThickness=0.04--conf[9]
    local baseThickness=conf['height']
    local wt=conf['wallThickness']

    local pr=outerRadius-width*0.5
    local padSep
    local i=0
    while true do
        padSep=padThickness*pr/outerRadius
        if math.pi*0.5*pr/padThickness<10 or i==1 then
            break
        end
        padThickness=padThickness*2
        i=i+1
    end

    local sc=0.004

    setShapeSize(front,outerRadius+sc,outerRadius+sc,baseThickness)
    sim.setObjectPosition(front,model,{padThickness*0.5+(outerRadius+sc)*0.5,padThickness*0.5+(outerRadius+sc)*0.5,-baseThickness*0.5})
    setShapeSize(back,outerRadius-width-sc,outerRadius-width-sc,baseThickness)
    sim.setObjectPosition(back,model,{padThickness*0.5+(outerRadius-width-sc)*0.5,padThickness*0.5+(outerRadius-width-sc)*0.5,-baseThickness*0.5})
    
    setShapeSize(endPart1,width+sc*2,padThickness,baseThickness)
    sim.setObjectPosition(endPart1,model,{padThickness*0.5+pr,padThickness*0.5,-baseThickness*0.5})
    setShapeSize(endPart2,padThickness,width+sc*2,baseThickness)
    sim.setObjectPosition(endPart2,model,{padThickness*0.5,padThickness*0.5+pr,-baseThickness*0.5})
    setShapeSize(pad,width,padThickness,padHeight)
    sim.setObjectPosition(path,model,{0,0,-baseThickness*0.5})
    sim.setObjectPosition(pad,sim.handle_parent,{0,-padHeight*0.5,0})

    while true do
        local h=sim.getObjectChild(path,0)
        if h>=0 then
            sim.removeObject(h)
        else
            break
        end
    end


    local hc=0
    for lc=1,2,1 do
        sim.cutPathCtrlPoints(path,-1,0)
        local pts={}
        local div=30
        for i=0,div-1,1 do
            pts[(i+0)*11+1]=padThickness*0.5+pr*math.cos(i*math.pi*0.5/(div-1))+((i==(div-1)) and 0.001 or 0)
            pts[(i+0)*11+2]=padThickness*0.5+pr*math.sin(i*math.pi*0.5/(div-1))+((i==0) and 0.001 or 0)
            pts[(i+0)*11+3]=baseThickness*0.5
            pts[(i+0)*11+4]=math.pi*0.5
            pts[(i+0)*11+5]=i*math.pi*0.5/(div-1)
            pts[(i+0)*11+6]=0
            pts[(i+0)*11+7]=1
            pts[(i+0)*11+8]=0
            pts[(i+0)*11+9]=((i==0)or(i==(div-1))) and 1 or 15
            pts[(i+0)*11+10]=0.95
            pts[(i+0)*11+11]=0.95
        end
        for i=0,div-1,1 do
            pts[(i+div)*11+1]=padThickness*0.5+pr*math.sin(i*math.pi*0.5/(div-1))+((i==0) and 0.001 or 0)
            pts[(i+div)*11+2]=padThickness*0.5+pr*math.cos(i*math.pi*0.5/(div-1))+((i==(div-1)) and 0.001 or 0)
            pts[(i+div)*11+3]=(baseThickness-padThickness)*0.5+hc
            pts[(i+div)*11+4]=math.pi*0.5
            pts[(i+div)*11+5]=(div-1-i)*math.pi*0.5/(div-1)
            pts[(i+div)*11+6]=0
            pts[(i+div)*11+7]=1
            pts[(i+div)*11+8]=0
            pts[(i+div)*11+9]=((i==0)or(i==(div-1))) and 1 or 15
            pts[(i+div)*11+10]=0.95
            pts[(i+div)*11+11]=0.95
        end
        sim.insertPathCtrlPoints(path,1,0,div*2,pts)
        if lc==1 then
            local l=sim.getPathLength(path)
            local cnt=math.floor(l/padSep)
            hc=(l-cnt*padSep)*0.5
        end
    end

    local l=sim.getPathLength(path)
    local cnt=math.floor(l/padSep)

    local dx=l/cnt
    for i=0,cnt-1,1 do
        local pb=sim.copyPasteObjects({padBase},0)[1]
        local p=sim.copyPasteObjects({pad},0)[1]
        sim.setObjectParent(p,pb,true)
        sim.setObjectParent(pb,path,true)
        sim.setObjectInt32Parameter(p,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(p,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(p,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectFloatParameter(pb,sim.dummyfloatparam_follow_path_offset,i*dx)

        if sim.boolAnd32(bitCoded,32)>0 then
            sim.setShapeTexture(p,-1,sim.texturemap_plane,12,{0.04,0.04})
        end

    end
    setShapeSize(endPad1,width,padThickness,padHeight)
    setShapeSize(endPad2,width,padThickness,padHeight)
    sim.setObjectPosition(endPad1,model,{padThickness*0.5+pr,padThickness*0.5,-padHeight*0.5})
    sim.setObjectPosition(endPad2,model,{padThickness*0.5,padThickness*0.5+pr,-padHeight*0.5})
    if sim.boolAnd32(bitCoded,32)==0 then
        local textureID=sim.getShapeTextureId(pad)
        sim.setShapeTexture(endPad1,textureID,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endPad2,textureID,sim.texturemap_plane,12,{0.04,0.04})
    else
        sim.setShapeTexture(endPad1,-1,sim.texturemap_plane,12,{0.04,0.04})
        sim.setShapeTexture(endPad2,-1,sim.texturemap_plane,12,{0.04,0.04})
    end


    local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0) -- do not report errors
    local obj=sim.getObjectHandle('genericCurvedConveyorTypeB_sides')
    sim.setInt32Parameter(sim.intparam_error_report_mode,err) -- report errors again
    if obj>=0 then
        sim.removeObject(obj)
    end

    local sideParts={}
    if sim.boolAnd32(bitCoded,4)==0 then
        local div=2+math.floor(math.pi*outerRadius*0.5/padThickness)
        for i=0,div-1,1 do
            local p1=sim.copyPasteObjects({sidePad},0)[1]
            setShapeSize(p1,wt,padThickness,wallHeight+baseThickness)
            sim.setObjectPosition(p1,model,{padThickness*0.5+(outerRadius+wt*0.5)*math.cos(i*math.pi*0.5/(div-1)),padThickness*0.5+(outerRadius+wt*0.5)*math.sin(i*math.pi*0.5/(div-1)),(wallHeight-baseThickness)*0.5})
            sim.setObjectOrientation(p1,model,{0,0,i*math.pi*0.5/(div-1)})
            sideParts[#sideParts+1]=p1
        end
    end

    if sim.boolAnd32(bitCoded,8)==0 then
        local div=2+math.floor(math.pi*innerRadius*0.5/padThickness)
        for i=0,div-1,1 do
            local p1=sim.copyPasteObjects({sidePad},0)[1]
            setShapeSize(p1,wt,padThickness,wallHeight+baseThickness)
            sim.setObjectPosition(p1,model,{padThickness*0.5+(innerRadius-wt*0.5)*math.cos(i*math.pi*0.5/(div-1)),padThickness*0.5+(innerRadius-wt*0.5)*math.sin(i*math.pi*0.5/(div-1)),(wallHeight-baseThickness)*0.5})
            sim.setObjectOrientation(p1,model,{0,0,i*math.pi*0.5/(div-1)})
            sideParts[#sideParts+1]=p1
        end
    end
    if #sideParts>0 then
        local h=sim.groupShapes(sideParts)
        sim.setObjectInt32Parameter(h,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(h,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(h,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        local suff=sim.getNameSuffix(sim.getObjectName(model))
        local name='genericCurvedConveyorTypeB_sides'
        if suff>=0 then
            name=name..'#'..suff
        end
        sim.setObjectName(h,name)
        sim.setObjectParent(h,model,true)
    end
--]]
    
end

function outerRadiusChange(ui,id,newVal)
    local conf=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<conf['width']+0.05 then l=conf['width']+0.05 end
        if l>2 then l=2 end
        if l~=conf['outerRadius'] then
            simBWF.markUndoPoint()
            conf['outerRadius']=l
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,2,simBWF.format("%.0f",conf['outerRadius']/0.001),true)
end

function widthChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.05 then w=0.05 end
        if w>conf['outerRadius']-0.05 then w=conf['outerRadius']-0.05 end
        if w~=conf['width'] then
            simBWF.markUndoPoint()
            conf['width']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,4,simBWF.format("%.0f",conf['width']/0.001),true)
end

function padHeightChange(ui,id,newVal)
    local conf=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.001 then w=0.001 end
        if w>0.2 then w=0.2 end
        if w~=conf['padHeight'] then
            simBWF.markUndoPoint()
            conf['padHeight']=w
            writeInfo(conf)
            updateConveyor()
        end
    end
    simUI.setEditValue(ui,20,simBWF.format("%.0f",conf['padHeight']/0.001),true)
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
    simUI.setEditValue(ui,28,simBWF.format("%.0f",conf['height']/0.001),true)
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
    simUI.setEditValue(ui,29,simBWF.format("%.0f",conf['wallThickness']/0.001),true)
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

function texturedClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],32)
    if newVal~=0 then
        conf['bitCoded']=conf['bitCoded']-32
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
    updateConveyor()
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
function enabledClicked(ui,id,newVal)
    local conf=readInfo()
    conf['bitCoded']=sim.boolOr32(conf['bitCoded'],64)
    if newVal==0 then
        conf['bitCoded']=conf['bitCoded']-64
    end
    simBWF.markUndoPoint()
    writeInfo(conf)
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
    simBWF.markUndoPoint()
    updateMasterCombobox()
    updateEnabledDisabledItems()
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
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,2,simStopped,true)
        simUI.setEnabled(ui,4,simStopped,true)

        simUI.setEnabled(ui,20,simStopped,true)
        simUI.setEnabled(ui,24,simStopped,true)
        simUI.setEnabled(ui,25,simStopped,true)
        simUI.setEnabled(ui,28,simStopped,true)
        simUI.setEnabled(ui,29,simStopped,true)
        simUI.setEnabled(ui,30,simStopped,true)

        simUI.setEnabled(ui,5,simStopped,true)
        simUI.setEnabled(ui,6,simStopped,true)
        simUI.setEnabled(ui,7,simStopped,true)
        simUI.setEnabled(ui,8,simStopped,true)

        simUI.setEnabled(ui,1000,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- enable
        simUI.setEnabled(ui,10,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- vel
        simUI.setEnabled(ui,12,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- accel
        
        simUI.setEnabled(ui,100,simStopped,true) -- stop trigger
        simUI.setEnabled(ui,101,simStopped,true) -- restart trigger
        simUI.setEnabled(ui,102,simStopped,true) -- master
    end
end

function onCloseClicked()
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
                <label text="Outer radius (mm)"/>
                <edit on-editing-finished="outerRadiusChange" id="2"/>

                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange" id="4"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange" id="28"/>
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
                <checkbox text="" on-change="texturedClicked" id="30"/>

                <label text="Front side open"/>
                <checkbox text="" on-change="frontSideOpenClicked" id="24"/>

                <label text="Back side open"/>
                <checkbox text="" on-change="backSideOpenClicked" id="25"/>

                <label text="Side height (mm)"/>
                <edit on-editing-finished="padHeightChange" id="20"/>

                <label text="Side thickness (mm)"/>
                <edit on-editing-finished="wallThicknessChange" id="29"/>
    </tab>
    </tabs>
        ]]


        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])

        local red,green,blue,spec=getColor()
        local config=readInfo()

        simUI.setEditValue(ui,10,simBWF.format("%.0f",config['velocity']/0.001),true)
        simUI.setEditValue(ui,12,simBWF.format("%.0f",config['acceleration']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['outerRadius']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,20,simBWF.format("%.0f",config['padHeight']/0.001),true)
        simUI.setEditValue(ui,28,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,29,simBWF.format("%.0f",config['wallThickness']/0.001),true)
        simUI.setCheckboxValue(ui,24,(sim.boolAnd32(config['bitCoded'],4)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,25,(sim.boolAnd32(config['bitCoded'],8)~=0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,30,(sim.boolAnd32(config['bitCoded'],32)==0) and 2 or 0,true)
        simUI.setCheckboxValue(ui,1000,(sim.boolAnd32(config['bitCoded'],64)~=0) and 2 or 0,true)

        simUI.setSliderValue(ui,5,red*100,true)
        simUI.setSliderValue(ui,6,green*100,true)
        simUI.setSliderValue(ui,7,blue*100,true)
        simUI.setSliderValue(ui,8,spec*100,true)

        updateStartStopTriggerComboboxes()
        updateMasterCombobox()
        updateEnabledDisabledItems()
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

    padBase=sim.getObjectHandle('genericCurvedConveyorTypeA90_padBase')
    pad=sim.getObjectHandle('genericCurvedConveyorTypeA90_pad')
    path=sim.getObjectHandle('genericCurvedConveyorTypeA90_path')
    endPad1=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPad1')
    endPad2=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPad2')
    front=sim.getObjectHandle('genericCurvedConveyorTypeA90_front')
    back=sim.getObjectHandle('genericCurvedConveyorTypeA90_back')
    endPart1=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPart1')
    endPart2=sim.getObjectHandle('genericCurvedConveyorTypeA90_endPart2')
    sidePad=sim.getObjectHandle('genericCurvedConveyorTypeA90_sidePad')
	
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