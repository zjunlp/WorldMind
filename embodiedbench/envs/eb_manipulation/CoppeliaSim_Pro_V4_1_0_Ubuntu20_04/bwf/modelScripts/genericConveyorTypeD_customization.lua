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

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='D'
    end
    if not info['length'] then
        info['length']=1
    end
    if not info['width'] then
        info['width']=0.4
    end
    if not info['height'] then
        info['height']=0.1
    end
    if not info['velocity'] then
        info['velocity']=0.1
    end
    if not info['acceleration'] then
        info['acceleration']=0.1
    end
    if not info['bitCoded'] then
        info['bitCoded']=0 -- 64: enabled
    end
    if not info['encoderDistance'] then
        info['encoderDistance']=0
    end
    if not info['padSpacing'] then
        info['padSpacing']=0.2
    end
    if not info['padLength'] then
        info['padLength']=0.2
    end
    if not info['padWidth'] then
        info['padWidth']=0.1
    end
    if not info['padThickness'] then
        info['padThickness']=0.005
    end
    if not info['padWallHeight'] then
        info['padWallHeight']=0.01
    end
    if not info['padWallThickness'] then
        info['padWallThickness']=0.005
    end
    if not info['padVerticalOffset'] then
        info['padVerticalOffset']=0
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

function setColor1(red,green,blue,spec)
    sim.setShapeColor(padBaseShape,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(padBaseShape,nil,sim.colorcomponent_specular,{spec,spec,spec})
    i=0
    while true do
        local h=sim.getObjectChild(path,i)
        if h>=0 then
            local ch=sim.getObjectChild(h,0)
            if ch>=0 then
                sim.setShapeColor(ch,'BASE',sim.colorcomponent_ambient_diffuse,{red,green,blue})
                sim.setShapeColor(ch,'BASE',sim.colorcomponent_specular,{spec,spec,spec})
            end
            i=i+1
        else
            break
        end
    end
end

function getColor1()
    local r,rgb=sim.getShapeColor(padBaseShape,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(padBaseShape,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function setColor2(red,green,blue,spec)
    sim.setShapeColor(padWallShape,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(padWallShape,nil,sim.colorcomponent_specular,{spec,spec,spec})
    i=0
    while true do
        local h=sim.getObjectChild(path,i)
        if h>=0 then
            local ch=sim.getObjectChild(h,0)
            if ch>=0 then
                sim.setShapeColor(ch,'WALL',sim.colorcomponent_ambient_diffuse,{red,green,blue})
                sim.setShapeColor(ch,'WALL',sim.colorcomponent_specular,{spec,spec,spec})
            end
            i=i+1
        else
            break
        end
    end
end

function getColor2()
    local r,rgb=sim.getShapeColor(padWallShape,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(padWallShape,nil,sim.colorcomponent_specular)
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
    comboStopTrigger=simBWF.populateCombobox(ui,200,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,1)),true,{{simBWF.NONE_TEXT,-1}})
    comboStartTrigger=simBWF.populateCombobox(ui,201,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,2)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateMasterCombobox()
    local c=readInfo()
    local loc=getAvailableMasterConveyors()
    comboMaster=simBWF.populateCombobox(ui,202,loc,{},simBWF.getObjectNameOrNone(simBWF.getReferencedObjectHandle(model,3)),true,{{simBWF.NONE_TEXT,-1}})
end

function updateEnabledDisabledItems()
    if ui then
        local c=readInfo()
        local enabled=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(ui,2,enabled,true)
        simUI.setEnabled(ui,4,enabled,true)
        simUI.setEnabled(ui,28,enabled,true)
        simUI.setEnabled(ui,21,enabled,true)
        simUI.setEnabled(ui,5,enabled,true)
        simUI.setEnabled(ui,6,enabled,true)
        simUI.setEnabled(ui,7,enabled,true)
        simUI.setEnabled(ui,8,enabled,true)

        simUI.setEnabled(ui,30,enabled,true)
        simUI.setEnabled(ui,31,enabled,true)
        simUI.setEnabled(ui,32,enabled,true)
        simUI.setEnabled(ui,33,enabled,true)

        simUI.setEnabled(ui,100,enabled,true)
        simUI.setEnabled(ui,101,enabled,true)
        simUI.setEnabled(ui,102,enabled,true)
        simUI.setEnabled(ui,103,enabled,true)
        simUI.setEnabled(ui,104,enabled,true)
        simUI.setEnabled(ui,105,enabled,true)

        simUI.setEnabled(ui,1000,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- enable
        simUI.setEnabled(ui,10,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- vel
        simUI.setEnabled(ui,12,simBWF.getReferencedObjectHandle(model,3)==-1,true) -- accel
        
        simUI.setEnabled(ui,200,enabled,true) -- stop trigger
        simUI.setEnabled(ui,201,enabled,true) -- restart trigger
        simUI.setEnabled(ui,202,enabled,true) -- master
    end
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,10,simBWF.format("%.0f",config['velocity']/0.001),true)
        simUI.setEditValue(ui,12,simBWF.format("%.0f",config['acceleration']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,21,simBWF.format("%.0f",config['padSpacing']/0.001),true)
        simUI.setEditValue(ui,28,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setCheckboxValue(ui,1000,(sim.boolAnd32(config['bitCoded'],64)~=0) and 2 or 0,true)
        simUI.setLabelText(ui,26,simBWF.format("%.0f",getActualPadSpacing()/0.001),true)


        simUI.setEditValue(ui,100,simBWF.format("%.0f",config['padLength']/0.001),true)
        simUI.setEditValue(ui,101,simBWF.format("%.0f",config['padWidth']/0.001),true)
        simUI.setEditValue(ui,102,simBWF.format("%.0f",config['padThickness']/0.001),true)
        simUI.setEditValue(ui,103,simBWF.format("%.0f",config['padWallHeight']/0.001),true)
        simUI.setEditValue(ui,104,simBWF.format("%.0f",config['padWallThickness']/0.001),true)
        simUI.setEditValue(ui,105,simBWF.format("%.0f",config['padVerticalOffset']/0.001),true)

        updateStartStopTriggerComboboxes()
        updateMasterCombobox()

        updateEnabledDisabledItems()
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateConveyor()
    local conf=readInfo()
    local length=conf['length']
    local width=conf['width']
    local padSpacing=conf['padSpacing']
    local bitCoded=conf['bitCoded']
    local height=conf['height']

    local plength=conf['padLength']
    local pwidth=conf['padWidth']
    local pthickness=conf['padThickness']
    local pwheight=conf['padWallHeight']
    local pwthickness=conf['padWallThickness']
    local pvoffset=conf['padVerticalOffset']

    setShapeSize(base,width,length,height)
    setShapeSize(baseBack,width,width,height)
    setShapeSize(baseFront,width,width,height)
    sim.setObjectPosition(path,model,{0,0,0})
    sim.setObjectPosition(base,model,{0,0,-height*0.5})
    sim.setObjectPosition(baseBack,model,{0,-length*0.5,-height*0.5})
    sim.setObjectPosition(baseFront,model,{0,length*0.5,-height*0.5})
    sim.setObjectPosition(path,model,{0,0,pvoffset})

    while true do
        local h=sim.getObjectChild(path,0)
        if h>=0 then
            sim.removeObject(h)
        else
            break
        end
    end

    sim.cutPathCtrlPoints(path,-1,0)
    local pts={}
    local prec=30
    for i=0,prec,1 do
        pts[i*11+1]=0
        pts[i*11+2]=-length*0.5-width*0.5*math.sin(i*math.pi/prec)
        pts[i*11+3]=-width*0.5*math.cos(i*math.pi/prec)
        pts[i*11+4]=0 --math.pi-i*math.pi/8
        pts[i*11+5]=0
        pts[i*11+6]=0
        pts[i*11+7]=1
        pts[i*11+8]=0
        pts[i*11+9]=3
        pts[i*11+10]=0.5
        pts[i*11+11]=0.5
    end
    for i=0,prec,1 do
        pts[(i+prec+1)*11+1]=0
        pts[(i+prec+1)*11+2]=length*0.5+width*0.5*math.sin(i*math.pi/prec)
        pts[(i+prec+1)*11+3]=width*0.5*math.cos(i*math.pi/prec)
        pts[(i+prec+1)*11+4]=0 --i*math.pi/8
        pts[(i+prec+1)*11+5]=0
        pts[(i+prec+1)*11+6]=0
        pts[(i+prec+1)*11+7]=1
        pts[(i+prec+1)*11+8]=0
        pts[(i+prec+1)*11+9]=3
        pts[(i+prec+1)*11+10]=0.5
        pts[(i+prec+1)*11+11]=0.5
    end
    sim.insertPathCtrlPoints(path,1,0,prec*2+2,pts)
    local l=sim.getPathLength(path)
    local cnt=math.floor(l/padSpacing)+1
    local dx=l/cnt

    -- Build first one pad:
    local pad=sim.copyPasteObjects({padBaseShape},0)[1]
    setShapeSize(pad,plength,pwidth,pthickness)
    if pwheight>0 then
        local walls={}
        walls[1]=sim.copyPasteObjects({padWallShape},0)[1]
        setShapeSize(walls[1],plength,pwthickness,pwheight)
        walls[2]=sim.copyPasteObjects({walls[1]},0)[1]

        walls[3]=sim.copyPasteObjects({padWallShape},0)[1]
        setShapeSize(walls[3],pwthickness,pwidth-pwthickness*2,pwheight)
        walls[4]=sim.copyPasteObjects({walls[3]},0)[1]
        sim.setObjectPosition(walls[1],pad,{0,(pwidth-pwthickness)*0.5,(pthickness+pwheight)*0.5})
        sim.setObjectPosition(walls[2],pad,{0,-(pwidth-pwthickness)*0.5,(pthickness+pwheight)*0.5})
        sim.setObjectPosition(walls[3],pad,{(plength-pwthickness)*0.5,0,(pthickness+pwheight)*0.5})
        sim.setObjectPosition(walls[4],pad,{-(plength-pwthickness)*0.5,0,(pthickness+pwheight)*0.5})
        walls[5]=pad
        pad=sim.groupShapes(walls)
        sim.reorientShapeBoundingBox(pad,-1)
    end
    sim.setObjectInt32Parameter(pad,sim.objintparam_visibility_layer,1+256)
    sim.setObjectProperty(pad,sim.objectproperty_selectable+sim.objectproperty_selectmodelbaseinstead)
    sim.setObjectInt32Parameter(pad,sim.shapeintparam_respondable,1)
    sim.setObjectSpecialProperty(pad,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
    sim.setObjectPosition(pad,padBase,{(pthickness+pwheight)*0.5,plength*0.5,0})
    sim.setObjectOrientation(pad,padBase,{0,-math.pi/2,math.pi/2})
    -- Duplicate it and put the copies in place:
    for i=0,cnt-1,1 do
        local pb=sim.copyPasteObjects({padBase},0)[1]
        local p=sim.copyPasteObjects({pad},0)[1]
        sim.setObjectParent(p,pb,true)
        sim.setObjectParent(pb,path,true)
        sim.setObjectInt32Parameter(p,sim.objintparam_visibility_layer,1+256)
        sim.setObjectInt32Parameter(p,sim.shapeintparam_respondable,1)
        sim.setObjectSpecialProperty(p,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectFloatParameter(pb,sim.dummyfloatparam_follow_path_offset,i*dx)
    end
    -- Delete the first pad:
    sim.removeObject(pad)
end

function lengthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.4 then l=0.4 end
        if l>5 then l=5 end
        if l~=c['length'] then
            simBWF.markUndoPoint()
            c['length']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function widthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.05 then l=0.05 end
        if l>2 then l=2 end
        if l~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function heightChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.01 then l=0.01 end
        if l>1 then l=1 end
        if l~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padSpacingChange(ui,id,newVal)
    local c=readInfo()
    local w=tonumber(newVal)
    if w then
        w=w*0.001
        if w<0.02 then w=0.02 end
        if w>5 then w=5 end
        if w~=c['padSpacing'] then
            simBWF.markUndoPoint()
            c['padSpacing']=w
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padLengthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.02 then l=0.02 end
        if l<c['padWallThickness']*2 then l=c['padWallThickness']*2 end
        if l>2 then l=2 end
        if l~=c['padLength'] then
            simBWF.markUndoPoint()
            c['padLength']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padWidthChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.02 then l=0.02 end
        if l<c['padWallThickness']*2 then l=c['padWallThickness']*2 end
        if l>2 then l=2 end
        if l~=c['padWidth'] then
            simBWF.markUndoPoint()
            c['padWidth']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padThicknessChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.2 then l=0.2 end
        if l~=c['padThickness'] then
            simBWF.markUndoPoint()
            c['padThickness']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padWallThicknessChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        local m=math.min(c['padWidth'],c['padLength'])*0.5
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>m then l=m end
        if l~=c['padWallThickness'] then
            simBWF.markUndoPoint()
            c['padWallThickness']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padWallHeightChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0 end
        if l>0.2 then l=0.2 end
        if l~=c['padWallHeight'] then
            simBWF.markUndoPoint()
            c['padWallHeight']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function padVerticalOffsetChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<-0.3 then l=-0.3 end
        if l>0.3 then l=0.3 end
        if l~=c['padVerticalOffset'] then
            simBWF.markUndoPoint()
            c['padVerticalOffset']=l
            writeInfo(c)
            updateConveyor()
        end
    end
    setDlgItemContent()
end

function redChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor1()
    setColor1(newVal/100,g,b,s)
end

function greenChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor1()
    setColor1(r,newVal/100,b,s)
end

function blueChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor1()
    setColor1(r,g,newVal/100,s)
end

function specularChange1(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor1()
    setColor1(r,g,b,newVal/100)
end

function redChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor2()
    setColor2(newVal/100,g,b,s)
end

function greenChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor2()
    setColor2(r,newVal/100,b,s)
end

function blueChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor2()
    setColor2(r,g,newVal/100,s)
end

function specularChange2(ui,id,newVal)
    simBWF.markUndoPoint()
    local r,g,b,s=getColor2()
    setColor2(r,g,b,newVal/100)
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
    setDlgItemContent()
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
    setDlgItemContent()
end

function enabledClicked(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],64)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-64
    end
    simBWF.markUndoPoint()
    writeInfo(c)
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="77">
    <tab title="General" layout="form">

                <label text="Enabled"/>
                <checkbox text="" on-change="enabledClicked" id="1000"/>

                <label text="Speed (mm/s)"/>
                <edit on-editing-finished="speedChange" id="10"/>

                <label text="Acceleration (mm/s^2)"/>
                <edit on-editing-finished="accelerationChange" id="12"/>

                <label text="Master conveyor"/>
                <combobox id="202" on-change="masterChange_callback">
                </combobox>

                <label text="Stop on trigger"/>
                <combobox id="200" on-change="triggerStopChange_callback">
                </combobox>

                <label text="Restart on trigger"/>
                <combobox id="201" on-change="triggerStartChange_callback">
                </combobox>

            <label text="" style="* {margin-left: 150px;}"/>
            <label text="" style="* {margin-left: 150px;}"/>
    </tab>
    <tab title="Dimensions"  layout="form">
                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange" id="2"/>

                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange" id="4"/>

                <label text="Height (mm)"/>
                <edit on-editing-finished="heightChange" id="28"/>
    </tab>
    <tab title="Pads"  layout="form">
                <label text="Pad length (mm)"/>
                <edit on-editing-finished="padLengthChange" id="100"/>

                <label text="Pad width (mm)"/>
                <edit on-editing-finished="padWidthChange" id="101"/>

                <label text="Pad thickness (mm)"/>
                <edit on-editing-finished="padThicknessChange" id="102"/>

                <label text="Pad wall height (mm)"/>
                <edit on-editing-finished="padWallHeightChange" id="103"/>

                <label text="Pad wall thickness (mm)"/>
                <edit on-editing-finished="padWallThicknessChange" id="104"/>

                <label text="Pad vertical offset (mm)"/>
                <edit on-editing-finished="padVerticalOffsetChange" id="105"/>

                <label text="Maximum pad spacing (mm)"/>
                <edit on-editing-finished="padSpacingChange" id="21"/>

                <label text="Actual pad spacing (mm)"/>
                <label text="xxx" id="26"/>
    </tab>
    <tab title="Pad colors">
        <tabs>
        <tab title="Pad base" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange1" id="5"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange1" id="6"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange1" id="7"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="specularChange1" id="8"/>
        </tab>
        <tab title="Pad walls" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange2" id="30"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange2" id="31"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange2" id="32"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="specularChange2" id="33"/>
        </tab>
        </tabs>
    </tab>
    </tabs>
        ]]
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos--[[,closeable,onCloseFunction,modal,resizable,activate,additionalUiAttribute--]])
 
        local red,green,blue,spec=getColor1()
        simUI.setSliderValue(ui,5,red*100,true)
        simUI.setSliderValue(ui,6,green*100,true)
        simUI.setSliderValue(ui,7,blue*100,true)
        simUI.setSliderValue(ui,8,spec*100,true)
        red,green,blue,spec=getColor2()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simUI.setSliderValue(ui,33,spec*100,true)
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
    base=sim.getObjectHandle('genericConveyorTypeD_base')
    baseBack=sim.getObjectHandle('genericConveyorTypeD_baseBack')
    baseFront=sim.getObjectHandle('genericConveyorTypeD_baseFront')
    padBase=sim.getObjectHandle('genericConveyorTypeD_padBase')
    padBaseShape=sim.getObjectHandle('genericConveyorTypeD_padBaseShape')
    padWallShape=sim.getObjectHandle('genericConveyorTypeD_padWallShape')
    path=sim.getObjectHandle('genericConveyorTypeD_path')

	
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