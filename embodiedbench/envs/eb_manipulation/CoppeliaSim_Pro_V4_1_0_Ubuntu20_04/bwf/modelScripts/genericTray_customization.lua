function removeFromPluginRepresentation()

end

function updatePluginRepresentation()

end

function getDefaultInfoForNonExistingFields(info)
    if not info['version'] then
        info['version']=_MODELVERSION_
    end
    if not info['subtype'] then
        info['subtype']='tray'
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
    if not info['bitCoded'] then
        info['bitCoded']=1 -- 1:base+borders have same color
    end
    if not info['borderHeight'] then
        info['borderHeight']=0.05
    end
    if not info['borderThickness'] then
        info['borderThickness']=0.005
    end
    if not info['mass'] then
        info['mass']=0.5
    end
    if not info['pocketType'] then
        info['pocketType']=0 -- 0=none, 1=rectangle, 2=honeycomb
    end
    if not info['linePocket'] then
        info['linePocket']={0.01,0.005,3,3} -- height, thickness, row, col
    end
    if not info['honeyPocket'] then
        info['honeyPocket']={0.01,0.005,3,3,false} -- height, thickness, row, col, first is odd
    end
    if not info['placeOffset'] then
        info['placeOffset']={0,0,0}
    end
end

function readInfo()
    local data=sim.readCustomDataBlock(model,'XYZ_TRAY_INFO')
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
        sim.writeCustomDataBlock(model,'XYZ_TRAY_INFO',sim.packTable(data))
    else
        sim.writeCustomDataBlock(model,'XYZ_TRAY_INFO','')
    end
end

function setColor1(red,green,blue,spec)
    sim.setShapeColor(model,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model,nil,sim.colorcomponent_specular,{spec,spec,spec})
    local c=readInfo()
    if sim.boolAnd32(c['bitCoded'],1)~=0 then
        setColor2(red,green,blue,spec)
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simUI.setSliderValue(ui,33,spec*100,true)
    end
end

function getColor1()
    local r,rgb=sim.getShapeColor(model,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function setColor2(red,green,blue,spec)
    sim.setShapeColor(borderElement,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(borderElement,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(border,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(border,nil,sim.colorcomponent_specular,{spec,spec,spec})
end

function getColor2()
    local r,rgb=sim.getShapeColor(borderElement,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(borderElement,nil,sim.colorcomponent_specular)
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

setShapeMass=function(handle,m)
    local transf=sim.getObjectMatrix(handle,-1)
    local m0,i0,com0=sim.getShapeMassAndInertia(handle,transf)
    sim.setShapeMassAndInertia(handle,m,{0.01*m,0,0,0,0.01*m,0,0,0,0.01*m},{0,0,0},transf)
end

function setDlgItemContent()
    if ui then
        local config=readInfo()
        local sel=simBWF.getSelectedEditWidget(ui)
        simUI.setEditValue(ui,1,simBWF.format("%.0f",config['width']/0.001),true)
        simUI.setEditValue(ui,2,simBWF.format("%.0f",config['length']/0.001),true)
        simUI.setEditValue(ui,3,simBWF.format("%.0f",config['height']/0.001),true)
        simUI.setEditValue(ui,4,simBWF.format("%.0f",config['borderHeight']/0.001),true)
        simUI.setEditValue(ui,5,simBWF.format("%.0f",config['borderThickness']/0.001),true)
        simUI.setEditValue(ui,6,simBWF.format("%.2f",config['mass']),true)
        simUI.setCheckboxValue(ui,19,simBWF.getCheckboxValFromBool(sim.boolAnd32(config['bitCoded'],1)~=0),true)
        local off=config['placeOffset']
        simUI.setEditValue(ui,40,simBWF.format("%.0f , %.0f , %.0f",off[1]*1000,off[2]*1000,off[3]*1000),true)

        simUI.setEnabled(ui,29,sim.boolAnd32(config['bitCoded'],1)==0,true)
        local pocketT=config['pocketType']
        simUI.setEnabled(ui,204,pocketT==1,true)
        simUI.setEnabled(ui,205,pocketT==2,true)
        simUI.setRadiobuttonValue(ui,103,simBWF.getRadiobuttonValFromBool(pocketT==0),true)
        simUI.setRadiobuttonValue(ui,104,simBWF.getRadiobuttonValFromBool(pocketT==1),true)
        simUI.setRadiobuttonValue(ui,105,simBWF.getRadiobuttonValFromBool(pocketT==2),true)

        local lineP=config['linePocket']
        simUI.setEditValue(ui,4000,simBWF.format("%.0f",lineP[1]/0.001),true)
        simUI.setEditValue(ui,4001,simBWF.format("%.0f",lineP[2]/0.001),true)
        simUI.setEditValue(ui,4002,tostring(lineP[3]),true)
        simUI.setEditValue(ui,4003,tostring(lineP[4]),true)

        local honeyP=config['honeyPocket']
        simUI.setEditValue(ui,5000,simBWF.format("%.0f",honeyP[1]/0.001),true)
        simUI.setEditValue(ui,5001,simBWF.format("%.0f",honeyP[2]/0.001),true)
        simUI.setEditValue(ui,5002,tostring(honeyP[3]),true)
        simUI.setEditValue(ui,5003,tostring(honeyP[4]),true)
        simUI.setCheckboxValue(ui,5004,simBWF.getCheckboxValFromBool(honeyP[5]),true)
        simBWF.setSelectedEditWidget(ui,sel)
    end
end

function updateTray()
    local c=readInfo()
    local width=c['width']
    local length=c['length']
    local height=c['height']
    local borderThickness=c['borderThickness']
    local borderHeight=c['borderHeight']
    local mass=c['mass']
    local palletPoints={}
    local palletOffset=c['placeOffset']
    setShapeSize(model,width,length,height)
    if border~=-1 then
        sim.removeObject(border)
        border=-1
    end
    local borders={}
    if borderHeight>0 then
        borders[1]=sim.copyPasteObjects({borderElement},0)[1]
        borders[2]=sim.copyPasteObjects({borderElement},0)[1]
        borders[3]=sim.copyPasteObjects({borderElement},0)[1]
        borders[4]=sim.copyPasteObjects({borderElement},0)[1]
        setShapeSize(borders[1],width,borderThickness,borderHeight)
        setShapeSize(borders[2],width,borderThickness,borderHeight)
        setShapeSize(borders[3],borderThickness,length,borderHeight)
        setShapeSize(borders[4],borderThickness,length,borderHeight)
        sim.setObjectPosition(borders[1],model,{0,-(length-borderThickness)*0.5,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[2],model,{0,(length-borderThickness)*0.5,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[3],model,{-(width-borderThickness)*0.5,0,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[4],model,{(width-borderThickness)*0.5,0,(height+borderHeight)*0.5})
    end
    if c['pocketType']==0 then
        palletPoints=simBWF.getSinglePalletPoint(palletOffset)
    end

    if c['pocketType']==1 then
        local pp=c['linePocket']
        local ph=pp[1] -- height
        local pt=pp[2] -- thickness
        local pr=pp[3] -- rows
        local pc=pp[4] -- cols
        local w=width-borderThickness*2
        local l=length-borderThickness*2
        local rs=w/pr
        local rss=-w/2
        for i=1,pr-1,1 do
            local h=sim.copyPasteObjects({borderElement},0)[1]
            setShapeSize(h,pt,l,ph)
            sim.setObjectPosition(h,model,{rss+rs*i,0,(height+ph)*0.5})
            borders[#borders+1]=h
        end
        rs=l/pc
        rss=-l/2
        for i=1,pc-1,1 do
            local h=sim.copyPasteObjects({borderElement},0)[1]
            setShapeSize(h,w,pt,ph)
            sim.setObjectPosition(h,model,{0,rss+rs*i,(height+ph)*0.5})
            borders[#borders+1]=h
        end

        palletPoints=simBWF.getLinePalletPoints(pr,w/pr,pc,l/pc,1,0,true,palletOffset)
   end
    if c['pocketType']==2 then
        local pp=c['honeyPocket']
        local ph=pp[1]
        local pt=pp[2]
        local pr=pp[3]
        local pc=pp[4]
        local firstRowOdd=pp[5]
        local w=width-borderThickness*2
        local l=length-borderThickness*2
        local rs=l/pc
        local rss=-l/2
        for i=1,pc-1,1 do
            local h=sim.copyPasteObjects({borderElement},0)[1]
            setShapeSize(h,w,pt,ph)
            sim.setObjectPosition(h,model,{0,rss+rs*i,(height+ph)*0.5})
            borders[#borders+1]=h
        end
        local sss=w/pr
        local rr={pr-1,pr}
        local indent={sss,sss*0.5}
        if sim.boolAnd32(pr,1)==0 then
            -- rows is even
            if not firstRowOdd then
                indent={sss*0.5,sss}
                rr={pr,pr-1}
            end
        else
            -- rows is odd
            if firstRowOdd then
                indent={sss*0.5,sss}
                rr={pr,pr-1}
            end
        end
        for i=1,pc,1 do
            local li=sim.boolAnd32(i,1)+1
            for j=1,rr[li],1 do
                local h=sim.copyPasteObjects({borderElement},0)[1]
                setShapeSize(h,pt,l/pc,ph)
                sim.setObjectPosition(h,model,{-w*0.5+indent[li]+sss*(j-1),rss+(rs*0.5)+rs*(i-1),(height+ph)*0.5})
                borders[#borders+1]=h
            end
        end

        palletPoints=simBWF.getHoneycombPalletPoints(pr,w/pr,pc,l/pc,1,0,firstRowOdd,true,palletOffset)
   end

    if #borders>0 then
        mass=mass*0.5
        border=sim.groupShapes(borders)
        sim.setObjectParent(border,connection,true)
        sim.setObjectInt32Parameter(border,sim.objintparam_visibility_layer,1+256)
        sim.setObjectSpecialProperty(border,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectInt32Parameter(border,sim.shapeintparam_static,0)
        sim.setObjectInt32Parameter(border,sim.shapeintparam_respondable,1)
        local p=sim.boolOr32(sim.getObjectProperty(border),sim.objectproperty_dontshowasinsidemodel)-sim.objectproperty_dontshowasinsidemodel
        sim.setObjectProperty(border,p)
        setShapeMass(border,mass)
    end
    setShapeMass(model,mass)
    -- Following sets the part tag, with palletPoints:
    local c=simBWF.readPartInfoV0(model)
    c['name']='TRAY'
    c['palletPoints']=palletPoints
    c['palletPattern']=5 -- 5=imported
    simBWF.writePartInfo(model,c)
end


function lengthChange(ui,id,newVal)
    local c=readInfo()
    local bt=c['borderThickness']
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.05 then l=0.05 end
        if l<bt*2+0.01 then l=bt*2+0.01 end
        if l>2 then l=2 end
        if l~=c['length'] then
            simBWF.markUndoPoint()
            c['length']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function widthChange(ui,id,newVal)
    local c=readInfo()
    local bt=c['borderThickness']
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.05 then l=0.05 end
        if l<bt*2+0.01 then l=bt*2+0.01 end
        if l>2 then l=2 end
        if l~=c['width'] then
            simBWF.markUndoPoint()
            c['width']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function heightChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c['height'] then
            simBWF.markUndoPoint()
            c['height']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function borderHeightChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0 end
        if l>1 then l=1 end
        if l~=c['borderHeight'] then
            simBWF.markUndoPoint()
            c['borderHeight']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function borderThicknessChange(ui,id,newVal)
    local c=readInfo()
    local mm=math.min(c['width'],c['length'])
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l>mm/2-0.01 then l=mm/2-0.01 end
        if l~=c['borderThickness'] then
            simBWF.markUndoPoint()
            c['borderThickness']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function massChange(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        if l<0.05 then l=0.05 end
        if l>10 then l=10 end
        if l~=c['mass'] then
            simBWF.markUndoPoint()
            c['mass']=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function sameColors_callback(ui,id,newVal)
    local c=readInfo()
    c['bitCoded']=sim.boolOr32(c['bitCoded'],1)
    if newVal==0 then
        c['bitCoded']=c['bitCoded']-1
    end
    simBWF.markUndoPoint()
    writeInfo(c)
    setColor1(getColor1())
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

function patternTypeClick_callback(ui,id)
    local c=readInfo()
    c['pocketType']=id-103
    simBWF.markUndoPoint()
    writeInfo(c)
    updateTray()
    setDlgItemContent()
end

function linePattern_heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c['linePocket'][1] then
            simBWF.markUndoPoint()
            c['linePocket'][1]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function linePattern_thicknessChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c['linePocket'][2] then
            simBWF.markUndoPoint()
            c['linePocket'][2]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function linePattern_rowsChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c['linePocket'][3] then
            simBWF.markUndoPoint()
            c['linePocket'][3]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function linePattern_colsChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c['linePocket'][4] then
            simBWF.markUndoPoint()
            c['linePocket'][4]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end




function honeyPattern_heightChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c['honeyPocket'][1] then
            simBWF.markUndoPoint()
            c['honeyPocket'][1]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function honeyPattern_thicknessChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=l*0.001
        if l<0.001 then l=0.001 end
        if l>0.1 then l=0.1 end
        if l~=c['honeyPocket'][2] then
            simBWF.markUndoPoint()
            c['honeyPocket'][2]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function honeyPattern_rowsChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c['honeyPocket'][3] then
            simBWF.markUndoPoint()
            c['honeyPocket'][3]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function honeyPattern_colsChange_callback(ui,id,newVal)
    local c=readInfo()
    local l=tonumber(newVal)
    if l then
        l=math.floor(l)
        if l<2 then l=2 end
        if l>20 then l=20 end
        if l~=c['honeyPocket'][4] then
            simBWF.markUndoPoint()
            c['honeyPocket'][4]=l
            writeInfo(c)
            updateTray()
        end
    end
    setDlgItemContent()
end

function honeyPattern_rowIsOddChange_callback(ui,id,newVal)
    local c=readInfo()
    c['honeyPocket'][5]=(newVal>0)
    simBWF.markUndoPoint()
    writeInfo(c)
    updateTray()
    setDlgItemContent()
end

function placeOffsetChange_callback(ui,id,newVal)
    local c=readInfo()
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
    c['placeOffset']={t[1],t[2],t[3]}
    simBWF.markUndoPoint()
    writeInfo(c)
    updateTray()
    setDlgItemContent()
end

function onCloseClicked()
    if sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_question,sim.msgbox_buttons_yesno,'Finalizing the tray',"By closing this customization dialog you won't be able to customize the tray anymore. Do you want to proceed?") then
        finalizeModel=true
        sim.removeScript(sim.handle_self)
    end
end

function createDlg()
    if (not ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
    <tabs id="77">
    <tab title="General properties" layout="form">
                <label text="Width (mm)"/>
                <edit on-editing-finished="widthChange" id="1"/>

                <label text="Length (mm)"/>
                <edit on-editing-finished="lengthChange" id="2"/>

                <label text="Base thickness (mm)"/>
                <edit on-editing-finished="heightChange" id="3"/>

                <label text="Border height (mm)"/>
                <edit on-editing-finished="borderHeightChange" id="4"/>

                <label text="Border thickness (mm)"/>
                <edit on-editing-finished="borderThicknessChange" id="5"/>

                <label text="Mass (Kg)"/>
                <edit on-editing-finished="massChange" id="6"/>
                
                <label text="Place offset (X, Y, Z, in mm)"/>
                <edit on-editing-finished="placeOffsetChange_callback" id="40"/>

                <label text="" style="* {margin-left: 150px;}"/>
                <label text="" style="* {margin-left: 150px;}"/>
    </tab>
    <tab title="Pockets">
        <tabs id="78">
            <tab title="None">
            <radiobutton text="Do not create any pockets" on-click="patternTypeClick_callback" id="103" />
            </tab>

            <tab title="Rectangle type">
            <radiobutton text="Create pockets arranged in a rectangular pattern" on-click="patternTypeClick_callback" id="104" />
            <group layout="form"  flat="true" id="204">
                <label text="Height (mm)"/>
                <edit on-editing-finished="linePattern_heightChange_callback" id="4000"/>

                <label text="Thickness (mm)"/>
                <edit on-editing-finished="linePattern_thicknessChange_callback" id="4001"/>

                <label text="Rows"/>
                <edit on-editing-finished="linePattern_rowsChange_callback" id="4002"/>

                <label text="Columns"/>
                <edit on-editing-finished="linePattern_colsChange_callback" id="4003"/>
            </group>
            </tab>

            <tab title="Honeycomb type">
            <radiobutton text="Create pockets arranged in a honeycomb pattern" on-click="patternTypeClick_callback" id="105" />
            <group layout="form"  flat="true"  id="205">
                <label text="Height (mm)"/>
                <edit on-editing-finished="honeyPattern_heightChange_callback" id="5000"/>

                <label text="Thickness (mm)"/>
                <edit on-editing-finished="honeyPattern_thicknessChange_callback" id="5001"/>

                <label text="Rows (longest)"/>
                <edit on-editing-finished="honeyPattern_rowsChange_callback" id="5002"/>

                <label text="Columns"/>
                <edit on-editing-finished="honeyPattern_colsChange_callback" id="5003"/>

                <label text="1st row is odd"/>
                <checkbox text="" on-change="honeyPattern_rowIsOddChange_callback" id="5004" />
            </group>
            </tab>
        </tabs>
    </tab>
    <tab title="Colors">
        <checkbox text="Base and borders have the same color" on-change="sameColors_callback" id="19" />
        <tabs>
        <tab title="Base" layout="form">
                <label text="Red"/>
                <hslider minimum="0" maximum="100" on-change="redChange1" id="20"/>
                <label text="Green"/>
                <hslider minimum="0" maximum="100" on-change="greenChange1" id="21"/>
                <label text="Blue"/>
                <hslider minimum="0" maximum="100" on-change="blueChange1" id="22"/>
                <label text="Specular"/>
                <hslider minimum="0" maximum="100" on-change="specularChange1" id="23"/>
        </tab>
        <tab title="Borders" layout="form" id="29">
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
        ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model,_MODELVERSION_,_CODEVERSION_),previousDlgPos,true,'onCloseClicked'--[[,modal,resizable,activate,additionalUiAttribute--]])

        local red,green,blue,spec=getColor1()
        simUI.setSliderValue(ui,20,red*100,true)
        simUI.setSliderValue(ui,21,green*100,true)
        simUI.setSliderValue(ui,22,blue*100,true)
        simUI.setSliderValue(ui,23,spec*100,true)
        red,green,blue,spec=getColor2()
        simUI.setSliderValue(ui,30,red*100,true)
        simUI.setSliderValue(ui,31,green*100,true)
        simUI.setSliderValue(ui,32,blue*100,true)
        simUI.setSliderValue(ui,33,spec*100,true)
        setDlgItemContent()

        local c=readInfo()
        local pattern=c['pocketType'] -- 0=none, 1=rectangle, 2=honeycomb
        local pat={}
        pat[0]=0
        pat[1]=1
        pat[2]=2
        simUI.setCurrentTab(ui,78,pat[pattern],true)
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
    connection=sim.getObjectHandle('genericTray_borderConnection')
    border=sim.getObjectChild(connection,0)
    borderElement=sim.getObjectHandle('genericTray_borderElement')
	
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
        sim.writeCustomDataBlock(model,'XYZ_TRAY_INFO','')
        sim.removeObject(borderElement) 
    end
    simBWF.writeSessionPersistentObjectData(model,"dlgPosAndSize",previousDlgPos,algoDlgSize,algoDlgPos,distributionDlgSize,distributionDlgPos,previousDlg1Pos)
end
