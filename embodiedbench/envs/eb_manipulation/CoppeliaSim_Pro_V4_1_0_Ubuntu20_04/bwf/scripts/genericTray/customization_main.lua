function model.setColor1(red,green,blue,spec)
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.handle,nil,sim.colorcomponent_specular,{spec,spec,spec})
    local c=model.readInfo()
    if sim.boolAnd32(c.partSpecific['bitCoded'],1)~=0 then
        model.setColor2(red,green,blue,spec)
        simUI.setSliderValue(model.dlg.ui,30,red*100,true)
        simUI.setSliderValue(model.dlg.ui,31,green*100,true)
        simUI.setSliderValue(model.dlg.ui,32,blue*100,true)
        simUI.setSliderValue(model.dlg.ui,33,spec*100,true)
    end
end

function model.getColor1()
    local r,rgb=sim.getShapeColor(model.handle,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model.handle,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function model.setColor2(red,green,blue,spec)
    sim.setShapeColor(model.specHandles.borderElement,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.borderElement,nil,sim.colorcomponent_specular,{spec,spec,spec})
    sim.setShapeColor(model.specHandles.border,nil,sim.colorcomponent_ambient_diffuse,{red,green,blue})
    sim.setShapeColor(model.specHandles.border,nil,sim.colorcomponent_specular,{spec,spec,spec})
end

function model.getColor2()
    local r,rgb=sim.getShapeColor(model.specHandles.borderElement,nil,sim.colorcomponent_ambient_diffuse)
    local r,spec=sim.getShapeColor(model.specHandles.borderElement,nil,sim.colorcomponent_specular)
    return rgb[1],rgb[2],rgb[3],(spec[1]+spec[2]+spec[3])/3
end

function model.setShapeSize(h,x,y,z)
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

function model.setShapeMass(handle,m)
    local transf=sim.getObjectMatrix(handle,-1)
    local m0,i0,com0=sim.getShapeMassAndInertia(handle,transf)
    sim.setShapeMassAndInertia(handle,m,{0.01*m,0,0,0,0.01*m,0,0,0,0.01*m},{0,0,0},transf)
end

function model.updateTray()
    local c=model.readInfo()
    local width=c.partSpecific['width']
    local length=c.partSpecific['length']
    local height=c.partSpecific['height']
    local borderThickness=c.partSpecific['borderThickness']
    local borderHeight=c.partSpecific['borderHeight']
    local mass=c.partSpecific['mass']
    local palletPoints={}
    local palletOffset=c.partSpecific['placeOffset']
    model.setShapeSize(model.handle,width,length,height)
    if model.specHandles.border~=-1 then
        sim.removeObject(model.specHandles.border)
        model.specHandles.border=-1
    end
    local borders={}
    if borderHeight>0 then
        borders[1]=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
        borders[2]=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
        borders[3]=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
        borders[4]=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
        model.setShapeSize(borders[1],width,borderThickness,borderHeight)
        model.setShapeSize(borders[2],width,borderThickness,borderHeight)
        model.setShapeSize(borders[3],borderThickness,length,borderHeight)
        model.setShapeSize(borders[4],borderThickness,length,borderHeight)
        sim.setObjectPosition(borders[1],model.handle,{0,-(length-borderThickness)*0.5,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[2],model.handle,{0,(length-borderThickness)*0.5,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[3],model.handle,{-(width-borderThickness)*0.5,0,(height+borderHeight)*0.5})
        sim.setObjectPosition(borders[4],model.handle,{(width-borderThickness)*0.5,0,(height+borderHeight)*0.5})
    end
    if c.partSpecific['pocketType']==0 then
        palletPoints=simBWF.getSinglePalletPoint(palletOffset)
    end

    if c.partSpecific['pocketType']==1 then
        local pp=c.partSpecific['linePocket']
        local ph=pp[1] -- height
        local pt=pp[2] -- thickness
        local pr=pp[3] -- rows
        local pc=pp[4] -- cols
        local w=width-borderThickness*2
        local l=length-borderThickness*2
        local rs=w/pr
        local rss=-w/2
        for i=1,pr-1,1 do
            local h=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
            model.setShapeSize(h,pt,l,ph)
            sim.setObjectPosition(h,model.handle,{rss+rs*i,0,(height+ph)*0.5})
            borders[#borders+1]=h
        end
        rs=l/pc
        rss=-l/2
        for i=1,pc-1,1 do
            local h=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
            model.setShapeSize(h,w,pt,ph)
            sim.setObjectPosition(h,model.handle,{0,rss+rs*i,(height+ph)*0.5})
            borders[#borders+1]=h
        end

        palletPoints=simBWF.getLinePalletPoints(pr,w/pr,pc,l/pc,1,0,true,palletOffset)
   end
    if c.partSpecific['pocketType']==2 then
        local pp=c.partSpecific['honeyPocket']
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
            local h=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
            model.setShapeSize(h,w,pt,ph)
            sim.setObjectPosition(h,model.handle,{0,rss+rs*i,(height+ph)*0.5})
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
                local h=sim.copyPasteObjects({model.specHandles.borderElement},0)[1]
                model.setShapeSize(h,pt,l/pc,ph)
                sim.setObjectPosition(h,model.handle,{-w*0.5+indent[li]+sss*(j-1),rss+(rs*0.5)+rs*(i-1),(height+ph)*0.5})
                borders[#borders+1]=h
            end
        end

        palletPoints=simBWF.getHoneycombPalletPoints(pr,w/pr,pc,l/pc,1,0,firstRowOdd,true,palletOffset)
   end

    if #borders>0 then
        mass=mass*0.5
        model.specHandles.border=sim.groupShapes(borders)
        sim.setObjectParent(model.specHandles.border,model.specHandles.connection,true)
        sim.setObjectInt32Parameter(model.specHandles.border,sim.objintparam_visibility_layer,1+256)
        sim.setObjectSpecialProperty(model.specHandles.border,sim.objectspecialproperty_collidable+sim.objectspecialproperty_measurable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_renderable)
        sim.setObjectInt32Parameter(model.specHandles.border,sim.shapeintparam_static,0)
        sim.setObjectInt32Parameter(model.specHandles.border,sim.shapeintparam_respondable,1)
        local p=sim.boolOr32(sim.getObjectProperty(model.specHandles.border),sim.objectproperty_dontshowasinsidemodel)-sim.objectproperty_dontshowasinsidemodel
        sim.setObjectProperty(model.specHandles.border,p)
        model.setShapeMass(model.specHandles.border,mass)
    end
    model.setShapeMass(model.handle,mass)
    -- Following sets the part tag, with palletPoints:
    local c=simBWF.readPartInfo(model.handle)
    c['name']='TRAY'
    c['palletPoints']=palletPoints
    c['palletPattern']=5 -- 5=imported
    simBWF.writePartInfo(model.handle,c)
end


function sysCall_cleanup_specific()
    sim.removeObject(model.specHandles.borderElement) 
end

