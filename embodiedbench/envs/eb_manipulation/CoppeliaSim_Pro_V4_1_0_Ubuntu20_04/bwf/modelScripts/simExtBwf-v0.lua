function simBWF.createOpenBox(size,baseThickness,wallThickness,density,inertiaCorrectionFact,static,respondable,color)
    local parts={}
    local dim={size[1],size[2],baseThickness}
    parts[1]=sim.createPureShape(0,16,dim,density*dim[1]*dim[2]*dim[3])
    sim.setObjectPosition(parts[1],-1,{0,0,baseThickness*0.5})
    dim={wallThickness,size[2],size[3]-baseThickness}
    parts[2]=sim.createPureShape(0,16,dim,density*dim[1]*dim[2]*dim[3])
    sim.setObjectPosition(parts[2],-1,{(size[1]-wallThickness)*0.5,0,baseThickness+dim[3]*0.5})
    parts[3]=sim.createPureShape(0,16,dim,density*dim[1]*dim[2]*dim[3])
    sim.setObjectPosition(parts[3],-1,{(-size[1]+wallThickness)*0.5,0,baseThickness+dim[3]*0.5})
    dim={size[1]-2*wallThickness,wallThickness,size[3]-baseThickness}
    parts[4]=sim.createPureShape(0,16,dim,density*dim[1]*dim[2]*dim[3])
    sim.setObjectPosition(parts[4],-1,{0,(size[2]-wallThickness)*0.5,baseThickness+dim[3]*0.5})
    parts[5]=sim.createPureShape(0,16,dim,density*dim[1]*dim[2]*dim[3])
    sim.setObjectPosition(parts[5],-1,{0,(-size[2]+wallThickness)*0.5,baseThickness+dim[3]*0.5})
    for i=1,#parts,1 do
        sim.setShapeColor(parts[i],'',sim.colorcomponent_ambient_diffuse,color)
    end
    local shape=sim.groupShapes(parts)
    if math.abs(1-inertiaCorrectionFact)>0.001 then
        local transf=sim.getObjectMatrix(shape,-1)
        local m0,i0,com0=sim.getShapeMassAndInertia(shape,transf)
        for i=1,#i0,1 do
            i0[i]=i0[1]*inertiaCorrectionFact
        end
        sim.setShapeMassAndInertia(shape,m0,i0,com0,transf)
    end
    if static then
        sim.setObjectInt32Parameter(shape,sim.shapeintparam_static,1)
    else
        sim.setObjectInt32Parameter(shape,sim.shapeintparam_static,0)
    end
    if respondable then
        sim.setObjectInt32Parameter(shape,sim.shapeintparam_respondable,1)
    else
        sim.setObjectInt32Parameter(shape,sim.shapeintparam_respondable,0)
    end
    sim.reorientShapeBoundingBox(shape,-1)
    return shape
end

function simBWF.generatePalletPoints(objectData)
    local isCentered=true
    local allItems={}
    local tp=objectData['palletPattern']
    if tp and tp>0 then
        if tp==1 then -- single
            local d=objectData['singlePatternData']
            allItems=simBWF.getSinglePalletPoint(d)
        end
        if tp==2 then -- circular
            local d=objectData['circularPatternData3']
            local off=d[1]
            local radius=d[2]
            local cnt=d[3]
            local angleOff=d[4]
            local center=d[5]
            local layers=d[6]
            local layersStep=d[7]
            allItems=simBWF.getCircularPalletPoints(radius,cnt,angleOff,center,layers,layersStep,off)
        end
        if tp==3 then -- rectangular
            local d=objectData['linePatternData']
            local off=d[1]
            local rows=d[2]
            local rowStep=d[3]
            local cols=d[4]
            local colStep=d[5]
            local layers=d[6]
            local layersStep=d[7]
            allItems=simBWF.getLinePalletPoints(rows,rowStep,cols,colStep,layers,layersStep,isCentered,off)
        end
        if tp==4 then -- honeycomb
            local d=objectData['honeycombPatternData']
            local off=d[1]
            local rows=d[2]
            local rowStep=d[3]
            local cols=d[4]
            local colStep=d[5]
            local layers=d[6]
            local layersStep=d[7]
            local firstRowIsOdd=d[8]
            allItems=simBWF.getHoneycombPalletPoints(rows,rowStep,cols,colStep,layers,layersStep,firstRowIsOdd,isCentered,off)
        end
        if tp==5 then -- custom/imported
            allItems=objectData['palletPoints'] -- leave it as it is
        end
    end
    return allItems
end

function simBWF.getCircularPalletPoints(radius,count,angleOffset,center,layers,layerStep,optionalGlobalOffset)
    local retVal={}
    if not optionalGlobalOffset then
        optionalGlobalOffset={0,0,0}    
    end
    local da=2*math.pi/count
    for j=1,layers,1 do
        for i=0,count-1,1 do
            local decItem=simBWF.getOneRawPalletItem()
            local relP={optionalGlobalOffset[1]+radius*math.cos(da*i+angleOffset),optionalGlobalOffset[2]+radius*math.sin(da*i+angleOffset),optionalGlobalOffset[3]+(j-1)*layerStep}
            decItem['pos']=relP
            decItem['ser']=#retVal
            decItem['layer']=j
            retVal[#retVal+1]=decItem
        end
        if center then -- the center point
            local decItem=simBWF.getOneRawPalletItem()
            local relP={optionalGlobalOffset[1],optionalGlobalOffset[2],optionalGlobalOffset[3]+(j-1)*layerStep}
            decItem['pos']=relP
            decItem['ser']=#retVal
            decItem['layer']=j
            retVal[#retVal+1]=decItem
        end
    end
    return retVal
end

function simBWF.readPartInfoV0(handle)
    local data=sim.readCustomDataBlock(handle,simBWF.modelTags.PART)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end
    
    data['labelInfo']=nil -- not used anymore
    data['weightDistribution']=nil -- not supported anymore (now part of the feeder distribution algo)
    if not data['version'] then
        data['version']=0
    end
    if not data['name'] then
        data['name']='<partName>'
    end
    if not data['destination'] then
        data['destination']='<defaultDestination>'
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=invisible, 2=non-respondable to other parts
    end
    if not data['palletPattern'] then
        data['palletPattern']=0 -- 0=none, 1=single, 2=circular, 3=line (rectangle), 4=honeycomb, 5=custom/imported
    end
    if not data['circularPatternData3'] then
        data['circularPatternData3']={{0,0,0},0.1,6,0,true,1,0.05} -- offset, radius, count,angleOffset, center, layers, layers step
    end
    if not data['customPatternData'] then
        data['customPatternData']=''
    end
    if not data['linePatternData'] then
        data['linePatternData']={{0,0,0},3,0.03,3,0.03,1,0.05} -- offset, rowCnt, rowStep, colCnt, colStep, layers, layers step
    end
    if not data['honeycombPatternData'] then
        data['honeycombPatternData']={{0,0,0},3,0.03,3,0.03,1,0.05,false} -- offset, rowCnt, rowStep, colCnt, colStep, layers, layers step, firstRowOdd
    end
    if not data['palletPoints'] then
        data['palletPoints']={}
    end

    return data
end

function simBWF.palletPointsToString(palletPoints)
    local txt=""
    for i=1,#palletPoints,1 do
        local pt=palletPoints[i]
        if i~=1 then
            txt=txt..",\n"
        end
        txt=txt.."{{"..pt['pos'][1]..","..pt['pos'][2]..","..pt['pos'][3].."},"
        txt=txt.."{"..pt['orient'][1]..","..pt['orient'][2]..","..pt['orient'][3].."},"
        txt=txt..pt['layer'].."}"
    end
    return txt
end

function simBWF.stringToPalletPoints(txt)
    local palletPoints=nil
    local arr=stringToArray(txt)
    if arr then
        palletPoints={}
        for i=1,#arr,1 do
            local item=arr[i]
            if type(item)~='table' then return end
            if #item<3 then return end
            local pos=item[1]
            local orient=item[2]
            local layer=item[3]
            if type(pos)~='table' or #pos<3 then return end
            if type(orient)~='table' or #orient<3 then return end
            for j=1,3,1 do
                if type(pos[j])~='number' then return end
                if type(orient[j])~='number' then return end
            end
            if type(layer)~='number' then return end
            local decItem=simBWF.getOneRawPalletItem()
            decItem['pos']=pos
            decItem['orient']=orient
            decItem['ser']=#palletPoints
            decItem['layer']=layer
            palletPoints[#palletPoints+1]=decItem
        end
    end
    return palletPoints
end

function simBWF.arePalletPointsSame_posOrientAndLayer(pall1,pall2)
    if #pall1~=pall2 then return false end
    local distToll=0.0001*0.0001
    local orToll=0.05*math.pi/180
    orToll=orToll*orToll
    for i=1,#pall1,1 do
        local p1=pall1[i]
        local p2=pall2[i]
        if p1['layer']~=p2['layer'] then return false end
        local pos1=p1['pos']
        local pos2=p2['pos']
        local dx={pos1[1]-pos2[1],pos1[2]-pos2[2],pos1[3]-pos2[3]}
        local ll=dx[1]*dx[1]+dx[2]*dx[2]+dx[3]*dx[3]
        if ll>distToll then return false end
        pos1=p1['orient']
        pos2=p2['orient']
        dx={pos1[1]-pos2[1],pos1[2]-pos2[2],pos1[3]-pos2[3]}
        ll=dx[1]*dx[1]+dx[2]*dx[2]+dx[3]*dx[3]
        if ll>orToll then return false end
    end
    return true
end

function simBWF.readPalletFromFile(file)
    local json=require('dkjson')
    local file="d:/v_rep/qrelease/release/palletTest.txt"
    local f = io.open(file,"rb")
    local retVal=nil
    if f then
        f:close()
        local jsonData=''
        for line in io.lines(file) do
            jsonData=jsonData..line
        end
        jsonData='{ '..jsonData..' }'
        local obj,pos,err=json.decode(jsonData,1,nil)
        if type(obj)=='table' then
            if type(obj.frames)=='table'  then
                for j=1,#obj.frames,1 do
                    local fr=obj.frames[j]
                    if fr.rawPallet then
                        local palletItemList=fr.rawPallet.palletItemList
                        if palletItemList then
                            retVal={}
                            for itmN=1,#palletItemList,1 do
                                local item=palletItemList[itmN]
                                local decItem=simBWF.getOneRawPalletItem()
                                decItem['pos']={item.location.x*0.001,item.location.y*0.001,item.location.z*0.001}
                                decItem['orient']={item.roll,item.pitch,item.yaw}
                                decItem['ser']=#retVal
                                retVal[#retVal+1]=decItem
                            end
                            break
                        end
                    end
                end
            end
        end
    end
    return retVal
end

function simBWF.getAllPossiblePartDestinationsV0()
    local allDestinations={}
    -- First the parts from the part repository:
    local lst=simBWF.getAllPartsFromPartRepositoryV0()
    if lst then
        for i=1,#lst,1 do
            allDestinations[#allDestinations+1]=lst[i][1]
        end
    end
    -- The pingpong packer destination:
    local lst=getObjectsWithTag(simBWF.modelTags.CONVEYOR,true)
    for i=1,#lst,1 do
        local data=sim.readCustomDataBlock(lst[i],simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        if data['locationName'] then
            allDestinations[#allDestinations+1]=data['locationName']
        end
    end
    -- The thermoformer destination:
    for i=1,#lst,1 do
        local data=sim.readCustomDataBlock(lst[i],simBWF.modelTags.CONVEYOR)
        data=sim.unpackTable(data)
        if data['partName'] then
            allDestinations[#allDestinations+1]=data['partName']
        end
    end
    -- The location destination
    local lst=getObjectsWithTag(simBWF.modelTags.OLDLOCATION,true)
    for i=1,#lst,1 do
        local data=sim.readCustomDataBlock(lst[i],simBWF.modelTags.OLDLOCATION)
        data=sim.unpackTable(data)
        if data['name'] then
            allDestinations[#allDestinations+1]=data['name']
        end
    end
    return allDestinations
end

function simBWF.checkIfCodeAndModelMatch(modelHandle,codeVersion,modelVersion)
    if codeVersion~=modelVersion then
        sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Code and Model Version Mismatch","There is a mismatch between the code version and model version for:\n\nModel name: "..sim.getObjectName(modelHandle).."\nModel version: "..modelVersion.."\nCode version: "..codeVersion)
    end
end

function simBWF.getAllPossibleTriggerableFeeders(except)
    local allFeeders={}
    local allObjs=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    for i=1,#allObjs,1 do
        local h=allObjs[i]
        if h~=except then
            local data=sim.readCustomDataBlock(h,simBWF.modelTags.PARTFEEDER)
            if data then
                data=sim.unpackTable(data)
                if sim.boolAnd32(data['bitCoded'],4+8+16)==16 then
                    allFeeders[#allFeeders+1]={sim.getObjectName(h),h}
                end
            else
                data=sim.readCustomDataBlock(h,simBWF.modelTags.MULTIFEEDER)
                if data then
                    data=sim.unpackTable(data)
                    if sim.boolAnd32(data['bitCoded'],4+8+16)==16 then
                        allFeeders[#allFeeders+1]={sim.getObjectName(h),h}
                    end
                end
            end
        end
    end
    return allFeeders
end
