local simBWF={}

function simBWF.modifyPartDeactivationTime(currentDeactivationTime)
    local objs=sim.getObjectsInTree(sim.handle_scene,sim.handle_all)
    for i=1,#objs,1 do
        local dat=sim.readCustomDataBlock(objs[i],simBWF.modelTags.OLDOVERRIDE)
        if dat then
            dat=sim.unpackTable(dat)
            if sim.boolAnd32(dat['bitCoded'],4)>0 then
                return dat['deactivationTime']
            end
            break
        end
    end
    return currentDeactivationTime
end

function simBWF.modifyAuxVisualizationItems(visualize)
    local objs=sim.getObjectsInTree(sim.handle_scene,sim.handle_all)
    for i=1,#objs,1 do
        local dat=sim.readCustomDataBlock(objs[i],simBWF.modelTags.OLDOVERRIDE)
        if dat then
            dat=sim.unpackTable(dat)
            local v=sim.boolAnd32(dat['bitCoded'],1+2)
            if v>0 then
                if v==1 then return false end
                if v==2 then return true end
            end
            break
        end
    end
    return visualize
end

function simBWF.canOpenPropertyDialog(modelHandle)
    local objs=sim.getObjectsInTree(sim.handle_scene,sim.handle_all)
    for i=1,#objs,1 do
        local dat=sim.readCustomDataBlock(objs[i],simBWF.modelTags.OLDOVERRIDE)
        if dat then
            dat=sim.unpackTable(dat)
            local v=sim.boolAnd32(dat['bitCoded'],16)
            if v>0 then
  --              sim.addStatusbarMessage("\nInfo: property dialog won't open, since it was disabled in the settings control center.\n")
            end
            return (v==0)
        end
    end
    return true
end

function simBWF.getOneRawPalletItem()
    local decItem={}
    decItem['pos']={0,0,0}
    decItem['orient']={0,0,0}
    decItem['processingStage']=0
    decItem['ser']=0
    decItem['layer']=1
    return decItem
end

function simBWF.getSinglePalletPoint(optionalGlobalOffset)
    if not optionalGlobalOffset then
        optionalGlobalOffset={0,0,0}
    end
    local decItem=simBWF.getOneRawPalletItem()
    decItem['pos']={optionalGlobalOffset[1],optionalGlobalOffset[2],optionalGlobalOffset[3]}
    return {decItem}
end


function simBWF.getLinePalletPoints(rows,rowStep,cols,colStep,layers,layerStep,pointsAreCentered,optionalGlobalOffset)
    local retVal={}
    local goff={0,0,0}
    if optionalGlobalOffset then
        goff={optionalGlobalOffset[1],optionalGlobalOffset[2],optionalGlobalOffset[3]}
    end
    if pointsAreCentered then
        goff[1]=goff[1]-(rows-1)*rowStep*0.5
        goff[2]=goff[2]-(cols-1)*colStep*0.5
    end
    for k=1,layers,1 do
        for j=1,cols,1 do
            for i=1,rows,1 do
                local decItem=simBWF.getOneRawPalletItem()
                local relP={goff[1]+(i-1)*rowStep,goff[2]+(j-1)*colStep,goff[3]+(k-1)*layerStep}
                decItem['pos']=relP
                decItem['ser']=#retVal
                decItem['layer']=k
                retVal[#retVal+1]=decItem
            end
        end
    end
    return retVal
end

function simBWF.getHoneycombPalletPoints(rows,rowStep,cols,colStep,layers,layerStep,firstRowIsOdd,pointsAreCentered,optionalGlobalOffset)
    local retVal={}
    local goff={0,0,0}
    if optionalGlobalOffset then
        goff={optionalGlobalOffset[1],optionalGlobalOffset[2],optionalGlobalOffset[3]}
    end
    local rowSize={rows,rows-1}
    if sim.boolAnd32(rows,1)==0 then
        -- max row is even
        if firstRowIsOdd then
            rowSize={rows-1,rows}
        end
    else
        -- max row is odd
        if not firstRowIsOdd then
            rowSize={rows-1,rows}
        end
    end
    local colOff=-(cols-1)*colStep*0.5
    local rowOffs={-(rowSize[1]-1)*rowStep*0.5,-(rowSize[2]-1)*rowStep*0.5}

    if not pointsAreCentered then
        goff[1]=goff[1]+(rowSize[1]-1)*rowStep*0.5
        goff[2]=goff[2]+(cols-1)*colStep*0.5
    end

    for k=1,layers,1 do
        for j=1,cols,1 do
            local r=rowSize[1+sim.boolAnd32(j-1,1)]
            for i=1,r,1 do
                local rowOff=rowOffs[1+sim.boolAnd32(j-1,1)]
                local decItem=simBWF.getOneRawPalletItem()
                local relP={goff[1]+rowOff+(i-1)*rowStep,goff[2]+colOff+(j-1)*colStep,goff[3]+(k-1)*layerStep}
                decItem['pos']=relP
                decItem['ser']=#retVal
                decItem['layer']=k
                retVal[#retVal+1]=decItem
            end
        end
    end
    return retVal
end


function simBWF._getPickPlaceSettingsDefaultInfoForNonExistingFields(info)
    if not info.overrideGripperSettings then
        info.overrideGripperSettings=false -- by default, we use the gripper settings
    end
    if not info.speed then
        info.speed=1 -- in %
    end
    if not info.accel then
        info.accel=1 -- in %
    end
    if not info.dynamics then
        info.dynamics=1 -- in %
    end
    if not info.dwellTime then
        info.dwellTime={0.1,0.1}
    end
    if not info.approachHeight then
        info.approachHeight={0.1,0.1}
    end
    if not info.useAbsoluteApproachHeight then
        info.useAbsoluteApproachHeight={false,false} -- makes only sensor for pick. No meaning (for now) for place
    end
    if not info.departHeight then
        info.departHeight={0.1,0.1}
    end
    if not info.offset then
        info.offset={{0,0,0},{0,0,0}}
    end
    if not info.rounding then
        info.rounding={0.05,0.05}
    end
    if not info.nullingAccuracy then
        info.nullingAccuracy={0.005,0.005}
    end
    if not info.actionTemplates then
        info.actionTemplates={release={cmd="M800M810"},activePick={cmd="M801M810"},activePlace={cmd="M800M811"}}
    end
    if not info.pickActions then
        info.pickActions={{name="release",dt=0},{name="activePick",dt=0.01}}
    end
    if not info.multiPickActions then
        info.multiPickActions={{name="release",dt=0},{name="activePick",dt=0.01}}
    end
    if not info.placeActions then
        info.placeActions={{name="activePlace",dt=0},{name="release",dt=0.01}}
    end
    if not info.relativeToBelt then
        info.relativeToBelt={false,false} -- makes only sensor for pick. No meaning (for now) for place
    end
    -- Following not supported anymore:
    info.freeModeTiming=nil
    info.actionModeTiming=nil
end

function simBWF.readPartInfo(handle)
    local data=sim.readCustomDataBlock(handle,simBWF.modelTags.PART)
    if data then
        data=sim.unpackTable(data)
    else
        data={}
    end

    -- Following few not supported anymore with V1
    data['labelInfo']=nil
    data['weightDistribution']=nil
    data['palletPattern']=nil
    data['circularPatternData3']=nil
    data['customPatternData']=nil
    data['linePatternData']=nil
    data['honeycombPatternData']=nil
    data['palletPoints']=nil
    data['name']=nil
    data['palletId']=nil
    data['destination']=nil
    data['notFinalized']=nil

    -- palletId is stored in the object referenced IDs
    if not data['vertMinMax'] then
        data['vertMinMax']={{0,0},{0,0},{0,0}}
    end
    if not data['version'] then
        data['version']=1
    end
    if not data['bitCoded'] then
        data['bitCoded']=0 -- 1=invisible, 2=non-respondable to other parts, 4=ignore base object (if associated with pallet), 8=use pallet colors, 16=attach to other parts
    end
    if not data['palletOffset'] then
        data['palletOffset']={0,0,0}
    end

    -- Following name (robotInfo) is not very good. It groups part pick/place settings. Place settings are ignored (since they are taken from the gripper of the pallet)
    if not data['robotInfo'] then
        data['robotInfo']={}
    end

    simBWF._getPickPlaceSettingsDefaultInfoForNonExistingFields(data.robotInfo)

    return data
end

function simBWF.writePartInfo(handle,data)
    if data then
        sim.writeCustomDataBlock(handle,simBWF.modelTags.PART,sim.packTable(data))
    else
        sim.writeCustomDataBlock(handle,simBWF.modelTags.PART,'')
    end
end

function simBWF.readPalletInfo(palletHandle)
    return sim.unpackTable(sim.readCustomDataBlock(palletHandle,simBWF.modelTags.PALLET))
end

function simBWF.writePalletInfo(palletHandle,data)
    sim.writeCustomDataBlock(palletHandle,simBWF.modelTags.PALLET,sim.packTable(data))
end


function simBWF.getPartRepositoryHandles()
    local repoP=sim.getObjectsWithTag(simBWF.modelTags.OLDPARTREPO,true) -- to support BR version 0
    if #repoP==0 then
        repoP=sim.getObjectsWithTag(simBWF.modelTags.PARTREPOSITORY,true)
    end

    if #repoP==1 then
        local repo=repoP[1]
        local suff=sim.getNameSuffix(sim.getObjectName(repo))
        local nm='partRepository_modelParts'
        if suff>=0 then
            nm=nm..'#'..suff
        end
        local partHolder=simBWF.getObjectHandle_noErrorNoSuffixAdjustment(nm)
        if partHolder>=0 then
            return repo,partHolder
        end
    end
end

function simBWF.getAllPartsFromPartRepositoryV0()
    local repo,partHolder=simBWF.getPartRepositoryHandles()
    if repo then
        local retVal={}
        local l=sim.getObjectsInTree(partHolder,sim.handle_all,1+2)
        for i=1,#l,1 do
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.PART)
            if data then
                data=sim.unpackTable(data)
                retVal[#retVal+1]={data['name'],l[i]}
            end
        end
        return retVal
    end
end

function simBWF.getAllPartsFromPartRepository()
    local repo,partHolder=simBWF.getPartRepositoryHandles()
    if repo then
        local retVal={}
        local l=sim.getObjectsInTree(partHolder,sim.handle_all,1+2)
        for i=1,#l,1 do
            local data=sim.readCustomDataBlock(l[i],simBWF.modelTags.PART)
            if data then
                data=sim.unpackTable(data)
                retVal[#retVal+1]={simBWF.getObjectAltName(l[i]),l[i]}
            end
        end
        return retVal
    end
end


function simBWF.removeTmpRem(txt)
    while true do
        local s=string.find(txt,"--%[%[tmpRem")
        if not s then break end
        local e=string.find(txt,"--%]%]",s+1)
        if not e then break end
        local tmp=''
        if s>1 then
            tmp=string.sub(txt,1,s-1)
        end
        tmp=tmp..string.sub(txt,e+4)
        txt=tmp
    end
    return txt
end


function simBWF.getAllPossiblePartDestinations()
    -- Returns a map with key: altName, value: handle
    local retVal={}

    local l=sim.getObjectsWithTag(simBWF.modelTags.LOCATIONFRAME,true)
    for i=1,#l,1 do
        retVal[sim.getObjectName(l[i]+sim.handleflag_altname)]=l[i]
    end
    local l=sim.getObjectsWithTag(simBWF.modelTags.TRACKINGWINDOW,true)
    for i=1,#l,1 do
        retVal[sim.getObjectName(l[i]+sim.handleflag_altname)]=l[i]
    end
    return retVal
end

function simBWF.getAllInstanciatedParts()
    local l=sim.getObjectsInTree(sim.handle_scene,sim.object_shape_type,0)
    local retL={}
    for i=1,#l,1 do
        local isPart,isInstanciated,data=simBWF.isObjectPartAndInstanciated(l[i])
        if isInstanciated then
            retL[#retL+1]=l[i]
        end
    end
    return retL
end

function simBWF.isObjectPartAndInstanciated(h)
    local data=sim.readCustomDataBlock(h,simBWF.modelTags.PART)
    if data then
        data=sim.unpackTable(data)
        return true, data.instanciated, data
    end
    return false, false, nil
end


function simBWF.getReferencedObjectHandle(modelHandle,index)
    local refH=sim.getReferencedHandles(modelHandle)
    if refH and #refH>=index then
        return refH[index]
    end
    return -1
end

function simBWF.setReferencedObjectHandle(modelHandle,index,referencedObjectHandle)
    local refH=sim.getReferencedHandles(modelHandle)
    if not refH then
        refH={}
    end
    while #refH<index do
        refH[#refH+1]=-1 -- pad with -1
    end
    refH[index]=referencedObjectHandle
    sim.setReferencedHandles(modelHandle,refH)
end

function simBWF.getObjectNameOrNone(objectHandle)
    if objectHandle>=0 then
        return sim.getObjectName(objectHandle)
    end
    return simBWF.NONE_TEXT
end

function simBWF.getObjectAltNameOrNone(objectHandle)
    if objectHandle>=0 then
        return simBWF.getObjectAltName(objectHandle)
    end
    return simBWF.NONE_TEXT
end

function simBWF.createCustomUi(nakedXml,title,dlgPos,closeable,onCloseFunction,modal,resizable,activate,additionalAttributes,dlgSize)
    -- Call utils function instead once version is stable
    local xml='<ui title="'..title..'" closeable="'
    if closeable then
        if onCloseFunction and onCloseFunction~='' then
            xml=xml..'true" on-close="'..onCloseFunction..'"'
        else
            xml=xml..'true"'
        end
    else
        xml=xml..'false"'
    end
    if modal then
        xml=xml..' modal="true"'
    else
        xml=xml..' modal="false"'
    end
    if resizable then
        xml=xml..' resizable="true"'
    else
        xml=xml..' resizable="false"'
    end
    if activate then
        xml=xml..' activate="true"'
    else
        xml=xml..' activate="false"'
    end
    if additionalAttributes and additionalAttributes~='' then
        xml=xml..' '..additionalAttributes
    end
    if dlgSize then
        xml=xml..' size="'..dlgSize[1]..','..dlgSize[2]..'"'
    end
    if not dlgPos then
        xml=xml..' placement="relative" position="-50,50">'
    else
        if type(dlgPos)=='string' then
            if dlgPos=='center' then
                xml=xml..' placement="center">'
            end
            if dlgPos=='bottomRight' then
                xml=xml..' placement="relative" position="-50,-50">'
            end
            if dlgPos=='bottomLeft' then
                xml=xml..' placement="relative" position="50,-50">'
            end
            if dlgPos=='topLeft' then
                xml=xml..' placement="relative" position="50,50">'
            end
            if dlgPos=='topRight' then
                xml=xml..' placement="relative" position="-50,50">'
            end
        else
            xml=xml..' placement="absolute" position="'..dlgPos[1]..','..dlgPos[2]..'">'
        end
    end
    xml=xml..nakedXml..'</ui>'
    local ui=simUI.create(xml)
    --[[
    if dlgSize then
        simUI.setSize(ui,dlgSize[1],dlgSize[2])
    end
    --]]
    if not activate then
        if 'linux'==simBWF.getPlatform() then
            -- To fix a Qt bug on Linux
            sim.auxFunc('activateMainWindow')
        end
    end
    return ui
end

function simBWF.populateCombobox(ui,id,items_array,exceptItems_map,currentItem,sort,additionalItemsToTop_array)
    local _itemsTxt={}
    local _itemsMap={}
    for i=1,#items_array,1 do
        local txt=items_array[i][1]
        if (not exceptItems_map) or (not exceptItems_map[txt]) then
            _itemsTxt[#_itemsTxt+1]=txt
            _itemsMap[txt]=items_array[i][2]
        end
    end
    if sort then
        table.sort(_itemsTxt)
    end
    local tableToReturn={}
    if additionalItemsToTop_array then
        for i=1,#additionalItemsToTop_array,1 do
            tableToReturn[#tableToReturn+1]={additionalItemsToTop_array[i][1],additionalItemsToTop_array[i][2]}
        end
    end
    for i=1,#_itemsTxt,1 do
        tableToReturn[#tableToReturn+1]={_itemsTxt[i],_itemsMap[_itemsTxt[i]]}
    end
    if additionalItemsToTop_array then
        for i=1,#additionalItemsToTop_array,1 do
            table.insert(_itemsTxt,i,additionalItemsToTop_array[i][1])
        end
    end
    local index=0
    for i=1,#_itemsTxt,1 do
        if _itemsTxt[i]==currentItem then
            index=i-1
            break
        end
    end
    simUI.setComboboxItems(ui,id,_itemsTxt,index,true)
    return tableToReturn,index
end

function simBWF.getSelectedEditWidget(ui)
    -- Call utils function instead once version is stable
    local ret=-1
    if simBWF.getVrepVersion()>30302 then
        ret=simUI.getCurrentEditWidget(ui)
    end
    return ret
end

function simBWF.setSelectedEditWidget(ui,id)
    -- Call utils function instead once version is stable
    if id>=0 then
        simUI.setCurrentEditWidget(ui,id)
    end
end

function simBWF.getRadiobuttonValFromBool(b)
    -- Call utils function instead once version is stable
    if b then
        return 1
    end
    return 0
end

function simBWF.getCheckboxValFromBool(b)
    -- Call utils function instead once version is stable
    if b then
        return 2
    end
    return 0
end


function simBWF.writeSessionPersistentObjectData(objectHandle,dataName,...)
    -- Call utils function instead once version is stable
    local data={...}
    local nm="___"..sim.getScriptHandle()..sim.getObjectName(objectHandle)..sim.getInt32Parameter(sim.intparam_scene_unique_id)..sim.getObjectStringParameter(objectHandle,sim.objstringparam_dna)..dataName
    data=sim.packTable(data)
    sim.writeCustomDataBlock(sim.handle_app,nm,data)
end

function simBWF.readSessionPersistentObjectData(objectHandle,dataName)
    -- Call utils function instead once version is stable
    local nm="___"..sim.getScriptHandle()..sim.getObjectName(objectHandle)..sim.getInt32Parameter(sim.intparam_scene_unique_id)..sim.getObjectStringParameter(objectHandle,sim.objstringparam_dna)..dataName
    local data=sim.readCustomDataBlock(sim.handle_app,nm)
    if data then
        data=sim.unpackTable(data)
        return unpack(data)
    else
        return nil
    end
end

function simBWF.getUiTitleNameFromModel(model,modelVersion,codeVersion)
    local retVal=sim.getObjectName(model+sim.handleflag_altname)
    if modelVersion then
        retVal=retVal.." (V"..modelVersion..")"
    end
    return retVal
end

function simBWF.getNormalizedVector(v)
    local l=math.sqrt(v[1]*v[1]+v[2]*v[2]+v[3]*v[3])
    return {v[1]/l,v[2]/l,v[3]/l}
end

function simBWF.getPtPtDistance(pt1,pt2)
    local p={pt2[1]-pt1[1],pt2[2]-pt1[2],pt2[3]-pt1[3]}
    return math.sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3])
end

function simBWF.getCrossProduct(v1,v2)
    local ret={}
    ret[1]=v1[2]*v2[3]-v1[3]*v2[2]
    ret[2]=v1[3]*v2[1]-v1[1]*v2[3]
    ret[3]=v1[1]*v2[2]-v1[2]*v2[1]
    return ret
end

function simBWF.getScaledVector(v,scalingFact)
    return {v[1]*scalingFact,v[2]*scalingFact,v[3]*scalingFact}
end

function simBWF.getModelMainTag(objHandle)
    local tags=sim.readCustomDataBlockTags(objHandle)
    if tags then
        for i=1,#tags,1 do
            if tags[i]==simBWF.modelTags.OUTPUTBOX then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.INPUTBOX then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.IOHUB then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.VISIONBOX then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.TESTMODEL then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.LOCATIONFRAME then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.TRACKINGWINDOW then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNAR then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.CONVEYOR then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.PARTFEEDER then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.PARTTAGGER then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.PACKML then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.BLUEREALITYAPP then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.PARTSINK then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.LIFT then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNARGRIPPER then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNARGRIPPERPLATFORM then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.VISIONWINDOW then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNARCAMERA then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNARSENSOR then
                return tags[i]
            end
            if tags[i]==simBWF.modelTags.RAGNARDETECTOR then
                return tags[i]
            end
        end
    end
    return ''
end

function simBWF.getModelTagsForMessages()
    local ret={}
    ret[1]=simBWF.modelTags.LOCATIONFRAME
    ret[2]=simBWF.modelTags.TRACKINGWINDOW
    ret[3]=simBWF.modelTags.RAGNAR
    ret[4]=simBWF.modelTags.CONVEYOR
    ret[5]=simBWF.modelTags.PARTFEEDER
    ret[6]=simBWF.modelTags.PARTTAGGER
    ret[7]=simBWF.modelTags.PARTSINK
    ret[8]=simBWF.modelTags.RAGNARGRIPPER
    ret[9]=simBWF.modelTags.RAGNARGRIPPERPLATFORM
    ret[10]=simBWF.modelTags.VISIONWINDOW
    ret[11]=simBWF.modelTags.RAGNARCAMERA
    ret[12]=simBWF.modelTags.RAGNARSENSOR
    ret[13]=simBWF.modelTags.RAGNARDETECTOR
    ret[14]=simBWF.modelTags.LIFT
    return ret
end

function simBWF.isSystemOnline()
    return sim.getIntegerSignal('__brOnline__')~=nil
end

function simBWF.isInTestMode()
    return sim.getIntegerSignal('__brTesting__')~=nil
end

function simBWF.markUndoPoint()
    local cnt=sim.getIntegerSignal('__brUndoPointCounter__')
    if cnt then
        sim.setIntegerSignal('__brUndoPointCounter__',cnt+1)
    end
end

function simBWF.outputMessage(msg,msgType)
    local msgs=sim.getStringSignal('__brMessages__')
    if not msgs then
        msgs={}
    else
        msgs=sim.unpackTable(msgs)
    end
    msgs[#msgs+1]={msg,msgType}
    sim.setStringSignal('__brMessages__',sim.packTable(msgs))
end

function simBWF.getSimulationOrOnlineTime()
    return sim.getFloatSignal('__brOnlineTime__')
end

function simBWF.getMatrixFromCalibrationBallPositions(ball1,ball2,ball3,relativeToVisionSensor)
    local calData={ball1,ball2,ball3}
    -- now set the location frame balls in place:
    -- normalized vector X:
    local x={calData[2][1]-calData[1][1],calData[2][2]-calData[1][2],calData[2][3]-calData[1][3]}
    x=simBWF.getNormalizedVector(x)
    -- normalized vector Z:
    local yp={calData[3][1]-calData[2][1],calData[3][2]-calData[2][2],calData[3][3]-calData[2][3]}
    local z=simBWF.getCrossProduct(x,yp)
    z=simBWF.getNormalizedVector(z)
    if z[3]<0 and not relativeToVisionSensor then
       z=simBWF.getScaledVector(z,-1) -- this is the case when the blue ball is on the 'other side'
    end
    -- normalized vector Y:
    local y=simBWF.getCrossProduct(z,x)
    -- Build the matrix:
    local m={x[1],y[1],z[1],calData[1][1],
            x[2],y[2],z[2],calData[1][2],
            x[3],y[3],z[3],calData[1][3]}
    return m
end

function simBWF.callScriptFunction_noError(funcName,objectId,scriptType,...)
    local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local funcNameAtScriptName=funcName..'@'..sim.getObjectName(objectId)
    local ret1,ret2,ret3,ret4,ret5,ret6,ret7,ret8=sim.callScriptFunction(funcNameAtScriptName,scriptType,...)
    sim.setInt32Parameter(sim.intparam_error_report_mode,err)
    return ret1,ret2,ret3,ret4,ret5,ret6,ret7,ret8
end

function simBWF.callScriptFunction(funcName,objectId,scriptType,...)
    local funcNameAtScriptName=funcName..'@'..sim.getObjectName(objectId)
    local ret1,ret2,ret3,ret4,ret5,ret6,ret7,ret8=sim.callScriptFunction(funcNameAtScriptName,scriptType,...)
    return ret1,ret2,ret3,ret4,ret5,ret6,ret7,ret8
end

function simBWF.callCustomizationScriptFunction(funcName,objectId,...)
    return simBWF.callScriptFunction(funcName,objectId,sim.scripttype_customizationscript,...)
end

function simBWF.callChildScriptFunction(funcName,objectId,...)
    return simBWF.callScriptFunction(funcName,objectId,sim.scripttype_childscript,...)
end

function simBWF.callCustomizationScriptFunction_noError(funcName,objectId,...)
    return simBWF.callScriptFunction_noError(funcName,objectId,sim.scripttype_customizationscript,...)
end

function simBWF.callChildScriptFunction_noError(funcName,objectId,...)
    return simBWF.callScriptFunction_noError(funcName,objectId,sim.scripttype_childscript,...)
end

function simBWF.getAvailablePallets()
    local repoP=sim.getObjectsWithTag(simBWF.modelTags.PALLETREPOSITORY,true)
    if #repoP==1 then
        local retData=simBWF.callCustomizationScriptFunction('model.ext.getListOfAvailablePallets',repoP[1])
        return retData
    end
    return {}
end

function simBWF.announcePalletWasRenamed(palletId)
    local modelTags={simBWF.modelTags.LOCATIONFRAME,simBWF.modelTags.PARTREPOSITORY,
                    simBWF.modelTags.RAGNARSENSOR,simBWF.modelTags.RAGNARDETECTOR,simBWF.modelTags.VISIONWINDOW
                    ,simBWF.modelTags.THERMOFORMER
                    }
    for j=1,#modelTags,1 do
        local allModelsOfThatType=sim.getObjectsWithTag(modelTags[j],true)
        for i=1,#allModelsOfThatType,1 do
            simBWF.callCustomizationScriptFunction('model.ext.announcePalletWasRenamed',allModelsOfThatType[i])
        end
    end
end

function simBWF.announcePalletWasCreated()
    local modelTags={simBWF.modelTags.LOCATIONFRAME,simBWF.modelTags.PARTREPOSITORY,
                    simBWF.modelTags.RAGNARSENSOR,simBWF.modelTags.RAGNARDETECTOR,simBWF.modelTags.VISIONWINDOW
                    ,simBWF.modelTags.THERMOFORMER
                    }
    for j=1,#modelTags,1 do
        local allModelsOfThatType=sim.getObjectsWithTag(modelTags[j],true)
        for i=1,#allModelsOfThatType,1 do
            simBWF.callCustomizationScriptFunction('model.ext.announcePalletWasCreated',allModelsOfThatType[i])
        end
    end
end

function simBWF.announcePalletWasDestroyed()
    local modelTags={simBWF.modelTags.LOCATIONFRAME,simBWF.modelTags.PARTREPOSITORY,
                    simBWF.modelTags.RAGNARSENSOR,simBWF.modelTags.RAGNARDETECTOR,simBWF.modelTags.VISIONWINDOW
                    ,simBWF.modelTags.THERMOFORMER
                    }
    for j=1,#modelTags,1 do
        local allModelsOfThatType=sim.getObjectsWithTag(modelTags[j],true)
        for i=1,#allModelsOfThatType,1 do
            simBWF.callCustomizationScriptFunction('model.ext.announcePalletWasDestroyed',allModelsOfThatType[i])
        end
    end
end

function simBWF.announceOnlineModeChanged(isNowOnline)
    -- 1. Location frames:
    local allLocationFrames=sim.getObjectsWithTag(simBWF.modelTags.LOCATIONFRAME,true)
    for i=1,#allLocationFrames,1 do
        simBWF.callCustomizationScriptFunction('model.ext.announceOnlineModeChanged',allLocationFrames[i],isOnlineNow)
    end
    -- 2. Tracking windows:
    local allTrackingWindows=sim.getObjectsWithTag(simBWF.modelTags.TRACKINGWINDOW,true)
    for i=1,#allTrackingWindows,1 do
        simBWF.callCustomizationScriptFunction('model.ext.announceOnlineModeChanged',allTrackingWindows[i],isOnlineNow)
    end
    -- 3. Vision box:
    local allVisionWindows=sim.getObjectsWithTag(simBWF.modelTags.VISIONBOX,true)
    for i=1,#allVisionWindows,1 do
        simBWF.callCustomizationScriptFunction('model.ext.refreshDlg', allVisionWindows[i])
    end
end

function simBWF.forbidInputForTrackingWindowChainItems(inputItem)
    local modelTags={simBWF.modelTags.TRACKINGWINDOW,simBWF.modelTags.VISIONWINDOW,simBWF.modelTags.RAGNARSENSOR,simBWF.modelTags.RAGNARDETECTOR}
    local objs=sim.getObjectsInTree(sim.handle_scene)
    for i=1,#objs,1 do
        local dat=sim.readCustomDataBlockTags(objs[i])
        if dat then
            local leave=false
            for j=1,#dat,1 do
                for k=1,#modelTags,1 do
                    if dat[j]==modelTags[k] then
                        simBWF.callCustomizationScriptFunction('model.ext.forbidInput',objs[i],inputItem)
                        leave=true
                        break
                    end
                end
                if leave then
                    break
                end
            end
        end
    end
end

function simBWF.getModelThatUsesThisModelAsInput(thisModelHandle)
    local modelTags={simBWF.modelTags.TRACKINGWINDOW,simBWF.modelTags.VISIONWINDOW,simBWF.modelTags.RAGNARSENSOR,simBWF.modelTags.RAGNARDETECTOR}
    for i=1,#modelTags,1 do
        local models=sim.getObjectsWithTag(modelTags[i],true)
        for j=1,#models,1 do
            local h=simBWF.callCustomizationScriptFunction('model.ext.getInputObjectHande',models[j])
            if h==thisModelHandle then
                return models[j]
            end
        end
    end
    return -1
end

function simBWF.getInputOutputBoxConnectedItem(boxHandle)
    local modelTags={simBWF.modelTags.IOHUB,simBWF.modelTags.RAGNAR}
    for i=1,#modelTags,1 do
        local models=sim.getObjectsWithTag(modelTags[i],true)
        for j=1,#models,1 do
            local connectionIndex=simBWF.callCustomizationScriptFunction('model.ext.isInputBoxConnection',models[j],boxHandle)
            if connectionIndex==-1 then
                connectionIndex=simBWF.callCustomizationScriptFunction('model.ext.isOutputBoxConnection',models[j],boxHandle)
            end
            if connectionIndex~=-1 then
                return models[j],connectionIndex
            end
        end
    end
    return -1,-1
end

function simBWF.disconnectInputOrOutputBox(boxHandle)
    local modelTags={simBWF.modelTags.IOHUB,simBWF.modelTags.RAGNAR}
    for i=1,#modelTags,1 do
        local models=sim.getObjectsWithTag(modelTags[i],true)
        for j=1,#models,1 do
            simBWF.callCustomizationScriptFunction('model.ext.disconnectInputOrOutputBoxConnection',models[j],boxHandle)
        end
    end
end

function simBWF.getRagnarCameraConnectedItem(cameraHandle)
    local models=sim.getObjectsWithTag(simBWF.modelTags.VISIONBOX,true)
    for i=1,#models,1 do
        local connectVisionBoxHandle=simBWF.callCustomizationScriptFunction('model.ext.isRagnarCameraConnection',models[i],cameraHandle)
        if connectVisionBoxHandle~=-1 then
            return connectVisionBoxHandle
        end
    end
    return -1
end

function simBWF.disconnectRagnarCamera(cameraHandle)
    local models=sim.getObjectsWithTag(simBWF.modelTags.VISIONBOX,true)
    for j=1,#models,1 do
        simBWF.callCustomizationScriptFunction('model.ext.disconnectRagnarCameraConnection',models[j],cameraHandle)
    end
end

function simBWF.getObjectHandle_noErrorNoSuffixAdjustment(name)
    local err=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local suff=sim.getNameSuffix(nil)
    sim.setNameSuffix(-1)
    local retVal=sim.getObjectHandle(name)
    sim.setNameSuffix(suff)
    sim.setInt32Parameter(sim.intparam_error_report_mode,err)
    return retVal
end

function simBWF.getObjectHandleFromAltName(altName)
    if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>15) then
        return sim.getObjectHandle(altName..'@alt')
    end
    return sim.getObjectHandle(altName)
end

function simBWF.getObjectAltName(objectHandle)
    if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>15) then
        return sim.getObjectName(objectHandle+sim.handleflag_altname)
    end
    return sim.getObjectName(objectHandle)
end

function simBWF.setObjectAltName(objectHandle,altName)
    if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>15) then
        if #altName>=1 then
            local correctedName=''
            for i=1,#altName,1 do
                local v=altName:sub(i,i)
                if (v>='0' and v<='9') or (v>='a' and v<='z') or (v>='A' and v<='Z') or v=='_' then
                    correctedName=correctedName..v
                else
                    correctedName=correctedName..'_'
                end
            end
            return sim.setObjectName(objectHandle+sim.handleflag_altname+sim.handleflag_silenterror,correctedName)
        end
    end
    return(-1)
end

function simBWF.getValidName(name,onlyUpperCase,optionalAllowedChars)
    if onlyUpperCase then
        name=name:upper()
    end
    local retVal=''
    for i=1,#name,1 do
        local v=name:sub(i,i)
        specialChar=false
        if type(optionalAllowedChars)=='table' then
            for j=1,#optionalAllowedChars,1 do
                if v==optionalAllowedChars[j] then
                    specialChar=true
                    break
                end
            end
        end
        if (v>='0' and v<='9') or (v>='a' and v<='z') or (v>='A' and v<='Z') or v=='_' or specialChar then
            retVal=retVal..v
        else
            retVal=retVal..'_'
        end
    end
    return retVal
end

function simBWF.openFile(fileAndPath)
    if simBWF.getPlatform()=='windows' then
        sim.launchExecutable(fileAndPath,'',0)
    end
    if simBWF.getPlatform()=='macos' then
        if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>14) then
            sim.launchExecutable('@open',fileAndPath,0)
        else
            sim.launchExecutable('/usr/bin/open',fileAndPath,0)
        end
    end
    if simBWF.getPlatform()=='linux' then
        if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>14) then
            sim.launchExecutable('@xdg-open',fileAndPath,0)
        else
            sim.launchExecutable('/usr/bin/xdg-open',fileAndPath,0)
        end
    end
end

function simBWF.openUrl(url)
    if string.find(url,"http://",1)~=1 then
        url="http://"..url
    end
    if simBWF.getPlatform()=='windows' then
        sim.launchExecutable(url,'',0)
    end
    if simBWF.getPlatform()=='macos' then
        if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>14) then
            sim.launchExecutable('@open',url,0)
        else
            sim.launchExecutable('/usr/bin/open',url,0)
        end
    end
    if simBWF.getPlatform()=='linux' then
        if (simBWF.getVrepVersion()>30400 or simBWF.getVrepRevision()>14) then
            sim.launchExecutable('@xdg-open',url,0)
        else
            sim.launchExecutable('/usr/bin/xdg-open',url,0)
        end
    end
end

function simBWF.getNameAndNumber(name)
    local baseName=''
    local nbTxt=''
    for i=#name,1,-1 do
        local v=name:sub(i,i)
        if (v>='0' and v<='9') and (baseName=='') then
            nbTxt=v..nbTxt
        else
            baseName=v..baseName
        end
    end
    local nb=tonumber(nbTxt)
    return baseName,nb
end

function simBWF.format(fmt,...)
    -- on some systems, Lua will format fractional numbers with the wrong decimal char, e.g.:
    -- "0.1" as "0,1"
    local a={...}
    for i=1,#a,1 do
        if type(a[i])=='string' then
            a[i]=string.gsub(a[i],",","@@##@@")
        end
    end
    local str=string.gsub(fmt,",","@@##@@")
    str=string.gsub(string.format(str,unpack(a)),",",".")
    return(string.gsub(str,"@@##@@",","))
end

function simBWF.appendCommonModelData(model,modelTag,createModelDataWithVersionNumber)
    model.handle=sim.getObjectAssociatedWithScript(sim.handle_self)
    model.tagName=modelTag
    model.codeVersion=-1
    local data=sim.readCustomDataBlock(model.handle,model.tagName)
    if data then
        data=sim.unpackTable(data)
        model.modelVersion=data.version
    else
        if createModelDataWithVersionNumber then
            model.modelVersion=createModelDataWithVersionNumber
            data={}
            data.version=createModelDataWithVersionNumber
            sim.writeCustomDataBlock(model.handle,model.tagName,sim.packTable(data))
        else
            model.modelVersion=-1
        end
    end
end

function simBWF.getAllJobNames()
    -- Returns the system jobs
    local currentJob=sim.getStringParameter(sim.stringparam_job)
    local ret={currentJob} -- have current job in position 1
    local cnt=sim.getInt32Parameter(sim.intparam_job_count)
    for i=1,cnt,1 do
        local s=sim.getStringParameter(sim.stringparam_job0-1+i)
        if s~=currentJob then
            ret[#ret+1]=s
        end
    end
    return ret
end

function simBWF.isJobDataConsistent(jobData)
    -- Checks whether the system jobs are same as the jobs provided in 'jobs' map
    local jobs=jobData.jobs
    local jobNames=simBWF.getAllJobNames()
    for i=1,#jobNames,1 do
        if jobs[jobNames[i]]==nil then
            return false
        end
    end
    local cnt=0
    for key,value in pairs(jobs) do
        cnt=cnt+1
    end
    return (cnt==#jobNames)
end

function simBWF.readObjectReferencesForSpecificJob(modelHandle,objRefJobInfo,jobIndex)
    -- Returns the object references for a given job. jobIndex 1 is the current job
    local objRefCnt=objRefJobInfo[1]-1
    local retVal={}
    for i=1,objRefCnt,1 do
        retVal[#retVal+1]=simBWF.getReferencedObjectHandle(modelHandle,(jobIndex-1)*objRefCnt+i)
    end
    return retVal
end


function simBWF.writeObjectReferencesForSpecificJob(modelHandle,objRefs,jobIndex)
    -- Writes the provided object references to a specific job slots
    local cnt=#objRefs
    for i=1,cnt,1 do
        simBWF.setReferencedObjectHandle(modelHandle,(jobIndex-1)*cnt+i,objRefs[i])
    end
end

function simBWF.swapJobsInObjectReferences(modelHandle,objRefJobInfo,newJobIndex)
    -- Swaps the object references between oldJob and newJob.
    local theMap={}
    for i=1,#objRefJobInfo-1,1 do
        theMap[objRefJobInfo[1+i]]=true
    end
    local objRefCnt=objRefJobInfo[1]-1
    for i=1,objRefCnt,1 do
        local tmp1=simBWF.getReferencedObjectHandle(modelHandle,i)
        local tmp2=simBWF.getReferencedObjectHandle(modelHandle,(newJobIndex-1)*objRefCnt+i)
        simBWF.setReferencedObjectHandle(modelHandle,(newJobIndex-1)*objRefCnt+i,tmp1)
        if theMap[i] then
            simBWF.setReferencedObjectHandle(modelHandle,i,tmp2) -- only references that are job-related!
        end
    end
end


function simBWF.handleJobConsistencyInObjectReferences(modelHandle,objRefJobInfo,jobData)
    local jobCnt=0
    for key,value in pairs(jobData.jobs) do
        jobCnt=jobCnt+1
    end
    if jobCnt>0 then
        if jobData.objRefJobInfo[1]~=objRefJobInfo[1] then
            -- The job start index for object references has shifted,
            -- probably because we have added new object references.
            local objRefCnt_old=jobData.objRefJobInfo[1]-1
            local objRefCnt_new=objRefJobInfo[1]-1
            local oldRefs={}
            for i=1,jobCnt,1 do
                for j=1,objRefCnt_old,1 do
                    oldRefs[#oldRefs+1]=simBWF.getReferencedObjectHandle(modelHandle,(i-1)*objRefCnt_old+j)
                end
            end
            sim.setReferencedHandles(modelHandle,{})
            local mm=math.min(objRefCnt_old,objRefCnt_new)
            for i=1,jobCnt,1 do
                for j=1,mm,1 do
                    simBWF.setReferencedObjectHandle(modelHandle,(i-1)*objRefCnt_new+j,oldRefs[(i-1)*objRefCnt_old+j])
                end
            end
            jobData.objRefJobInfo[1]=objRefJobInfo[1]
        end
        if sim.packTable(jobData.objRefJobInfo)~=sim.packTable(objRefJobInfo) then
            -- We have added/excluded some objRef items from the job scope
            -- So we set all excluded items in the job section same as current items:
            local objRefCnt=objRefJobInfo[1]-1
            for i=1,jobCnt-1,1 do
                for j=1,#objRefJobInfo-1,1 do
                    local h=simBWF.getReferencedObjectHandle(modelHandle,objRefJobInfo[1+j])
                    simBWF.setReferencedObjectHandle(modelHandle,i*objRefCnt+objRefJobInfo[1+j],h)
                end
            end
        end
    end
    jobData.objRefJobInfo=objRefJobInfo
end

function simBWF.deleteJobInObjectReferences(modelHandle,objRefJobInfo,jobData,oldJobName)
    -- Removes a job on an object references level
    local objRefCnt=objRefJobInfo[1]-1
    jobIndexToRemove=jobData.jobs[oldJobName].jobIndex
    local cnt=0
    for key,value in pairs(jobData.jobs) do
        if jobData.jobs[key].jobIndex>jobIndexToRemove then
            jobData.jobs[key].jobIndex=jobData.jobs[key].jobIndex-1
        end
        cnt=cnt+1
    end
    local newRefs={}
    for i=1,cnt,1 do
        if i~=jobIndexToRemove then
            for j=1,objRefCnt,1 do
                newRefs[#newRefs+1]=simBWF.getReferencedObjectHandle(modelHandle,(i-1)*objRefCnt+j)
            end
        end
    end
    sim.setReferencedHandles(modelHandle,{})
    for i=1,#newRefs,1 do
        simBWF.setReferencedObjectHandle(modelHandle,i,newRefs[i])
    end
end

function simBWF.createNewJobInObjectReferences(modelHandle,objRefJobInfo,jobData)
    local objRefCnt=objRefJobInfo[1]-1
    local cnt=0
    for key,value in pairs(jobData.jobs) do
        cnt=cnt+1
    end
    cnt=cnt+1
    local objRefs=simBWF.readObjectReferencesForSpecificJob(modelHandle,objRefJobInfo,1)
    simBWF.writeObjectReferencesForSpecificJob(modelHandle,objRefs,cnt)
    return cnt
end

function simBWF.printJobData(modelHandle,objRefJobInfo,jobs)
    local objRefCnt=objRefJobInfo[1]-1
    local jb={}
    for key,value in pairs(jobs) do
        jb[jobs[key].jobIndex]={key,value}
    end
    local t=''
    for i=1,#jb,1 do
        local key=jb[i][1]
        local value=jb[i][2]
        local jobIndex=jobs[key].jobIndex
        local refs=simBWF.readObjectReferencesForSpecificJob(modelHandle,objRefJobInfo,jobIndex)
        t=t..'  |  '..key..': ('..jobIndex..') : '
        for i=1,#refs,1 do
            t=t..refs[i]..' '
        end
        t=t..' - '
        for key2,value2 in pairs(value) do
            t=t..key2..' '
        end
    end
    --print(t)
end

function simBWF.handleJobConsistency_generic(removeJobsExceptCurrent)
    -- Make sure stored jobs are consistent with current scene:
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local data=model.readInfo()
    simBWF.handleJobConsistencyInObjectReferences(model.handle,model.objRefJobInfo,data.jobData)
    model.writeInfo(data)
    
    if (not simBWF.isJobDataConsistent(data.jobData)) or removeJobsExceptCurrent then
        -- Remove all jobs in this model:
        data.jobData.jobs={}
        -- Set-up job data to be identical for all jobs:
        local jobNames=simBWF.getAllJobNames()
        local objRefs=simBWF.readObjectReferencesForSpecificJob(model.handle,model.objRefJobInfo,1)
        sim.setReferencedHandles(model.handle,{})
        for i=1,#jobNames,1 do
            local newJob={jobIndex=i}
            data.jobData.jobs[jobNames[i]]=newJob
            model.copyModelParamsToJob(data,jobNames[i])
            simBWF.writeObjectReferencesForSpecificJob(model.handle,objRefs,i)
        end
        data.jobData.activeJobInModel=model.currentJob
        model.writeInfo(data)
    else
        -- Switch to the correct job if needed (could have been copied before switching job, but pasted after switching job)
        local jobNames=simBWF.getAllJobNames()
        if #jobNames>1 then
            if data.jobData.activeJobInModel~=model.currentJob then
                simBWF.switchFromJobToCurrent_generic(data.jobData.activeJobInModel)
            end
        else
            data.jobData.activeJobInModel=model.currentJob
            model.writeInfo(data)
        end
    end
end

function simBWF.createNewJob_generic()
    -- Job was created by the system. Reflect changes in this model:
    -- 1. Create a new job:
    local oldJobName=model.currentJob
    local newJobName=sim.getStringParameter(sim.stringparam_job)
    local data=model.readInfo()
    local objRef_jobIndex=simBWF.createNewJobInObjectReferences(model.handle,model.objRefJobInfo,data.jobData)
--    print(objRef_jobIndex)
--    print(newJobName)
    data.jobData.jobs[newJobName]={jobIndex=objRef_jobIndex}
    model.copyModelParamsToJob(data,newJobName)
    model.writeInfo(data)
    -- 2. Switch to it:
    simBWF.switchFromJobToCurrent_generic(oldJobName)
end

function simBWF.deleteJob_generic()
    -- Job was deleted by the system. Reflect changes in this model:
    -- 1. Switch to current job:
    local oldJobName=model.currentJob
    simBWF.switchFromJobToCurrent_generic(oldJobName)
    -- 2. Delete previous job:
    local data=model.readInfo()
    simBWF.deleteJobInObjectReferences(model.handle,model.objRefJobInfo,data.jobData,oldJobName)
    data.jobData.jobs[oldJobName]=nil
    model.writeInfo(data)
end

function simBWF.renameJob_generic()
    -- Job was renamed by the system. Reflect changes in this model:
    local oldJobName=model.currentJob
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local data=model.readInfo()
    data.jobData.jobs[model.currentJob]=data.jobData.jobs[oldJobName]
    data.jobData.jobs[oldJobName]=nil
    data.jobData.activeJobInModel=model.currentJob
    model.writeInfo(data)
end

function simBWF.switchJob_generic()
    -- Job was switched by the system. Reflect changes in this model:
    simBWF.switchFromJobToCurrent_generic(model.currentJob)
end

function simBWF.switchFromJobToCurrent_generic(oldJobName)
    model.currentJob=sim.getStringParameter(sim.stringparam_job)
    local data=model.readInfo()
    local oldJ=data.jobData.jobs[oldJobName]
    local currentJ=data.jobData.jobs[model.currentJob]
    -- oldJ.jobIndex should always be 1: we always switch from current to another
    simBWF.swapJobsInObjectReferences(model.handle,model.objRefJobInfo,currentJ.jobIndex)
    model.copyModelParamsToJob(data,oldJobName)
    model.copyJobToModelParams(data,model.currentJob)
    local tmp=oldJ.jobIndex
    oldJ.jobIndex=currentJ.jobIndex
    currentJ.jobIndex=tmp
    data.jobData.activeJobInModel=model.currentJob
    model.writeInfo(data)
end







function simBWF.getVrepVersion()
    if simBWF._vrepVersion==nil then
        simBWF._vrepVersion=sim.getInt32Parameter(sim.intparam_program_version)
    end
    return simBWF._vrepVersion
end

function simBWF.getVrepRevision()
    if simBWF._vrepRevision==nil then
        simBWF._vrepRevision=sim.getInt32Parameter(sim.intparam_program_revision)
    end
    return simBWF._vrepRevision
end

function simBWF.getPlatform()
    if simBWF._platform==nil then
        local platf=sim.getInt32Parameter(sim.intparam_platform)
        if platf==0 then
            simBWF._platform='windows'
        end
        if platf==1 then
            simBWF._platform='macos'
        end
        if platf==2 then
            simBWF._platform='linux'
        end
    end
    return simBWF._platform
end

function simBWF.getApplication()
    if simBWF._application==nil then
        local app=sim.getInt32Parameter(sim.intparam_compilation_version)
        if app==0 then
            simBWF._application='edu'
        end
        if app==1 then
            simBWF._application='pro'
        end
        if app==2 then
            simBWF._application='player'
        end
        if app==3 then
            simBWF._application='br'
        end
    end
    return simBWF._application
end

function simBWF.instanciatePart(partHandle,itemPosition,itemOrientation,itemMass,itemScaling,allowChildItemsIfApplicable)
    local l=sim.getObjectsWithTag(simBWF.modelTags.INSTANCIATEDPARTHOLDER,true)
    if l and #l==1 then
        return simBWF.callCustomizationScriptFunction('model.ext.instanciatePart',l[1],partHandle,itemPosition,itemOrientation,itemMass,itemScaling,allowChildItemsIfApplicable)
    end
end

function simBWF.markModelAsCopy(modelHandle,isACopy)
    if isACopy then
        sim.writeCustomDataBlock(modelHandle,simBWF.BR_MODEL_COPY_MARK,sim.packTable({}))
    else
        sim.writeCustomDataBlock(modelHandle,simBWF.BR_MODEL_COPY_MARK,nil)
    end
end

function simBWF.isModelACopy_ifYesRemoveCopyTag(modelHandle)
    local data=sim.readCustomDataBlock(modelHandle,simBWF.BR_MODEL_COPY_MARK)
    if data then
        sim.writeCustomDataBlock(modelHandle,simBWF.BR_MODEL_COPY_MARK,nil)
    end
    return data~=nil
end

function simBWF.getRatatosk()
    return simBWF.RATATOSK
end

function simBWF.setRatatosk(state)
    simBWF.RATATOSK=state
end

simBWF.overallCodeVersion='2018/10/05'

-- Model tags (do not modify):
simBWF.modelTags={}
simBWF.modelTags.THERMOFORMER='XYZ_THERMOFORMER_INFO'
simBWF.modelTags.INPUTBOX='XYZ_INPUTBOX_INFO'
simBWF.modelTags.OUTPUTBOX='XYZ_OUTPUTBOX_INFO'
simBWF.modelTags.IOHUB='XYZ_IOHUB_INFO'
simBWF.modelTags.VISIONBOX='XYZ_VISIONBOX_INFO'
simBWF.modelTags.TESTMODEL='XYZ_TESTMODEL_INFO'
simBWF.modelTags.LOCATIONFRAME='XYZ_LOCATIONFRAME_INFO'
simBWF.modelTags.TRACKINGWINDOW='XYZ_TRACKINGWINDOW_INFO'
simBWF.modelTags.RAGNAR='RAGNAR_CONF'
simBWF.modelTags.CONVEYOR='CONVEYOR_CONF'
simBWF.modelTags.RAGNARFRAME='XYZ_RAGNARFRAME_INFO'
simBWF.modelTags.PARTFEEDER='XYZ_FEEDER_INFO'
simBWF.modelTags.MULTIFEEDER='XYZ_MULTIFEEDERTRIGGER_INFO'
simBWF.modelTags.PARTTAGGER='XYZ_PARTTAGGER_INFO'
simBWF.modelTags.PART='XYZ_FEEDERPART_INFO' -- "FEEDERPART" is actually a bad name, since such a part does not always have to be instanciated by the feeder
simBWF.modelTags.PALLET='XYZ_PALLET_INFO'
simBWF.modelTags.PACKML='XYZ_PACKML_INFO'
simBWF.modelTags.BLUEREALITYAPP='XYZ_BLUEREALITYAPP_INFO'
simBWF.modelTags.PARTREPOSITORY='XYZ_PARTREPO_INFO'
simBWF.modelTags.PALLETREPOSITORY='XYZ_PALLETREPO_INFO'
simBWF.modelTags.INSTANCIATEDPARTHOLDER='XYZ_INSTANCIATEDPARTS_INFO'
simBWF.modelTags.PARTSINK='XYZ_PARTSINK_INFO'
simBWF.modelTags.LIFT='XYZ_LIFT_INFO'
simBWF.modelTags.RAGNARGRIPPER='XYZ_RAGNARGRIPPER_INFO'
simBWF.modelTags.RAGNARGRIPPERPLATFORM='XYZ_RAGNARGRIPPERPLATFORM_INFO'
simBWF.modelTags.VISIONWINDOW='XYZ_RAGNARVISION_INFO'
simBWF.modelTags.RAGNARCAMERA='XYZ_RAGNARCAMERA_INFO'
simBWF.modelTags.RAGNARSENSOR='XYZ_RAGNARSENSOR_INFO'
simBWF.modelTags.RAGNARDETECTOR='XYZ_RAGNARDETECTOR_INFO'
simBWF.modelTags.BINARYSENSOR='XYZ_BINARYSENSOR_INFO'
simBWF.modelTags.RAGNARGRIPPERPLATFORMIKPT='XYZ_RAGNARGRIPPERPLATFORM_IKPT_INFO'
simBWF.modelTags.CALIBRATIONBALL1='XYZ_CALIBRATIONBALL1_INFO'
simBWF.modelTags.GENERIC_PART='XYZ_GENERICPART_INFO'
simBWF.modelTags.BOX_PART='XYZ_BOX_INFO'
simBWF.modelTags.CYLINDER_PART='XYZ_CYLINDER_INFO'
simBWF.modelTags.SPHERE_PART='XYZ_SPHERE_INFO'
simBWF.modelTags.TRAY_PART='XYZ_TRAY_INFO'
simBWF.modelTags.PACKINGBOX_PART='XYZ_PACKINGBOX_INFO'
simBWF.modelTags.PILLOWBAG_PART='XYZ_PILLOWBAG_INFO'
simBWF.modelTags.SHIPPINGBOX_PART='XYZ_SHIPPINGBOX_INFO'
simBWF.modelTags.LABEL_PART='XYZ_PARTLABEL_INFO'
simBWF.modelTags.GEOMETRY_PART='PART_GEOMETRY_INFO'
simBWF.modelTags.PART_GENERATOR='PART_GENERATOR_INFO'

simBWF.modelTags.OLDLOCATION='XYZ_LOCATION_INFO'
simBWF.modelTags.OLDPARTREPO='XYZ_PARTREPOSITORY_INFO'
simBWF.modelTags.OLDOVERRIDE='XYZ_OVERRIDE_INFO'
simBWF.modelTags.OLDSTATICPICKWINDOW='XYZ_STATICPICKWINDOW_INFO'
simBWF.modelTags.OLDSTATICPLACEWINDOW='XYZ_STATICPLACEWINDOW_INFO'

simBWF.NONE_TEXT='<NONE>' -- do not modify, is serialized
simBWF.BR_MODEL_COPY_MARK='BR_MODEL_IS_COPY'
simBWF.MSG_TXT=1
simBWF.MSG_WARN=2
simBWF.MSG_ERROR=4

simBWF.RATATOSK=false -- global variable to ensure only one instance of ratatosk is opened



-- Part referenced object slots (do not modify):
simBWF.PART_PALLET_REF=1
simBWF.PART_DESTINATIONFIRST_REF=2
simBWF.PART_DESTINATIONLAST_REF=49

-- Part palletizer referenced object slots (do not modify):
simBWF.PALLETIZER_CONVEYOR_REF=1

-- Static pick window referenced object slots (do not modify):
simBWF.STATICPICKWINDOW_SENSOR_REF=1

-- Static place window referenced object slots (do not modify):
simBWF.STATICPLACEWINDOW_SENSOR_REF=1

-- Old tracking window referenced object slots (do not modify):
simBWF.OLDTRACKINGWINDOW_CONVEYOR_REF=1
simBWF.OLDTRACKINGWINDOW_INPUT_REF=2

-- Old Ragnar referenced object slots (do not modify):
simBWF.OLDRAGNAR_PARTTRACKING1_REF=1
simBWF.OLDRAGNAR_PARTTRACKING2_REF=2
simBWF.OLDRAGNAR_STATICWINDOW1_REF=3
simBWF.OLDRAGNAR_TARGETTRACKING1_REF=11
simBWF.OLDRAGNAR_TARGETTRACKING2_REF=12
simBWF.OLDRAGNAR_STATICTARGETWINDOW1_REF=13
simBWF.OLDRAGNAR_DROPLOCATION1_REF=21
simBWF.OLDRAGNAR_DROPLOCATION2_REF=22
simBWF.OLDRAGNAR_DROPLOCATION3_REF=23
simBWF.OLDRAGNAR_DROPLOCATION4_REF=24


-- Part teleporter referenced object slots (do not modify):
simBWF.TELEPORTER_DESTINATION_REF=1


-- Old, V0 stuff:
----------------------------------------------------------
function simBWF.createOpenBox(size,baseThickness,wallThickness,density,inertiaCorrectionFact,static,respondable,color) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.createOpenBox(size,baseThickness,wallThickness,density,inertiaCorrectionFact,static,respondable,color) end
function simBWF.generatePalletPoints(objectData) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.generatePalletPoints(objectData) end
function simBWF.getCircularPalletPoints(radius,count,angleOffset,center,layers,layerStep,optionalGlobalOffset) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.getCircularPalletPoints(radius,count,angleOffset,center,layers,layerStep,optionalGlobalOffset) end
function simBWF.readPartInfoV0(handle) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.readPartInfoV0(handle) end
function simBWF.palletPointsToString(palletPoints) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.palletPointsToString(palletPoints) end
function simBWF.stringToPalletPoints(txt) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.stringToPalletPoints(txt) end
function simBWF.arePalletPointsSame_posOrientAndLayer(pall1,pall2) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.arePalletPointsSame_posOrientAndLayer(pall1,pall2) end
function simBWF.readPalletFromFile(file) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.readPalletFromFile(file) end
function simBWF.getAllPossiblePartDestinationsV0() require("/bwf/modelScripts/simExtBwf-v0") return simBWF.getAllPossiblePartDestinationsV0() end
function simBWF.checkIfCodeAndModelMatch(modelHandle,codeVersion,modelVersion) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.checkIfCodeAndModelMatch(modelHandle,codeVersion,modelVersion) end
function simBWF.getAllPossibleTriggerableFeeders(except) require("/bwf/modelScripts/simExtBwf-v0") return simBWF.getAllPossibleTriggerableFeeders(except) end
----------------------------------------------------------


return simBWF
