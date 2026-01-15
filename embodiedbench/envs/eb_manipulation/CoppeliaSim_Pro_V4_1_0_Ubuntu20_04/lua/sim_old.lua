function sim.include(relativePathAndFile,cmd)
    -- Relative to the CoppeliaSimLib path
    if not __notFirst__ then
        local appPath=sim.getStringParameter(sim.stringparam_application_path)
        if sim.getInt32Parameter(sim.intparam_platform)==1 then
            appPath=appPath.."/../../.."
        end
        sim.includeAbs(appPath..relativePathAndFile,cmd)
    else
        if __scriptCodeToRun__ then
            __scriptCodeToRun__()
        end
    end
end

function sim.includeRel(relativePathAndFile,cmd)
    -- Relative to the current scene path
    if not __notFirst__ then
        local scenePath=sim.getStringParameter(sim.stringparam_scene_path)
        sim.includeAbs(scenePath..relativePathAndFile,cmd)
    else
        if __scriptCodeToRun__ then
            __scriptCodeToRun__()
        end
    end
end

function sim.includeAbs(absPathAndFile,cmd)
    -- Absolute path
    if not __notFirst__ then
        __notFirst__=true
        __scriptCodeToRun__=assert(loadfile(absPathAndFile))
        if cmd then
            local tmp=assert(loadstring(cmd))
            if tmp then
                tmp()
            end
        end
    end
    if __scriptCodeToRun__ then
        __scriptCodeToRun__()
    end
end

function sim.canScaleObjectNonIsometrically(objHandle,scaleAxisX,scaleAxisY,scaleAxisZ)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        return true -- iso scaling in this case
    end
    local t=sim.getObjectType(objHandle)
    if t==sim.object_joint_type then
        return true
    end
    if t==sim.object_dummy_type then
        return true
    end
    if t==sim.object_camera_type then
        return true
    end
    if t==sim.object_mirror_type then
        return true
    end
    if t==sim.object_light_type then
        return true
    end
    if t==sim.object_forcesensor_type then
        return true
    end
    if t==sim.object_path_type then
        return true
    end
    if t==sim.object_pointcloud_type then
        return false
    end
    if t==sim.object_octree_type then
        return false
    end
    if t==sim.object_graph_type then
        return false
    end
    if t==sim.object_proximitysensor_type then
        local r,p=sim.getObjectInt32Parameter(objHandle,sim.proxintparam_volume_type)
        if p==sim.volume_cylinder then
            return xIsY
        end
        if p==sim.volume_disc then
            return xIsZ
        end
        if p==sim.volume_cone then
            return false
        end
        if p==sim.volume_randomizedray then
            return false
        end
        return true
    end
    if t==sim.object_mill_type then
        local r,p=sim.getObjectInt32Parameter(objHandle,sim.millintparam_volume_type)
        if p==sim.volume_cylinder then
            return xIsY
        end
        if p==sim.volume_disc then
            return xIsZ
        end
        if p==sim.volume_cone then
            return false
        end
        return true
    end
    if t==sim.object_visionsensor_type then
        return xIsY
    end
    if t==sim.object_shape_type then
        local r,pt=sim.getShapeGeomInfo(objHandle)
        if sim.boolAnd32(r,1)~=0 then
            return false -- compound
        end
        if pt==sim.pure_primitive_spheroid then
            return false
        end
        if pt==sim.pure_primitive_disc then
            return xIsY
        end
        if pt==sim.pure_primitive_cylinder then
            return xIsY
        end
        if pt==sim.pure_primitive_cone then
            return xIsY
        end
        if pt==sim.pure_primitive_heightfield then
            return xIsY
        end
        return true
    end
end

function sim.canScaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,ignoreNonScalableItems)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local yIsZ=(math.abs(1-math.abs(scaleAxisY/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        return true -- iso scaling in this case
    end
    local allDescendents=sim.getObjectsInTree(modelHandle,sim.handle_all,1)
    -- First the model base:
    local t=sim.getObjectType(modelHandle)
    if (t==sim.object_pointcloud_type) or (t==sim.object_pointcloud_type) or (t==sim.object_pointcloud_type) then
        if not ignoreNonScalableItems then
            if not sim.canScaleObjectNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) then
                return false
            end
        end
    else
        if not sim.canScaleObjectNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ) then
            return false
        end
    end
    -- Ok, we can scale the base, now check the descendents:
    local baseFrameScalingFactors={scaleAxisX,scaleAxisY,scaleAxisZ}
    for i=1,#allDescendents,1 do
        local h=allDescendents[i]
        t=sim.getObjectType(h)
        if ( (t~=sim.object_pointcloud_type) and (t~=sim.object_pointcloud_type) and (t~=sim.object_pointcloud_type) ) or (not ignoreNonScalableItems) then
            local m=sim.getObjectMatrix(h,modelHandle)
            local axesMapping={-1,-1,-1} -- -1=no mapping
            local matchingAxesCnt=0
            local objFrameScalingFactors={nil,nil,nil}
            local singleMatchingAxisIndex
            for j=1,3,1 do
                local newAxis={m[j],m[j+4],m[j+8]}
                local x={math.abs(newAxis[1]),math.abs(newAxis[2]),math.abs(newAxis[3])}
                local v=math.max(math.max(x[1],x[2]),x[3])
                if v>0.99 then
                    matchingAxesCnt=matchingAxesCnt+1
                    if x[1]>0.9 then
                        axesMapping[j]=1
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                    if x[2]>0.9 then
                        axesMapping[j]=2
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                    if x[3]>0.9 then
                        axesMapping[j]=3
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                        singleMatchingAxisIndex=j
                    end
                end
            end
            if matchingAxesCnt==0 then
                -- the child frame is not aligned at all with the model frame. And scaling is not iso-scaling
                -- Dummies, cameras, lights and force sensors do not mind:
                local t=sim.getObjectType(h)
                if (t~=sim.object_dummy_type) and (t~=sim.object_camera_type) and (t~=sim.object_light_type) and (t~=sim.object_forcesensor_type) then
                    return false
                end
            else
                if matchingAxesCnt==3 then
                    if not sim.canScaleObjectNonIsometrically(h,objFrameScalingFactors[1],objFrameScalingFactors[2],objFrameScalingFactors[3]) then
                        return false
                    end
                else
                    -- We have only one axis that matches. We can scale the object only if the two non-matching axes have the same scaling factor:
                    local otherFactors={nil,nil}
                    for j=1,3,1 do
                        if j~=axesMapping[singleMatchingAxisIndex] then
                            if otherFactors[1] then
                                otherFactors[2]=baseFrameScalingFactors[j]
                            else
                                otherFactors[1]=baseFrameScalingFactors[j]
                            end
                        end
                    end
                    if (math.abs(1-math.abs(otherFactors[1]/otherFactors[2]))<0.001) then
                        local fff={otherFactors[1],otherFactors[1],otherFactors[1]}
                        fff[singleMatchingAxisIndex]=objFrameScalingFactors[singleMatchingAxisIndex]
                        if not sim.canScaleObjectNonIsometrically(h,fff[1],fff[2],fff[3]) then
                            return false
                        end
                    else
                        return false
                    end
                end
            end
        end
    end
    return true
end

function sim.scaleModelNonIsometrically(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ)
    local xIsY=(math.abs(1-math.abs(scaleAxisX/scaleAxisY))<0.001)
    local xIsZ=(math.abs(1-math.abs(scaleAxisX/scaleAxisZ))<0.001)
    local xIsYIsZ=(xIsY and xIsZ)
    if xIsYIsZ then
        sim.scaleObjects({modelHandle},scaleAxisX,false) -- iso scaling in this case
    else
        local avgScaling=(scaleAxisX+scaleAxisY+scaleAxisZ)/3
        local allDescendents=sim.getObjectsInTree(modelHandle,sim.handle_all,1)
        -- First the model base:
        sim.scaleObject(modelHandle,scaleAxisX,scaleAxisY,scaleAxisZ,0)
        -- Now scale all the descendents:
        local baseFrameScalingFactors={scaleAxisX,scaleAxisY,scaleAxisZ}
        for i=1,#allDescendents,1 do
            local h=allDescendents[i]
            -- First scale the object itself:
            local m=sim.getObjectMatrix(h,modelHandle)
            local axesMapping={-1,-1,-1} -- -1=no mapping
            local matchingAxesCnt=0
            local objFrameScalingFactors={nil,nil,nil}
            for j=1,3,1 do
                local newAxis={m[j],m[j+4],m[j+8]}
                local x={math.abs(newAxis[1]),math.abs(newAxis[2]),math.abs(newAxis[3])}
                local v=math.max(math.max(x[1],x[2]),x[3])
                if v>0.99 then
                    matchingAxesCnt=matchingAxesCnt+1
                    if x[1]>0.9 then
                        axesMapping[j]=1
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                    if x[2]>0.9 then
                        axesMapping[j]=2
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                    if x[3]>0.9 then
                        axesMapping[j]=3
                        objFrameScalingFactors[j]=baseFrameScalingFactors[axesMapping[j]]
                    end
                end
            end
            if matchingAxesCnt==0 then
                -- the child frame is not aligned at all with the model frame.
                sim.scaleObject(h,avgScaling,avgScaling,avgScaling,0)
            end

            if matchingAxesCnt==3 then
                -- the child frame is orthogonally aligned with the model frame
                sim.scaleObject(h,objFrameScalingFactors[1],objFrameScalingFactors[2],objFrameScalingFactors[3],0)
            else
                -- We have only one axis that is aligned with the model frame
                local objFactor,objIndex
                for j=1,3,1 do
                    if objFrameScalingFactors[j]~=nil then
                        objFactor=objFrameScalingFactors[j]
                        objIndex=j
                        break
                    end
                end
                local otherFactors={nil,nil}
                for j=1,3,1 do
                    if baseFrameScalingFactors[j]~=objFactor then
                        if otherFactors[1]==nil then
                            otherFactors[1]=baseFrameScalingFactors[j]
                        else
                            otherFactors[2]=baseFrameScalingFactors[j]
                        end
                    end
                end
                if (math.abs(1-math.abs(otherFactors[1]/otherFactors[2]))<0.001) then
                    local fff={otherFactors[1],otherFactors[1],otherFactors[1]}
                    fff[objIndex]=objFactor
                    sim.scaleObject(h,fff[1],fff[2],fff[3],0)
                else
                    local of=(otherFactors[1]+otherFactors[2])/2
                    local fff={of,of,of}
                    fff[objIndex]=objFactor
                    sim.scaleObject(h,fff[1],fff[2],fff[3],0)
                end
            end
            -- Now scale also the position of that object:
            local parentObjH=sim.getObjectParent(h)
            local m=sim.getObjectMatrix(parentObjH,modelHandle)
            m[4]=0
            m[8]=0
            m[12]=0
            local mi={}
            for j=1,12,1 do
                mi[j]=m[j]
            end
            sim.invertMatrix(mi)
            local p=sim.getObjectPosition(h,parentObjH)
            p=sim.multiplyVector(m,p)
            p[1]=p[1]*scaleAxisX
            p[2]=p[2]*scaleAxisY
            p[3]=p[3]*scaleAxisZ
            p=sim.multiplyVector(mi,p)
            sim.setObjectPosition(h,parentObjH,p)
        end
    end
end

function sim.UI_populateCombobox(ui,id,items_array,exceptItems_map,currentItem,sort,additionalItemsToTop_array)
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
