model.dlg={}

function model.dlg.refresh()
    if model.dlg.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)
        local connectionHandle=simBWF.getRagnarCameraConnectedItem(model.handle)
        local tmp=simBWF.getObjectAltNameOrNone(connectionHandle)
        simUI.setLabelText(model.dlg.ui,1366,tmp)

        local typeComboItems={
            {"none",0},
            {"color",1},
            {"depth",2},
--            {"processed",3},
        }
        simBWF.populateCombobox(model.dlg.ui,1001,typeComboItems,{},typeComboItems[config['imgToDisplay'][1]+1][1],false,nil)
        local sizeComboItems={
            {"small",0},
            {"medium",1},
            {"large",2}
        }
        simBWF.populateCombobox(model.dlg.ui,1002,sizeComboItems,{},sizeComboItems[config['imgSizeToDisplay'][1]+1][1],false,nil)
        local updateFrequComboItems={
            {"every 50 ms",0},
            {"every 200 ms",1},
            {"every 1000 ms",2}
        }
        simBWF.populateCombobox(model.dlg.ui,1003,updateFrequComboItems,{},updateFrequComboItems[config['imgUpdateFrequ'][1]+1][1],false,nil)
        simBWF.populateCombobox(model.dlg.ui,1101,typeComboItems,{},typeComboItems[config['imgToDisplay'][2]+1][1],false,nil)
        simBWF.populateCombobox(model.dlg.ui,1102,sizeComboItems,{},sizeComboItems[config['imgSizeToDisplay'][2]+1][1],false,nil)
        simBWF.populateCombobox(model.dlg.ui,1103,updateFrequComboItems,{},updateFrequComboItems[config['imgUpdateFrequ'][2]+1][1],false,nil)

        model.dlg.updateDeviceIdCombobox()

        simUI.setEditValue(model.dlg.ui,1365,simBWF.getObjectAltName(model.handle),true)
        simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f , %.0f , %.0f",config.size[1]*1000,config.size[2]*1000,config.size[3]*1000),true)
        simUI.setEditValue(model.dlg.ui,4,simBWF.format("%i",config['resolution'][1]),true)
        simUI.setEditValue(model.dlg.ui,5,simBWF.format("%i",config['resolution'][2]),true)
        simUI.setEditValue(model.dlg.ui,6,simBWF.format("%.0f",config['clippPlanes'][1]/0.001),true)
        simUI.setEditValue(model.dlg.ui,7,simBWF.format("%.0f",config['clippPlanes'][2]/0.001),true)
        simUI.setEditValue(model.dlg.ui,8,simBWF.format("%.1f",config['fov']*180/math.pi),true)
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
        model.dlg.updateEnabledDisabledItems()
    end
end

function model.dlg.updateEnabledDisabledItems()
    if model.dlg.ui then
        local c=model.readInfo()
        local simStopped=sim.getSimulationState()==sim.simulation_stopped
        simUI.setEnabled(model.dlg.ui,1365,simStopped,true)
        simUI.setEnabled(model.dlg.ui,1,simStopped,true)
        simUI.setEnabled(model.dlg.ui,4,simStopped,true)
        simUI.setEnabled(model.dlg.ui,5,simStopped,true)
        simUI.setEnabled(model.dlg.ui,6,simStopped,true)
        simUI.setEnabled(model.dlg.ui,7,simStopped,true)
        simUI.setEnabled(model.dlg.ui,8,simStopped,true)
        simUI.setEnabled(model.dlg.ui,4899,simStopped,true)
    end
end

function model.dlg.sizeChange_callback(ui,id,newValue)
    local c=model.readInfo()
    local i=1
    local t=c.size
    for token in (newValue..","):gmatch("([^,]*),") do
        t[i]=tonumber(token)
        if t[i]==nil then t[i]=0 end
        t[i]=t[i]*0.001
        if i==1 then
            if t[i]<0.005 then t[i]=0.005 end
            if t[i]>0.15 then t[i]=0.15 end
        end
        if i==2 then
            if t[i]<0.01 then t[i]=0.01 end
            if t[i]>0.3 then t[i]=0.3 end
        end
        if i==3 then
            if t[i]<0.005 then t[i]=0.005 end
            if t[i]>0.15 then t[i]=0.15 end
        end
        i=i+1
    end
    c.size=t
    model.writeInfo(c)
    model.setCameraBodySizes()
    simBWF.markUndoPoint()
    model.dlg.refresh()
end

function model.dlg.fovChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<45 then v=45 end
        if v>110 then v=110 end
        v=v*math.pi/180
        if v~=c.fov then
            c.fov=v
            model.writeInfo(c)
            model.setResolutionAndFov(c.resolution,c.fov)
            simBWF.markUndoPoint()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.resXChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<128 then v=128 end
        if v>1920 then v=1920 end
        if v~=c.resolution[1] then
            c.resolution[1]=v
--            c.detectionPolygonSimulation={}
            model.writeInfo(c)
            model.setResolutionAndFov(c.resolution,c.fov)
            simBWF.markUndoPoint()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.resYChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=math.floor(v)
        if v<128 then v=128 end
        if v>1080 then v=1080 end
        if v~=c.resolution[2] then
            c.resolution[2]=v
--            c.detectionPolygonSimulation={}
            model.writeInfo(c)
            model.setResolutionAndFov(c.resolution,c.fov)
            simBWF.markUndoPoint()
            model.updatePluginRepresentation()
        end
    end
    model.dlg.refresh()
end

function model.dlg.nearClippingPlaneChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.02 then v=0.02 end
        if v>0.5 then v=0.5 end
        if v>(c['clippPlanes'][2]-0.01) then v=c['clippPlanes'][2]-0.01 end
        if v~=c['clippPlanes'][1] then
            c['clippPlanes'][1]=v
            model.writeInfo(c)
            model.setClippingPlanes(c['clippPlanes'])
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.farClippingPlaneChange_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        v=v*0.001
        if v<0.1 then v=0.1 end
        if v>2 then v=2 end
        if v<(c['clippPlanes'][1]+0.01) then v=c['clippPlanes'][1]+0.01 end
        if v~=c['clippPlanes'][2] then
            c['clippPlanes'][2]=v
            model.writeInfo(c)
            model.setClippingPlanes(c['clippPlanes'])
            simBWF.markUndoPoint()
        end
    end
    model.dlg.refresh()
end

function model.dlg.simImageChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgToDisplay'][1]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.simVisualizationSizeChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgSizeToDisplay'][1]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.simImgUpdateFrequChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgUpdateFrequ'][1]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.realImageChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgToDisplay'][2]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.realVisualizationSizeChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgSizeToDisplay'][2]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.realImgUpdateFrequChange_callback(ui,id,newIndex)
    local c=model.readInfo()
    c['imgUpdateFrequ'][2]=newIndex
    model.writeInfo(c)
    simBWF.markUndoPoint()
end

function model.dlg.nameChange(ui,id,newVal)
    if simBWF.setObjectAltName(model.handle,newVal)>0 then
        simBWF.markUndoPoint()
        model.updatePluginRepresentation()
        simUI.setTitle(ui,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion))
    end
    model.dlg.refresh()
end

function model.dlg.deviceIdComboChange_callback(uiHandle,id,newValue)
    local newDeviceId=comboDeviceIds[newValue+1][1]
    local c=model.readInfo()
    c.deviceId=newDeviceId
    model.writeInfo(c)
    simBWF.markUndoPoint()
--    model.dlg.updateDeviceIdCombobox()
    model.updatePluginRepresentation()
    model.dlg.refresh()
end

function model.dlg.updateDeviceIdCombobox()
    local c=model.readInfo()
    local resp,data
    if simBWF.isInTestMode() then
        resp='ok'
        data={}
        data.deviceIds={'RAGNAR-95426587447','RAGNAR-35426884525','CONVEYOR-35426884525-0','CONVEYOR-00:1b:63:84:45:e6-1','SENSOR-00:fa:08:46:8b:11-1','VISION-00:3b:99:34:7d:1f-0'}
    else
        resp,data=simBWF.query('get_deviceIds')
        if resp~='ok' then
            data.deviceIds={}
        end
    end

    local ids={}
    for i=1,#data.deviceIds,1 do
        if string.find(data.deviceIds[i],"VISION-")==1 then
            local idToShow = string.gsub(data.deviceIds[i], 'VISION%-', '')
            ids[#ids+1]=idToShow
        end
    end

    local selected=c.deviceId
    local isKnown=false
    local items={}
    for i=1,#ids,1 do
        if ids[i]==selected then
            isKnown=true
        end
        items[#items+1]={ids[i],i}
    end
    if not isKnown then
        table.insert(items,1,{selected,#items+1})
    end
    if selected~=simBWF.NONE_TEXT then
        table.insert(items,1,{simBWF.NONE_TEXT,#items+1})
    end
    comboDeviceIds=simBWF.populateCombobox(model.dlg.ui,4899,items,{},selected,false,{})
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml =[[
        <tabs id="77">
            <tab title="General">
            <group layout="form" flat="false">

                <label text="Name"/>
                <edit on-editing-finished="model.dlg.nameChange" id="1365"/>

                <label text="Connected to"/>
                <label text="x" id="1366"/>

            </group>
            </tab>

            <tab title="Simulation">
            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Camera image to show"/>
                <combobox id="1001" on-change="model.dlg.simImageChange_callback"></combobox>

                <label text="Visualization size"/>
                <combobox id="1002" on-change="model.dlg.simVisualizationSizeChange_callback"></combobox>

                <label text="Visualization update freq."/>
                <combobox id="1003" on-change="model.dlg.simImgUpdateFrequChange_callback"></combobox>
            </group>
            <group layout="form" flat="false">
                <label text="Simulated camera specific" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Horizontal field of view (deg)"/>
                <edit on-editing-finished="model.dlg.fovChange_callback" id="8"/>

                <label text="Resolution X"/>
                <edit on-editing-finished="model.dlg.resXChange_callback" id="4"/>

                <label text="Resolution Y"/>
                <edit on-editing-finished="model.dlg.resYChange_callback" id="5"/>
            </group>
            </tab>
             <tab title="Online">

            <group layout="form" flat="false">
                <label text="Visualization" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Camera image to show"/>
                <combobox id="1101" on-change="model.dlg.realImageChange_callback"></combobox>

                <label text="Visualization size"/>
                <combobox id="1102" on-change="model.dlg.realVisualizationSizeChange_callback"></combobox>

                <label text="Visualization update freq."/>
                <combobox id="1103" on-change="model.dlg.realImgUpdateFrequChange_callback"></combobox>
            </group>

            <group layout="form" flat="false">
                <label text="Real camera specific" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Device ID"/>
                <combobox id="4899" on-change="model.dlg.deviceIdComboChange_callback"> </combobox>
            </group>
            </tab>
            <tab title="More">
            <group layout="form" flat="false">

                <label text="Camera body size (X, Y, Z, in mm)"/>
                <edit on-editing-finished="model.dlg.sizeChange_callback" id="1"/>

                <label text="Camera Near clipping plane (mm)"/>
                <edit on-editing-finished="model.dlg.nearClippingPlaneChange_callback" id="6"/>

                <label text="Camera far clipping plane (mm)"/>
                <edit on-editing-finished="model.dlg.farClippingPlaneChange_callback" id="7"/>
            </group>
            </tab>

       </tabs>
        ]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,model.modelVersion,model.codeVersion),model.dlg.previousDlgPos)

        model.dlg.refresh()
        simUI.setCurrentTab(model.dlg.ui,77,model.dlg.mainTabIndex,true)
    end
end

function model.dlg.showDlg()
    if not model.dlg.ui then
        model.dlg.createDlg()
    end
end

function model.dlg.removeDlg()
    if model.dlg.ui then
        local x,y=simUI.getPosition(model.dlg.ui)
        model.dlg.previousDlgPos={x,y}
        model.dlg.mainTabIndex=simUI.getCurrentTab(model.dlg.ui,77)
        simUI.destroy(model.dlg.ui)
        model.dlg.ui=nil
    end
end

function model.dlg.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model.handle then
        model.dlg.showDlg()
    else
        model.dlg.removeDlg()
    end
end

function model.dlg.init()
    model.dlg.mainTabIndex=0
    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function model.dlg.cleanup()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
end
