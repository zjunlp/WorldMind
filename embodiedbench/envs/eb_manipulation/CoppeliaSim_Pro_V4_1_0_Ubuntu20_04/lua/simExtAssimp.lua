local simAssimp={}

--@fun importDlg Offers import parameters via dialog, before calling simAssimp.import
--@arg string filename The filename (including extension) of the CAD data
--@ret table handles The handles of the imported shapes
function simAssimp.importShapesDlg(filenames)
    configUiData={}
    function configUiData.onImport(ui,id,newVal)
        simUI.destroy(configUiData.dlg)
        local xml = [[
        <ui title="Importing..." closeable="true" resizable="false" activate="false" modal="true">
        <label text="Please wait a few seconds..."/>
        </ui>
        ]]
        local waitUi=simUI.create(xml)
        local scaling=configUiData.scaling
        if configUiData.autoScaling then scaling=0 end
        local options=0
        if configUiData.dropTextures then options=options+1 end
        if configUiData.dropColors then options=options+2 end
        if configUiData.dropTransparency then options=options+4 end
        if configUiData.doNotOptimizeMeshes then options=options+8 end
        if configUiData.keepIdenticalVertices then options=options+16 end
        if configUiData.generateOneShape then options=options+32 end
        if configUiData.alignedOrientations then options=options+64 end
        if configUiData.ignoreFileformatUp then options=options+128 end
        simAssimp.importShapes(configUiData.filenames,configUiData.maxRes,scaling,configUiData.upVector,options)
        configUiData=nil
        simUI.destroy(waitUi)
        print("done.")
        sim.announceSceneContentChange()
    end


    function configUiData.onMaxResolutionChanged(ui,id,newVal)
        local n=tonumber(newVal)
        if n then
            n=math.floor(n)
            if n<8 then n=8 end
            if n>2048 then n=2048 end
            configUiData.maxRes=n
        end
        simUI.setEditValue(configUiData.dlg,1,tostring(configUiData.maxRes))
    end

    function configUiData.onScalingChanged(ui,id,newVal)
        local n=tonumber(newVal)
        if n then
            if n<0.001 then n=0.001 end
            if n>1000 then n=1000 end
            configUiData.scaling=n
        end
        simUI.setEditValue(configUiData.dlg,2,tostring(configUiData.scaling))
    end

    function configUiData.onUpVectorChanged(uiHandle,id,newIndex)
        configUiData.upVector=newIndex
    end

    function configUiData.updateUpVectorCombobox()
        local items={'auto','Z','Y'}
        simUI.setComboboxItems(configUiData.dlg,6,items,configUiData.upVector)
    end

    function configUiData.onAutoScalingChanged(ui,id,newval)
        configUiData.autoScaling=not configUiData.autoScaling
        simUI.setEnabled(ui,2,not configUiData.autoScaling)
    end

    function configUiData.onDropTexturesChanged(ui,id,newval)
        configUiData.dropTextures=not configUiData.dropTextures
    end

    function configUiData.onDropColorsChanged(ui,id,newval)
        configUiData.dropColors=not configUiData.dropColors
    end

    function configUiData.onDropTransparencyChanged(ui,id,newval)
        configUiData.dropTransparency=not configUiData.dropTransparency
    end

    function configUiData.onDoNotOptimizeMeshesChanged(ui,id,newval)
        configUiData.doNotOptimizeMeshes=not configUiData.doNotOptimizeMeshes
        configUiData.generateOneShape=false
        simUI.setCheckboxValue(configUiData.dlg,10,configUiData.generateOneShape and 2 or 0)
    end
    
    function configUiData.onKeepIdenticalVerticesChanged(ui,id,newval)
        configUiData.keepIdenticalVertices=not configUiData.keepIdenticalVertices
        configUiData.generateOneShape=false
        simUI.setCheckboxValue(configUiData.dlg,10,configUiData.generateOneShape and 2 or 0)
    end

    function configUiData.onGenerateOneShapeChanged(ui,id,newval)
        configUiData.generateOneShape=not configUiData.generateOneShape
        configUiData.doNotOptimizeMeshes=false
        configUiData.keepIdenticalVertices=false
        simUI.setCheckboxValue(configUiData.dlg,8,configUiData.doNotOptimizeMeshes and 2 or 0)
        simUI.setCheckboxValue(configUiData.dlg,9,configUiData.keepIdenticalVertices and 2 or 0)
    end
    
    function configUiData.onAlignedOrientationsChanged(ui,id,newval)
        configUiData.alignedOrientations=not configUiData.alignedOrientations
    end
    
    function configUiData.onIgnoreFileformatUpChanged(ui,id,newval)
        configUiData.ignoreFileformatUp=not configUiData.ignoreFileformatUp
    end
    
    local maxTextures=1024
    local scaling=0
    local vectorUp=simAssimp.upVector.auto
    local options=0
    
    local xml = [[
    <ui title="Import Options" closeable="false" resizable="false" activate="false" modal="true">
    <group layout="form" flat="true">
    <label text="Textures max. resolution"/>
    <edit on-editing-finished="configUiData.onMaxResolutionChanged" id="1"/>
    <label text="Auto scaling"/>
    <checkbox text="" on-change="configUiData.onAutoScalingChanged" id="7" />
    <label text="Scaling"/>
    <edit on-editing-finished="configUiData.onScalingChanged" id="2"/>
    <label text="Drop textures"/>
    <checkbox text="" on-change="configUiData.onDropTexturesChanged" id="3" />
    <label text="Drop colors"/>
    <checkbox text="" on-change="configUiData.onDropColorsChanged" id="4" />
    <label text="Drop transparency"/>
    <checkbox text="" on-change="configUiData.onDropTransparencyChanged" id="5" />
    <label text="Do not optimize meshes"/>
    <checkbox text="" on-change="configUiData.onDoNotOptimizeMeshesChanged" id="8" />
    <label text="Keep identical vertices"/>
    <checkbox text="" on-change="configUiData.onKeepIdenticalVerticesChanged" id="9" />
    <label text="Generate one shape per file"/>
    <checkbox text="" on-change="configUiData.onGenerateOneShapeChanged" id="10" />
    <label text="Shapes have aligned orientations"/>
    <checkbox text="" on-change="configUiData.onAlignedOrientationsChanged" id="11" />
    <label text="Ignore up-vector coded in fileformat"/>
    <checkbox text="" on-change="configUiData.onIgnoreFileformatUpChanged" id="12" />
    <label text="Up-vector"/>
    <combobox id="6" on-change="configUiData.onUpVectorChanged"></combobox>

    <label text=""/>
    <button text="Import" checked="false" on-click="configUiData.onImport" />

    </group>
    </ui>
    ]]
    configUiData.dlg=simUI.create(xml)
    configUiData.maxRes=512
    configUiData.autoScaling=true
    configUiData.scaling=1
    configUiData.dropTextures=false
    configUiData.dropColors=false
    configUiData.dropTransparency=false
    configUiData.doNotOptimizeMeshes=false
    configUiData.keepIdenticalVertices=false
    configUiData.generateOneShape=false
    configUiData.alignedOrientations=false
    configUiData.ignoreFileformatUp=false
    configUiData.upVector=0
    configUiData.autoScaling=true
    configUiData.filenames=filenames
    simUI.setEditValue(configUiData.dlg,1,tostring(configUiData.maxRes))
    simUI.setEditValue(configUiData.dlg,2,tostring(configUiData.scaling))
    simUI.setCheckboxValue(configUiData.dlg,7,configUiData.autoScaling and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,3,configUiData.dropTextures and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,4,configUiData.dropColors and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,5,configUiData.dropTransparency and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,8,configUiData.doNotOptimizeMeshes and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,9,configUiData.keepIdenticalVertices and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,10,configUiData.generateOneShape and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,11,configUiData.alignedOrientations and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,12,configUiData.ignoreFileformatUp and 2 or 0)
    simUI.setEnabled(configUiData.dlg,2,not configUiData.autoScaling)
    configUiData.updateUpVectorCombobox()
end

--@fun exportDlg Offers export parameters via dialog, before calling simAssimp.export
--@arg string filename The filename (including extension) of the CAD data
--@arg table shapeHandles The handles of the shapes to export
function simAssimp.exportShapesDlg(filename,shapeHandles)
    configUiData={}
    function configUiData.onExport(ui,id,newVal)
        simUI.destroy(configUiData.dlg)
        local fformat=''
        local fn=string.lower(filename)
        if string.find(fn,'.obj') then fformat='obj' end
        if string.find(fn,'.ply') then fformat='plyb' end
        if string.find(fn,'.stl') then fformat='stlb' end
        if string.find(fn,'.dae') then fformat='collada' end
        if fformat~='' then
            local xml = [[
            <ui title="Exporting..." closeable="true" resizable="false" activate="false" modal="true">
            <label text="Please wait a few seconds..."/>
            </ui>
            ]]
            local waitUi=simUI.create(xml)
            local scaling=configUiData.scaling
            if configUiData.autoScaling then scaling=0 end
            local options=0
            if configUiData.dropTextures then options=options+1 end
            if configUiData.dropColors then options=options+2 end
            if configUiData.dropNormals then options=options+4 end
            if configUiData.onlyVisible then options=options+8 end
            simAssimp.exportShapes(shapeHandles,filename,fformat,scaling,configUiData.upVector+1,options)
            configUiData=nil
            simUI.destroy(waitUi)
            print("done.")
        else
            print("simAssimp.exportShapesDlg: Unsupported file format.")
        end
        sim.removeObjectFromSelection(sim.handle_all,-1)
    end


    function configUiData.onScalingChanged(ui,id,newVal)
        local n=tonumber(newVal)
        if n then
            if n<0.001 then n=0.001 end
            if n>1000 then n=1000 end
            configUiData.scaling=n
        end
        simUI.setEditValue(configUiData.dlg,2,tostring(configUiData.scaling))
    end

    function configUiData.onUpVectorChanged(uiHandle,id,newIndex)
        configUiData.upVector=newIndex
    end

    function configUiData.updateUpVectorCombobox()
        local items={'Z','Y'}
        simUI.setComboboxItems(configUiData.dlg,6,items,configUiData.upVector)
    end

    function configUiData.onDropTexturesChanged(ui,id,newval)
        configUiData.dropTextures=not configUiData.dropTextures
    end

    function configUiData.onDropColorsChanged(ui,id,newval)
        configUiData.dropColors=not configUiData.dropColors
    end
    
    function configUiData.onDropNormalsChanged(ui,id,newval)
        configUiData.dropNormals=not configUiData.dropNormals
    end
    
    function configUiData.onOnlyVisibleChanged(ui,id,newval)
        configUiData.onlyVisible=not configUiData.onlyVisible
    end
    
    local scaling=1
    local vectorUp=0
    local options=0
    
    local xml = [[
    <ui title="Export Options" closeable="false" resizable="false" activate="false" modal="true">
    <group layout="form" flat="true">
    <label text="Scaling"/>
    <edit on-editing-finished="configUiData.onScalingChanged" id="2"/>
    <label text="Drop textures"/>
    <checkbox text="" on-change="configUiData.onDropTexturesChanged" id="3" />
    <label text="Drop colors"/>
    <checkbox text="" on-change="configUiData.onDropColorsChanged" id="4" />
    <label text="Drop normals"/>
    <checkbox text="" on-change="configUiData.onDropNormalsChanged" id="5" />
    <label text="Export only what is visible"/>
    <checkbox text="" on-change="configUiData.onOnlyVisibleChanged" id="7" />
    <label text="Up-vector"/>
    <combobox id="6" on-change="configUiData.onUpVectorChanged"></combobox>

    <label text=""/>
    <button text="Export" checked="false" on-click="configUiData.onExport" />

    </group>
    </ui>
    ]]
    configUiData.dlg=simUI.create(xml)
    configUiData.scaling=1
    configUiData.dropTextures=false
    configUiData.dropColors=false
    configUiData.dropNormals=false
    configUiData.onlyVisible=true
    configUiData.upVector=0
    configUiData.filename=filename
    simUI.setEditValue(configUiData.dlg,2,tostring(configUiData.scaling))
    simUI.setCheckboxValue(configUiData.dlg,3,configUiData.dropTextures and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,4,configUiData.dropColors and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,5,configUiData.dropNormals and 2 or 0)
    simUI.setCheckboxValue(configUiData.dlg,7,configUiData.onlyVisible and 2 or 0)
    configUiData.updateUpVectorCombobox()
end

function simAssimp.init()
    -- can only be executed once sim.* functions were initialized
    sim.registerScriptFunction('simAssimp.importShapesDlg@Assimp','table shapeHandles=simAssimp.importShapesDlg(string filenames)')
    sim.registerScriptFunction('simAssimp.exportShapesDlg@Assimp','simAssimp.exportShapesDlg(string filename,table shapeHandles)')
    simAssimp.init=nil
end

if not __initFunctions then
    __initFunctions={}
end
__initFunctions[#__initFunctions+1]=simAssimp.init

return simAssimp
