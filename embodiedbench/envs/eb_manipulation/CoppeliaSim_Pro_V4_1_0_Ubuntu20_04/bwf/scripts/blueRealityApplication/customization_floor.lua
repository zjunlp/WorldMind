model.floor={}
model.floor.handles={}
model.floor.handles.e1=sim.getObjectHandle('ResizableFloor_10_50_element')
model.floor.handles.e2=sim.getObjectHandle('ResizableFloor_10_50_visibleElement')
model.floor.handles.itemsHolder=sim.getObjectHandle('Floor_floorItems')

function model.floor.update()
    local c=model.readInfo()
    local sx=c['floorSizes'][1]/10
    local sy=c['floorSizes'][2]/10
    local sizeFact=sim.getObjectSizeFactor(model.handle)
    sim.setObjectParent(model.floor.handles.e1,-1,true)
    local child=sim.getObjectChild(model.floor.handles.itemsHolder,0)
    while child~=-1 do
        sim.removeObject(child)
        child=sim.getObjectChild(model.floor.handles.itemsHolder,0)
    end
    local xPosInit=(sx-1)*-5*sizeFact
    local yPosInit=(sy-1)*-5*sizeFact
    local f1,f2
    for x=1,sx,1 do
        for y=1,sy,1 do
            if (x==1)and(y==1) then
                sim.setObjectParent(model.floor.handles.e1,model.floor.handles.itemsHolder,true)
                f1=model.floor.handles.e1
            else
                f1=sim.copyPasteObjects({model.floor.handles.e1},0)[1]
                f2=sim.copyPasteObjects({model.floor.handles.e2},0)[1]
                sim.setObjectParent(f1,model.floor.handles.itemsHolder,true)
                sim.setObjectParent(f2,f1,true)
            end
            local p=sim.getObjectPosition(f1,sim.handle_parent)
            p[1]=xPosInit+(x-1)*10*sizeFact
            p[2]=yPosInit+(y-1)*10*sizeFact
            sim.setObjectPosition(f1,sim.handle_parent,p)
        end
    end
end

function model.floor.updateUi()
    local c=model.readInfo()
    local sizeFact=sim.getObjectSizeFactor(model.handle)
    simUI.setLabelText(model.floor.ui,1,'X-size (m): '..simBWF.format("%.2f",c['floorSizes'][1]*sizeFact),true)
    simUI.setSliderValue(model.floor.ui,2,c['floorSizes'][1]/10,true)
    simUI.setLabelText(model.floor.ui,3,'Y-size (m): '..simBWF.format("%.2f",c['floorSizes'][2]*sizeFact),true)
    simUI.setSliderValue(model.floor.ui,4,c['floorSizes'][2]/10,true)
end

function model.floor.sliderXchange(ui,id,newVal)
    local c=model.readInfo()
    c['floorSizes'][1]=newVal*10
    model.writeInfo(c)
    model.floor.updateUi()
    model.floor.update()
end

function model.floor.sliderYchange(ui,id,newVal)
    local c=model.readInfo()
    c['floorSizes'][2]=newVal*10
    model.writeInfo(c)
    model.floor.updateUi()
    model.floor.update()
end

function model.floor.showDlg()
    if not model.floor.ui then
    local xml = [[
    <group layout="form" flat="true">
        <label text="X-size (m): 1" id="1"/>
        <hslider tick-position="above" tick-interval="1" minimum="1" maximum="5" on-change="model.floor.sliderXchange" id="2"/>
        <label text="Y-size (m): 1" id="3"/>
        <hslider tick-position="above" tick-interval="1" minimum="1" maximum="5" on-change="model.floor.sliderYchange" id="4"/>
    </group>
    <label text="" style="* {margin-left: 400px;}"/>
]]
        model.floor.ui=simBWF.createCustomUi(xml,'Floor',model.floor.previousDlgPos,false,nil,false,false,false)
        model.floor.updateUi()
    end
end

function model.floor.removeDlg()
    if model.floor.ui then
        local x,y=simUI.getPosition(model.floor.ui)
        model.floor.previousDlgPos={x,y}
        simUI.destroy(model.floor.ui)
        model.floor.ui=nil
    end
end

function model.floor.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[1]==model.handle then
        model.floor.showDlg()
    else
        model.floor.removeDlg()
    end
end
