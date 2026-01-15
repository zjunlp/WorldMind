model.packMlButtons={}

function model.packMlButtons.event(uiHandle,id)
    local data={}
    data.event=model.packMlButtons.eventButtons[id-1]
    simBWF.query('packml_event',data)
end

function model.packMlButtons.onClose()
    if model.packMlButtons.ui then
        model.packMlButtons.dlg_wasClosed=true
        model.packMlButtons.closeDlg()
    end
end

function model.packMlButtons.closeDlg()
    if model.packMlButtons.ui then
        local x,y=simUI.getPosition(model.packMlButtons.ui)
        model.packMlButtons.previousDlgPos={x,y}
        simUI.destroy(model.packMlButtons.ui)
        model.packMlButtons.ui=nil
    end
end

function model.packMlButtons.updateState(state,buttons)
    if model.packMlButtons.ui then
        if #buttons==#model.packMlButtons.eventButtons then
            state=string.lower(state)
            state=string.upper(string.sub(state,1,1))..string.sub(state,2)
            simUI.setButtonText(model.packMlButtons.ui,1,state)
            for i=1,#buttons,1 do
                local bt=string.lower(buttons[i])
                bt=string.upper(string.sub(bt,1,1))..string.sub(bt,2)
                simUI.setButtonText(model.packMlButtons.ui,i+1,bt)
            end
            model.packMlButtons.eventButtons=buttons
        else
            model.packMlButtons.closeDlg()
            model.packMlButtons.createDlg(state,buttons)
        end
    end
end

function model.packMlButtons.createDlg(state,buttons)
    if not model.packMlButtons.ui then
        if state==nil then state='packMl state' end
        state=string.lower(state)
        state=string.upper(string.sub(state,1,1))..string.sub(state,2)
        local xml='<button text="'..state..'" enabled="false" id="1" style="* {min-width: 170px; min-height: 30px; font-size: 12px; background-color: #bbbbbb}"/>' 
        if buttons then
            for i=1,#buttons,1 do
                local bt=string.lower(buttons[i])
                bt=string.upper(string.sub(bt,1,1))..string.sub(bt,2)
                xml=xml..'<button text="'..bt..'" id="'..tostring(i+1)..'" on-click="model.packMlButtons.event" style="* {min-width: 170px; min-height: 30px; font-size: 12px; }"/>'
            end
            model.packMlButtons.eventButtons=buttons
        else
            model.packMlButtons.eventButtons={}
        end
        model.packMlButtons.ui=simBWF.createCustomUi(xml,'PackML actions',model.packMlButtons.previousDlgPos,true,'model.packMlButtons.onClose',false,false,false)
    end
end

