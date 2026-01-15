model.timeDlg={}

function model.timeDlg.onClose()
    if model.timeDlg.ui then
        model.timeDlg.timeDlg_wasClosed=true
        model.timeDlg.closeDlg()
    end
end

function model.timeDlg.closeDlg()
    if model.timeDlg.ui then
        local x,y=simUI.getPosition(model.timeDlg.ui)
        model.timeDlg.timeUi_previousDlgPos={x,y}
        simUI.destroy(model.timeDlg.ui)
        model.timeDlg.ui=nil
    end
end

function model.timeDlg.createDlg()
    if not model.timeDlg.ui then
        if not model.timeDlg.timeUi_previousDlgPos then
            model.timeDlg.timeUi_previousDlgPos='bottomLeft'
        end
        if model.simplifiedTimeDisplay or model.online then
            local xml =[[
                    <label text="Time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="1" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
            ]]
            local title='Simulation Time'
            if model.online then
                title='Time'
            end
            model.timeDlg.ui=simBWF.createCustomUi(xml,title,model.timeDlg.timeUi_previousDlgPos,true,'model.timeDlg.onClose',false,false,false,'layout="form"')
        else
            local xml =[[
                    <label text="Simulation time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="1" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label text="Real-time " style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                    <label id="2" text="" style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
            ]]
            model.timeDlg.ui=simBWF.createCustomUi(xml,'Simulation Time',model.timeDlg.timeUi_previousDlgPos,true,'model.timeDlg.onClose',false,false,false,'layout="form"')
        end
    end
end

