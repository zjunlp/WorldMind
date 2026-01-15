model.packMlDlg={}

function model.packMlDlg.onClose()
    if model.packMlDlg.ui then
        model.packMlDlg.dlg_wasClosed=true
        model.packMlDlg.closeDlg()
    end
end

function model.packMlDlg.closeDlg()
    if model.packMlDlg.ui then
        local x,y=simUI.getPosition(model.packMlDlg.ui)
        model.packMlDlg.packMLState_previousDlgPos={x,y}
        simUI.destroy(model.packMlDlg.ui)
        model.packMlDlg.ui=nil
    end
end

function model.packMlDlg.updateState(state)
    if model.packMlDlg.ui then
        simUI.setStyleSheet(model.packMlDlg.ui,1,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,2,"* {background-color: #ffffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,3,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,4,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,5,"* {background-color: #ffffbb}")

        simUI.setStyleSheet(model.packMlDlg.ui,6,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,7,"* {background-color: #ffffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,8,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,9,"* {background-color: #bbffbb}")

        simUI.setStyleSheet(model.packMlDlg.ui,10,"* {background-color: #ffffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,11,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,12,"* {background-color: #bbddff}")
        simUI.setStyleSheet(model.packMlDlg.ui,13,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,14,"* {background-color: #ffffbb}")

        simUI.setStyleSheet(model.packMlDlg.ui,15,"* {background-color: #bbffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,16,"* {background-color: #ffffbb}")
        simUI.setStyleSheet(model.packMlDlg.ui,17,"* {background-color: #bbffbb}")
        
        state=string.lower(state)
        local id=-1
        if state=='aborting' then id=1 end
        if state=='aborted' then id=2 end
        if state=='clearing' then id=3 end
        if state=='stopping' then id=4 end
        if state=='stopped' then id=5 end
        if state=='suspending' then id=6 end
        if state=='suspended' then id=7 end
        if state=='un-suspending' then id=8 end
        if state=='resetting' then id=9 end
        if state=='complete' then id=10 end
        if state=='completing' then id=11 end
        if state=='execute' then id=12 end
        if state=='starting' then id=13 end
        if state=='idle' then id=14 end
        if state=='holding' then id=15 end
        if state=='hold' then id=16 end
        if state=='un-holding' then id=17 end
        
        if id>=0 then
            simUI.setStyleSheet(model.packMlDlg.ui,id,"* {background-color: #ff6600}")
        end
    end
end

function model.packMlDlg.createDlg()
    if not model.packMlDlg.ui then
        local xml =[[
                <image geometry="0,0,1088,607" width="702" height="390" id="1000"/>
                
                <button text="Aborting" geometry="591,312,79,50" enabled="false" id="1" style="* {background-color: #bbffbb}"/>                
                <button text="Aborted" geometry="451,312,79,50" enabled="false" id="2" style="* {background-color: #ffffbb}"/>                
                <button text="Clearing" geometry="311,312,79,50" enabled="false" id="3" style="* {background-color: #bbffbb}"/>                
                <button text="Stopping" geometry="170,312,79,50" enabled="false" id="4" style="* {background-color: #bbffbb}"/>                
                <button text="Stopped" geometry="29,312,79,50" enabled="false" id="5" style="* {background-color: #ffffbb}"/>                
                
                <button text="Suspending" geometry="451,187,79,50" enabled="false" id="6" style="* {background-color: #bbffbb}"/>                
                <button text="Suspended" geometry="311,187,79,50" enabled="false" id="7" style="* {background-color: #ffffbb}"/>                
                <button text="Un-Suspending" geometry="170,187,79,50" enabled="false" id="8" style="* {background-color: #bbffbb}"/>                
                <button text="Resetting" geometry="29,187,79,50" enabled="false" id="9" style="* {background-color: #bbffbb}"/>                

                <button text="Complete" geometry="591,108,79,50" enabled="false" id="10" style="* {background-color: #ffffbb}"/>                
                <button text="Completing" geometry="451,108,79,50" enabled="false" id="11" style="* {background-color: #bbffbb}"/>                
                <button text="Execute" geometry="311,108,79,50" enabled="false" id="12" style="* {background-color: #bbddff}"/>                
                <button text="Starting" geometry="170,108,79,50" enabled="false" id="13" style="* {background-color: #bbffbb}"/>                
                <button text="Idle" geometry="29,108,79,50" enabled="false" id="14" style="* {background-color: #ffffbb}"/>                

                <button text="Holding" geometry="451,30,79,50" enabled="false" id="15" style="* {background-color: #bbffbb}"/>                
                <button text="Hold" geometry="311,30,79,50" enabled="false" id="16" style="* {background-color: #ffffbb}"/>                
                <button text="Un-Holding" geometry="170,30,79,50" enabled="false" id="17" style="* {background-color: #bbffbb}"/>                
                ]]
        if model.packMlDlg.packMLState_previousDlgPos==nil then
            model.packMlDlg.packMLState_previousDlgPos='topLeft'
        end
        model.packMlDlg.ui=simBWF.createCustomUi(xml,'Current PackML state',model.packMlDlg.packMLState_previousDlgPos,true,'model.packMlDlg.onClose',false,false,false,'layout="none"',{702,390})

        local img=sim.loadImage(0,sim.getStringParameter(sim.stringparam_application_path).."/BlueWorkforce/resources/packML-run.png")
        simUI.setImageData(model.packMlDlg.ui,1000,img,702,390)
    end
end

