model.generalProperties={}

function model.generalProperties.close_callback()
    simUI.destroy(model.generalProperties.ui)
    model.generalProperties.ui=nil
    simBWF.markUndoPoint()
    model.updatePluginRepresentation_generalProperties()
end

function model.generalProperties.masterIpChange_callback(ui,id,newVal)
    local c=model.readInfo()
    c.masterIp=newVal
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.partDeactivationTime_callback(ui,id,newVal)
    local c=model.readInfo()
    local v=tonumber(newVal)
    if v then
        if v<1 then v=1 end
        if v>100000 then v=100000 end
        if v~=c['deactivationTime'] then
            simBWF.markUndoPoint()
            c['deactivationTime']=v
            model.writeInfo(c)
        end
    end
    model.generalProperties.refreshDlg()
end

function model.generalProperties.oee_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,16)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.warningAtRunStart_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,32)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.camerasNoReset_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,64)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.simplifiedSimulationTime_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,8)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.simulationTime_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,4)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.packMLState_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,1)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.packMLStateChangeButtons_callback(ui,id)
    local c=model.readInfo()
    c.bitCoded=sim.boolXor32(c.bitCoded,2)
    model.writeInfo(c)
    model.generalProperties.refreshDlg()
end


function model.generalProperties.openDlg(ui,id)
    local xml =[[
                <group layout="form" flat="false">
                    <label text="Resolver IP" style="* {font-weight: bold;}"/>  <label text=""/>
                    
                    <label text="Address"/>
                    <edit on-editing-finished="model.generalProperties.masterIpChange_callback" style="* {min-width: 100px;}" id="1"/>
                </group>
                
                <group layout="form" flat="false">
                    <label text="PackML" style="* {font-weight: bold;}"/>  <label text=""/>
                    
                    <label text="Behaviour"/>
                    <button text="Edit" on-click="model.packMl.createDlg" id="2" />
                    
                    <label text="Display state when running"/>
                    <checkbox text="" on-change="model.generalProperties.packMLState_callback" id="3" />

                    <label text="Display state change buttons"/>
                    <checkbox text="" on-change="model.generalProperties.packMLStateChangeButtons_callback" id="4" />
                </group>

                <group layout="form" flat="false">
                    <label text="Time" style="* {font-weight: bold;}"/>  <label text=""/>
                    
                    <label text="Display when running"/>
                    <checkbox text="" on-change="model.generalProperties.simulationTime_callback" id="5" />
                    
                    <label text="Simplified display"/>
                    <checkbox text="" on-change="model.generalProperties.simplifiedSimulationTime_callback" id="6" />
                </group>
                
                <group layout="form" flat="false">
                    <label text="OEE" style="* {font-weight: bold;}"/>  <label text=""/>
                    
                    <label text="Display when running"/>
                    <checkbox text="" on-change="model.generalProperties.oee_callback" id="7" />
                </group>
                
                <group layout="form" flat="false">
                    <label text="Various" style="* {font-weight: bold;}"/>  <label text=""/>
                    
                    <label text="Display warnings when running"/>
                    <checkbox text="" on-change="model.generalProperties.warningAtRunStart_callback" id="8" />
                    
                    <label text="Do not reset cameras"/>
                    <checkbox text="" on-change="model.generalProperties.camerasNoReset_callback" id="9" />
                    
                    <label text="Part deactivation time"/>
                    <edit on-editing-finished="model.generalProperties.partDeactivationTime_callback" style="* {min-width: 100px;}" id="10"/>
                </group>
                
                <label text="Code version: ]]..simBWF.overallCodeVersion..'"/>'
    model.generalProperties.ui=simBWF.createCustomUi(xml,"Global Properties","center",true,"model.generalProperties.close_callback",true,false,true)
    model.generalProperties.refreshDlg()
end

function model.generalProperties.refreshDlg()
    local c=model.readInfo()
    local sel=simBWF.getSelectedEditWidget(model.generalProperties.ui)
    
    simUI.setEditValue(model.generalProperties.ui,1,c.masterIp)
    simUI.setCheckboxValue(model.generalProperties.ui,3,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],1)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,4,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],2)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,5,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],4)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,6,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],8)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,7,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],16)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,8,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],32)~=0),true)
    simUI.setCheckboxValue(model.generalProperties.ui,9,simBWF.getCheckboxValFromBool(sim.boolAnd32(c['bitCoded'],64)~=0),true)
    simUI.setEditValue(model.generalProperties.ui,10,simBWF.format("%.1f",c['deactivationTime']),true)
    
    local simStopped=sim.getSimulationState()==sim.simulation_stopped
    simUI.setEnabled(model.generalProperties.ui,1,simStopped,true)
    simUI.setEnabled(model.generalProperties.ui,2,simStopped,true)
    
--    simUI.setEnabled(model.generalProperties.ui,4,sim.boolAnd32(c['bitCoded'],1)~=0,true)
    simUI.setEnabled(model.generalProperties.ui,6,sim.boolAnd32(c['bitCoded'],4)~=0,true)
    
    simBWF.setSelectedEditWidget(model.generalProperties.ui,sel)
end

