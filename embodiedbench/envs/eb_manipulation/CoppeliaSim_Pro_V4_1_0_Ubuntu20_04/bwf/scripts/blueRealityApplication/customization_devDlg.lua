model.dev={}

function model.dev.refreshDlg()
    if model.dev.ui then
        local config=model.readInfo()
        local sel=simBWF.getSelectedEditWidget(model.dev.ui)
        simUI.setCheckboxValue(model.dev.ui,8,simBWF.getCheckboxValFromBool(sim.getIntegerSignal('__brTesting__')==1),true)
 --       simUI.setCheckboxValue(model.dev.ui,5,simBWF.getCheckboxValFromBool(sim.getBoolParameter(sim.boolparam_online_mode)),true)
        simBWF.setSelectedEditWidget(model.dev.ui,sel)
    end
end

function model.dev.testingWithoutPlugin_callback(ui,id,val)
    if val~=0 then
        sim.setIntegerSignal('__brTesting__',1)
    else
        sim.clearIntegerSignal('__brTesting__')
    end
end

--[[
function model.dev.connectWhenRunning_callback(ui,id,val)
    local s=sim.getBoolParameter(sim.boolparam_online_mode)
    sim.setBoolParameter(sim.boolparam_online_mode,not s)
end
--]]

function model.dev.createDlg()
    if not model.dev.ui then
        local xml =[[
                <group layout="form" flat="true">
                
                <label text="Testing without plugin"/>
                <checkbox text="" on-change="model.dev.testingWithoutPlugin_callback" id="8" />

                </group>
        ]]

--[[        
                        <label text="Connect when running"/>
                <checkbox text="" on-change="model.dev.connectWhenRunning_callback" id="5" />
--]]
        
        model.dev.ui=simBWF.createCustomUi(xml,'BlueReality Settings',nil,false,nil,false,false,false)
        model.dev.refreshDlg()
    end
end

function model.dev.showDlg()
    if not model.dev.ui then
        model.dev.createDlg()
    end
end

function model.dev.removeDlg()
    if model.dev.ui then
        simUI.destroy(model.dev.ui)
        model.dev.ui=nil
    end
end

function model.dev.showOrHideDlgIfNeeded()
    if sim.getInt32Parameter(sim_intparam_compilation_version)==1 then
        -- Do not show this helper dlg in BlueReality, only in V-REP PRO
        local s=sim.getObjectSelection()
        if s and #s>=1 and s[#s]==model.handle then
            model.dev.showDlg()
        else
            model.dev.removeDlg()
        end
    end
end

