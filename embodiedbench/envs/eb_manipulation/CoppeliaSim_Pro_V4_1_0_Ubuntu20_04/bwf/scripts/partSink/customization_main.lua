function model.setObjectSize(h,x,y,z)
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_x)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_x)
    local sx=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_y)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_y)
    local sy=mmax-mmin
    local r,mmin=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_min_z)
    local r,mmax=sim.getObjectFloatParameter(h,sim.objfloatparam_objbbox_max_z)
    local sz=mmax-mmin
    sim.scaleObject(h,x/sx,y/sy,z/sz)
end

function model.setSizes()
    local c=model.readInfo()
    local w=c['width']
    local l=c['length']
    local h=c['height']
    model.setObjectSize(model.handles.frame,w,l,h)
    model.setObjectSize(model.handles.cross,w-0.002,l-0.002,h-0.002)
    sim.setObjectPosition(model.handles.frame,model.handle,{0,0,h*0.5+0.001})
    sim.setObjectPosition(model.handles.cross,model.handles.frame,{0,0,0})
end

function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()

    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_afterSimulation()
    sim.setObjectInt32Parameter(model.handles.frame,sim.objintparam_visibility_layer,1)
    sim.setObjectInt32Parameter(model.handles.cross,sim.objintparam_visibility_layer,1)
    
    sim.setModelProperty(model.handle,0)
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_beforeSimulation()
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
    model.dlg.removeDlg()
    local c=model.readInfo()
    local hide=simBWF.modifyAuxVisualizationItems(sim.boolAnd32(c['bitCoded'],1)~=0)
    if hide then
        sim.setObjectInt32Parameter(model.handles.frame,sim.objintparam_visibility_layer,256)
        sim.setObjectInt32Parameter(model.handles.cross,sim.objintparam_visibility_layer,256)
    end
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
end

function sysCall_afterInstanceSwitch()
    model.updatePluginRepresentation()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
    model.removeFromPluginRepresentation()
    model.dlg.cleanup()
end

