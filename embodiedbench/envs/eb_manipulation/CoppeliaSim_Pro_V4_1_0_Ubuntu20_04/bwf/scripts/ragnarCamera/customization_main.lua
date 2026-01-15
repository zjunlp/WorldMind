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

function model.setCameraBodySizes()
    local c=model.readInfo()
    local w=c['size'][2]
    local l=c['size'][1]
    local h=c['size'][3]
    model.setObjectSize(model.handles.body,w,h,l)
end

function model.setResolutionAndFov(res,fov)
    sim.setObjectInt32Parameter(model.handles.sensor,sim.visionintparam_resolution_x,res[1])
    sim.setObjectInt32Parameter(model.handles.sensor,sim.visionintparam_resolution_y,res[2])
    -- local ratio=res[1]/res[2]
    -- if ratio>1 then
    --     fov=2*math.atan(math.tan(fov/2)*ratio)
    --     --print(180*2*math.atan(math.tan(60*math.pi/360)/ratio)/math.pi)
    -- end
    sim.setObjectFloatParameter(model.handles.sensor,sim.visionfloatparam_perspective_angle,fov)
end

function model.setClippingPlanes(clipp)
    sim.setObjectFloatParameter(model.handles.sensor,sim.visionfloatparam_near_clipping,clipp[1])
    sim.setObjectFloatParameter(model.handles.sensor,sim.visionfloatparam_far_clipping,clipp[2])
end

function sysCall_init()
    model.codeVersion=1

    model.dlg.init()

    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
    model.updatePluginRepresentation()
end

function sysCall_sensing()
    if model.simJustStarted then
        model.dlg.updateEnabledDisabledItems()
    end
    model.simJustStarted=nil
    model.dlg.showOrHideDlgIfNeeded()
    model.ext.outputPluginRuntimeMessages()
end

function sysCall_suspended()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_afterSimulation()
--    sim.setObjectInt32Parameter(model.handles.arrows,sim.objintparam_visibility_layer,1)
    model.dlg.updateEnabledDisabledItems()
end

function sysCall_beforeSimulation()
    model.simJustStarted=true
--    sim.setObjectInt32Parameter(model.handles.arrows,sim.objintparam_visibility_layer,0)
    model.ext.outputBrSetupMessages()
    model.ext.outputPluginSetupMessages()
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
