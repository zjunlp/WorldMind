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
    if z then
        sim.scaleObject(h,x/sx,y/sy,z/sz)
    else
        sim.scaleObject(h,x/sx,y/sy,1) -- for the labels that are z-flat
    end
end

function model.setEngineDamping(bulletLinDamp,bulletAngDamp,odeSoftErp)
    local l=sim.getObjectsInTree(model.handle,sim.object_shape_type,0)
    for i=1,#l,1 do
        sim.setEngineFloatParameter(sim.bullet_body_lineardamping,l[i],bulletLinDamp)
        sim.setEngineFloatParameter(sim.bullet_body_angulardamping,l[i],bulletAngDamp)
        sim.setEngineFloatParameter(sim.ode_body_softerp,l[i],odeSoftErp)
    end
end

function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()

    local data=simBWF.readPartInfo(model.handle)
    data.partType=model.partType
    data.instanciated=nil -- just in case
    simBWF.writePartInfo(model.handle,data)
    
    model.setEngineDamping(0.9,0.999,0.1)
    
    model.handleJobConsistency(simBWF.isModelACopy_ifYesRemoveCopyTag(model.handle))
    model.updatePluginRepresentation()
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
end


function sysCall_afterSimulation()
    local data=simBWF.readPartInfo(model.handle)
    if not data.instanciated then
        -- Part was not finalized. We need to reactivate it after simulation:
        sim.setModelProperty(model.handle,0)
    end
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_beforeSimulation()
    model.dlg.removeDlg()
    local data=simBWF.readPartInfo(model.handle)
    if not data.instanciated then
        -- Part was not finalized. We kind of deactivate it for the simulation:
        sim.setModelProperty(model.handle,sim.modelproperty_not_collidable+sim.modelproperty_not_detectable+sim.modelproperty_not_dynamic+sim.modelproperty_not_measurable+sim.modelproperty_not_renderable+sim.modelproperty_not_respondable+sim.modelproperty_not_visible)
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
    local repo,modelHolder=simBWF.getPartRepositoryHandles()
    if (repo and (sim.getObjectParent(model.handle)==modelHolder)) or model.finalizeModel then
        -- This means the box is part of the part repository or that we want to finalize the model (i.e. won't be customizable anymore)
        sysCall_cleanup_specific()
        local c=model.readInfo()
        sim.writeCustomDataBlock(model.handle,model.tagName,'')
    end
    model.dlg.cleanup()
end
