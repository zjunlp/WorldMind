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

function model.adjustFrame(frameState,width,height,doorState)
    -- nil for args that should stay same
    -- frameState: 0=not present, 1=present, 2=hidden
    -- doorState: 0=closed, 1=open, 2=hidden
    if frameState~=nil then
        local modelProperty=0
        if frameState==0 then
            modelProperty=sim.modelproperty_not_collidable+sim.modelproperty_not_detectable+sim.modelproperty_not_dynamic+
              sim.modelproperty_not_measurable+sim.modelproperty_not_renderable+sim.modelproperty_not_respondable+
              sim.modelproperty_not_visible+sim.modelproperty_not_showasinsidemodel
        end
        if frameState==2 then
            modelProperty=sim.modelproperty_not_visible
        end
        sim.setModelProperty(model.handle,modelProperty)
    end
    
    if doorState then
        for i=1,#model.handles.doorShapes,1 do
            if doorState<2 then
                sim.setObjectSpecialProperty(model.handles.doorShapes[i],sim.objectspecialproperty_collidable+sim.objectspecialproperty_detectable_all+sim.objectspecialproperty_measurable+sim.objectspecialproperty_renderable)
                sim.setObjectInt32Parameter(model.handles.doorShapes[i],sim.objintparam_visibility_layer,1)
            else
                sim.setObjectSpecialProperty(model.handles.doorShapes[i],0)
                sim.setObjectInt32Parameter(model.handles.doorShapes[i],sim.objintparam_visibility_layer,0)
            end
        end
        
        for i=1,#model.handles.doorJoints,1 do
            if doorState==0 then
                sim.setJointPosition(model.handles.doorJoints[i],0)
            end
            if doorState==1 then
                sim.setJointPosition(model.handles.doorJoints[i],30*math.pi/180)
            end
        end
    end

    if height then
        for i=1,#model.handles.heightJoints,1 do
            sim.setJointPosition(model.handles.heightJoints[i],height)
        end
    end

    if width then
        for i=1,#model.handles.widthJoints,1 do
            sim.setJointPosition(model.handles.widthJoints[i],width*0.5)
        end

        local w=width-0.95
        if w<0.2078 then
            w=0.2078
        end
        for i=1,#model.handles.centralResizeShapes,1 do
            model.setObjectSize(model.handles.centralResizeShapes[i],w,0.62072,0.070719)
        end
    end
end

function sysCall_init()
    model.codeVersion=1
end
