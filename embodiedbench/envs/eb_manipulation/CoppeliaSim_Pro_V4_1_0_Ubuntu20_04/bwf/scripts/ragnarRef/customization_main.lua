function model.dlg.refreshDlg()
    if model.dlg.ui then
        local sel=simBWF.getSelectedEditWidget(model.dlg.ui)

--        simUI.setEditValue(model.dlg.ui,1,simBWF.format("%.0f",sim.getJointPosition(model.handles.zOffsetJ)/0.001),true)
        simUI.setEditValue(model.dlg.ui,2,simBWF.format("%.0f",sim.getJointPosition(model.handles.xOffsetJ1)/0.001),true)
        simUI.setEditValue(model.dlg.ui,3,simBWF.format("%.0f",sim.getJointPosition(model.handles.yOffsetJ1)/0.001),true)
        simUI.setEditValue(model.dlg.ui,4,simBWF.format("%.2f",sim.getJointPosition(model.handles.alphaOffsetJ1)*180/math.pi),true)
        simUI.setEditValue(model.dlg.ui,5,simBWF.format("%.2f",sim.getJointPosition(model.handles.betaOffsetJ1)*180/math.pi),true)
        simBWF.setSelectedEditWidget(model.dlg.ui,sel)
    end
end
--[[
function model.dlg.zChange_callback(ui,id,newVal)
    sim.setJointPosition(model.handles.zOffsetJ,newVal/1000)
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
    model.executeIk()
end
--]]
function model.dlg.xChange_callback(ui,id,newVal)
    sim.setJointPosition(model.handles.xOffsetJ1,newVal/1000)
    sim.setJointPosition(model.handles.xOffsetJ2,newVal/1000)
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
    model.executeIk()
end

function model.dlg.yChange_callback(ui,id,newVal)
    sim.setJointPosition(model.handles.yOffsetJ1,newVal/1000)
    sim.setJointPosition(model.handles.yOffsetJ2,newVal/1000)
    sim.setJointPosition(model.handles.yOffsetJ3,newVal/1000)
    sim.setJointPosition(model.handles.yOffsetJ4,newVal/1000)
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
    model.executeIk()
end

function model.dlg.alphaChange_callback(ui,id,newVal)
    sim.setJointPosition(model.handles.alphaOffsetJ1,newVal*math.pi/180)
    sim.setJointPosition(model.handles.alphaOffsetJ2,newVal*math.pi/180)
    sim.setJointPosition(model.handles.alphaOffsetJ3,newVal*math.pi/180)
    sim.setJointPosition(model.handles.alphaOffsetJ4,newVal*math.pi/180)
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
    model.executeIk()
end

function model.dlg.betaChange_callback(ui,id,newVal)
    sim.setJointPosition(model.handles.betaOffsetJ1,newVal*math.pi/180)
    sim.setJointPosition(model.handles.betaOffsetJ2,newVal*math.pi/180)
    sim.setJointPosition(model.handles.betaOffsetJ3,newVal*math.pi/180)
    sim.setJointPosition(model.handles.betaOffsetJ4,newVal*math.pi/180)
    simBWF.markUndoPoint()
    model.dlg.refreshDlg()
    model.executeIk()
end

function model.dlg.resetJoints_callback(ui,id)
    local allJoints=sim.getObjectsInTree(model.handle,sim.object_joint_type)
    for i=1,#allJoints,1 do
        local mode=sim.getJointMode(allJoints[i])
        if mode==sim.jointmode_ik then
            sim.setJointPosition(allJoints[i],0)
        end
    end
    sim.setJointPosition(model.handles.ikJ1_a,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ2_a,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ3_a,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ4_a,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ1_b,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ2_b,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ3_b,0*math.pi/180)
    sim.setJointPosition(model.handles.ikJ4_b,0*math.pi/180)
    simBWF.markUndoPoint()
    model.executeIk()
end

function model.dlg.createDlg()
    if (not model.dlg.ui) and simBWF.canOpenPropertyDialog() then
        local xml=[[
            <group layout="form" flat="true">
                <label text="X (mm)"/>
                <edit on-editing-finished="model.dlg.xChange_callback" id="2"/>

                <label text="Y (mm)"/>
                <edit on-editing-finished="model.dlg.yChange_callback" id="3"/>

                <label text="Alpha (deg)"/>
                <edit on-editing-finished="model.dlg.alphaChange_callback" id="4"/>

                <label text="Beta (deg)"/>
                <edit on-editing-finished="model.dlg.betaChange_callback" id="5"/>
            </group>
            <button text="Reset IK joints" on-click="model.dlg.resetJoints_callback" id="6" />
        ]]
        --[[
                <label text="Z (mm)"/>
                <edit on-editing-finished="model.dlg.zChange_callback" id="1"/>
                --]]
        model.dlg.ui=simBWF.createCustomUi(xml,simBWF.getUiTitleNameFromModel(model.handle,_MODELVERSION_,_CODEVERSION_),model.dlg.previousDlgPos)
        model.dlg.refreshDlg()
    end
end

function model.dlg.showDlg()
    if not model.dlg.ui then
        model.dlg.createDlg()
    end
end

function model.dlg.removeDlg()
    if model.dlg.ui then
        local x,y=simUI.getPosition(model.dlg.ui)
        model.dlg.previousDlgPos={x,y}
        simUI.destroy(model.dlg.ui)
        model.dlg.ui=nil
    end
end

function model.dlg.showOrHideDlgIfNeeded()
    local s=sim.getObjectSelection()
    if s and #s>=1 and s[#s]==model.handle then
        model.dlg.showDlg()
    else
        model.dlg.removeDlg()
    end
end

function model.executeIk()
    for i=1,4,1 do
        -- We handle each branch individually:
        local ld=sim.getLinkDummy(model.handles.ikTips[i])
        if ld>=0 then
            -- We make sure we don't perform too large jumps:
            local p=sim.getObjectPosition(model.handles.ikTips[i],ld)
            local l=math.sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3])
            local steps=math.ceil(0.00001+l/0.05)
            local start=sim.getObjectPosition(model.handles.ikTips[i],-1)
            local goal=sim.getObjectPosition(ld,-1)
            for j=1,steps,1 do
                local t=j/steps
                local pos={start[1]*(1-t)+goal[1]*t,start[2]*(1-t)+goal[2]*t,start[3]*(1-t)+goal[3]*t}
                sim.setObjectPosition(ld,-1,pos)
                sim.handleIkGroup(model.handles.ikGroups[i])
            end
        end
    end
end


function sysCall_init()
    
    model.handles={}
    model.handles.zOffsetJ=sim.getObjectHandle('Ragnar_zOffset')
    model.handles.xOffsetJ1=sim.getObjectHandle('Ragnar_yOffsetLeft')
    model.handles.xOffsetJ2=sim.getObjectHandle('Ragnar_yOffsetRight')
    model.handles.yOffsetJ1=sim.getObjectHandle('Ragnar_xOffsetLeftFront')
    model.handles.yOffsetJ2=sim.getObjectHandle('Ragnar_xOffsetLeftRear')
    model.handles.yOffsetJ3=sim.getObjectHandle('Ragnar_xOffsetRightFront')
    model.handles.yOffsetJ4=sim.getObjectHandle('Ragnar_xOffsetRightRear')
    model.handles.alphaOffsetJ1=sim.getObjectHandle('Ragnar_zRotLeftFront')
    model.handles.alphaOffsetJ2=sim.getObjectHandle('Ragnar_zRotLeftRear')
    model.handles.alphaOffsetJ3=sim.getObjectHandle('Ragnar_zRotRightFront')
    model.handles.alphaOffsetJ4=sim.getObjectHandle('Ragnar_zRotRightRear')
    model.handles.betaOffsetJ1=sim.getObjectHandle('Ragnar_xRotLeftFront')
    model.handles.betaOffsetJ2=sim.getObjectHandle('Ragnar_xRotLeftRear')
    model.handles.betaOffsetJ3=sim.getObjectHandle('Ragnar_xRotRightFront')
    model.handles.betaOffsetJ4=sim.getObjectHandle('Ragnar_xRotRightRear')
    model.handles.ikJ1_a=sim.getObjectHandle('Ragnar_motor1')
    model.handles.ikJ2_a=sim.getObjectHandle('Ragnar_motor2')
    model.handles.ikJ3_a=sim.getObjectHandle('Ragnar_motor3')
    model.handles.ikJ4_a=sim.getObjectHandle('Ragnar_motor4')
    model.handles.ikJ1_b=sim.getObjectHandle('Ragnar_primaryArm1_j1')
    model.handles.ikJ2_b=sim.getObjectHandle('Ragnar_primaryArm2_j1')
    model.handles.ikJ3_b=sim.getObjectHandle('Ragnar_primaryArm3_j1')
    model.handles.ikJ4_b=sim.getObjectHandle('Ragnar_primaryArm4_j1')

    model.handles.ikTips={}
    for i=1,4,1 do
        model.handles.ikTips[i]=sim.getObjectHandle('Ragnar_secondaryArm'..i..'a_tip')
    end

    model.handles.ikGroups={}
    for i=1,4,1 do
        model.handles.ikGroups[i]=sim.getIkGroupHandle('ragnarIk_arm'..i)
    end

    model.dlg.previousDlgPos=simBWF.readSessionPersistentObjectData(model.handle,"dlgPosAndSize")
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
end

function sysCall_beforeSimulation()
    model.dlg.removeDlg()
end

function sysCall_beforeInstanceSwitch()
    model.dlg.removeDlg()
end

function sysCall_cleanup()
    model.dlg.removeDlg()
    simBWF.writeSessionPersistentObjectData(model.handle,"dlgPosAndSize",model.dlg.previousDlgPos)
end

