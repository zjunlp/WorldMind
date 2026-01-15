----------------------------------
-- Add your custom functions here:
----------------------------------

function AuxiliaryConsoleClose(...)
    debugFunc("AuxiliaryConsoleClose",...)
    local consoleHandle=...
    return sim.auxiliaryConsoleClose(consoleHandle)
end

function AuxiliaryConsoleOpen(...)
    debugFunc("AuxiliaryConsoleOpen",...)
    local title,maxLines,mode,position,size,textColor,backgroundColor=...
    for i=1,3,1 do
        textColor[i]=textColor[i]/255
        backgroundColor[i]=backgroundColor[i]/255
    end
    return sim.auxiliaryConsoleOpen(title,maxLines,mode,position,size,textColor,backgroundColor)
end

function AuxiliaryConsolePrint(...)
    debugFunc("AuxiliaryConsolePrint",...)
    local consoleHandle,text=...
    if #text==0 then
        text=nil
    end
    return sim.auxiliaryConsolePrint(consoleHandle,text)
end

function AuxiliaryConsoleShow(...)
    debugFunc("AuxiliaryConsoleShow",...)
    local consoleHandle,showState=...
    return sim.auxiliaryConsoleShow(consoleHandle,showState)
end

function AddDrawingObject_points(...)
    debugFunc("AddDrawingObject_points",...)
    local size,color,coords=...
    local adCol={0,0,0}
    local eCol={0,0,0}
    if color[1]>=0 then
        adCol={color[1]/255,color[2]/255,color[3]/255}
    else
        eCol={-color[1]/255,-color[2]/255,-color[3]/255}
    end
    local obj=sim.addDrawingObject(sim.drawing_points,size,0,-1,10000,adCol,nil,nil,eCol)
    for i=0,math.floor(#coords/3)-1,1 do
        sim.addDrawingObjectItem(obj,{coords[3*i+1],coords[3*i+2],coords[3*i+3]})
    end
    return obj
end

function AddDrawingObject_spheres(...)
    debugFunc("AddDrawingObject_spheres",...)
    local size,color,coords=...
    local adCol={0,0,0}
    local eCol={0,0,0}
    if color[1]>=0 then
        adCol={color[1]/255,color[2]/255,color[3]/255}
    else
        eCol={-color[1]/255,-color[2]/255,-color[3]/255}
    end
    local obj=sim.addDrawingObject(sim.drawing_spherepoints,size,0,-1,10000,adCol,nil,nil,eCol)
    for i=0,math.floor(#coords/3)-1,1 do
        sim.addDrawingObjectItem(obj,{coords[3*i+1],coords[3*i+2],coords[3*i+3]})
    end
    return obj
end

function AddDrawingObject_cubes(...)
    debugFunc("AddDrawingObject_cubes",...)
    local size,color,coords=...
    local adCol={0,0,0}
    local eCol={0,0,0}
    if color[1]>=0 then
        adCol={color[1]/255,color[2]/255,color[3]/255}
    else
        eCol={-color[1]/255,-color[2]/255,-color[3]/255}
    end
    local obj=sim.addDrawingObject(sim.drawing_cubepoints,size,0,-1,10000,adCol,nil,nil,eCol)
    for i=0,math.floor(#coords/3)-1,1 do
        sim.addDrawingObjectItem(obj,{coords[3*i+1],coords[3*i+2],coords[3*i+3],0,0,1})
    end
    return obj
end

function AddDrawingObject_segments(...)
    debugFunc("AddDrawingObject_segments",...)
    local lineSize,color,segments=...
    local adCol={0,0,0}
    local eCol={0,0,0}
    if color[1]>=0 then
        adCol={color[1]/255,color[2]/255,color[3]/255}
    else
        eCol={-color[1]/255,-color[2]/255,-color[3]/255}
    end
    local obj=sim.addDrawingObject(sim.drawing_lines,lineSize,0,-1,10000,adCol,nil,nil,eCol)
    for i=0,math.floor(#segments/6)-1,1 do
        sim.addDrawingObjectItem(obj,{segments[6*i+1],segments[6*i+2],segments[6*i+3],segments[6*i+4],segments[6*i+5],segments[6*i+6]})
    end
    return obj
end

function AddDrawingObject_triangles(...)
    debugFunc("AddDrawingObject_triangles",...)
    local color,triangles=...
    local adCol={0,0,0}
    local eCol={0,0,0}
    if color[1]>=0 then
        adCol={color[1]/255,color[2]/255,color[3]/255}
    else
        eCol={-color[1]/255,-color[2]/255,-color[3]/255}
    end
    local obj=sim.addDrawingObject(sim.drawing_triangles,0,0,-1,10000,adCol,nil,nil,eCol)
    for i=0,math.floor(#triangles/9)-1,1 do
        sim.addDrawingObjectItem(obj,{triangles[9*i+1],triangles[9*i+2],triangles[9*i+3],triangles[9*i+4],triangles[9*i+5],triangles[9*i+6],triangles[9*i+7],triangles[9*i+8],triangles[9*i+9]})
    end
    return obj
end

function RemoveDrawingObject(...)
    debugFunc("RemoveDrawingObject",...)
    local handle=...
    return sim.removeDrawingObject(handle)
end

function CallScriptFunction(...)
    local funcAtObjName,scriptType,packedArg=...
    local arg=messagePack.unpack(packedArg)
    debugFunc("CallScriptFunction",funcAtObjName,scriptType,arg)
    if type(scriptType)=='string' then
        scriptType=evalStr(scriptType)
    end
    return sim.callScriptFunction(funcAtObjName,scriptType,arg)
end

function CheckCollision(...)
    debugFunc("CheckCollision",...)
    local entity1,entity2=...
    if type(entity2)=='string' then
        entity2=evalStr(entity2)
    end
    return sim.checkCollision(entity1,entity2)
end

function ReadCollision(...)
    debugFunc("ReadCollision",...)
    local handle=...
    return sim.readCollision(handle)
end

function CheckDistance(...)
    debugFunc("CheckDistance",...)
    local entity1,entity2,threshold=...
    if type(entity2)=='string' then
        entity2=evalStr(entity2)
    end
    local r,d=sim.checkDistance(entity1,entity2,threshold)
    return r,d[7],{d[1],d[2],d[3]},{d[4],d[5],d[6]}
end

function ReadDistance(...)
    debugFunc("ReadDistance",...)
    local handle=...
    return sim.readDistance(handle)
end

function CheckProximitySensor(...)
    debugFunc("CheckProximitySensor",...)
    local sensor,entity=...
    if type(entity)=='string' then
        entity=evalStr(entity)
    end
    return sim.checkProximitySensor(sensor,entity)
end

function ReadProximitySensor(...)
    debugFunc("ReadProximitySensor",...)
    local handle=...
    return sim.readProximitySensor(handle)
end

function CheckVisionSensor(...)
    debugFunc("CheckVisionSensor",...)
    local sensor,entity=...
    if type(entity2)=='string' then
        entity=evalStr(entity)
    end
    return sim.checkVisionSensor(sensor,entity)
end

function ReadVisionSensor(...)
    debugFunc("ReadVisionSensor",...)
    local handle=...
    return sim.readVisionSensor(handle)
end

function ReadForceSensor(...)
    debugFunc("ReadForceSensor",...)
    local handle=...
    return sim.readForceSensor(handle)
end

function BreakForceSensor(...)
    debugFunc("BreakForceSensor",...)
    local handle=...
    return sim.breakForceSensor(handle)
end

function ClearFloatSignal(...)
    debugFunc("ClearFloatSignal",...)
    local sig=...
    return sim.clearFloatSignal(sig)
end

function ClearIntegerSignal(...)
    debugFunc("ClearIntegerSignal",...)
    local sig=...
    return sim.clearIntegerSignal(sig)
end

function ClearStringSignal(...)
    debugFunc("ClearStringSignal",...)
    local sig=...
    return sim.clearStringSignal(sig)
end

function SetFloatSignal(...)
    debugFunc("SetFloatSignal",...)
    local sig,v=...
    return sim.setFloatSignal(sig,v)
end

function SetIntSignal(...)
    debugFunc("SetIntSignal",...)
    local sig,v=...
    return sim.setIntegerSignal(sig,v)
end

function SetStringSignal(...)
    debugFunc("SetStringSignal",...)
    local sig,v=...
    return sim.setStringSignal(sig,v)
end

function GetFloatSignal(...)
    debugFunc("GetFloatSignal",...)
    local sig=...
    return sim.getFloatSignal(sig)
end

function GetIntSignal(...)
    debugFunc("GetIntSignal",...)
    local sig=...
    return sim.getIntegerSignal(sig)
end

function GetStringSignal(...)
    debugFunc("GetStringSignal",...)
    local sig=...
    return sim.getStringSignal(sig)
end

function AddStatusbarMessage(...)
    debugFunc("AddStatusbarMessage",...)
    local txt=...
    return sim.addLog(sim.verbosity_msgs,txt)
end

function GetObjectPosition(...)
    debugFunc("GetObjectPosition",...)
    local objHandle,relObjHandle=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.getObjectPosition(objHandle,relObjHandle)
end

function GetObjectHandle(...)
    debugFunc("GetObjectHandle",...)
    local objName=...
    if string.find(objName,'#')==nil then
        objName=objName..'#'
    end
    return sim.getObjectHandle(objName)
end

function StartSimulation(...)
    debugFunc("StartSimulation",...)
    return sim.startSimulation()
end

function StopSimulation(...)
    debugFunc("StopSimulation",...)
    return sim.stopSimulation()
end

function PauseSimulation(...)
    debugFunc("PauseSimulation",...)
    return sim.pauseSimulation()
end

function GetVisionSensorImage(...)
    debugFunc("GetVisionSensorImage",...)
    local objHandle,greyScale=...
    if greyScale then
        objHandle=objHandle+sim.handleflag_greyscale
    end
    local img,x,y=sim.getVisionSensorCharImage(objHandle)
    return {x,y},img
end

function GetVisionSensorResolution(...)
    debugFunc("GetVisionSensorResolution",...)
    local objHandle=...
    return sim.getVisionSensorResolution(objHandle)
end

function GetVisionSensorDepthBuffer(...)
    debugFunc("GetVisionSensorDepthBuffer",...)
    local objHandle,toMeters,asByteArray=...
    if toMeters then
        objHandle=objHandle+sim.handleflag_depthbuffermeters
    end
    if asByteArray then
        objHandle=objHandle+sim.handleflag_codedstring
    end
    local buff=sim.getVisionSensorDepthBuffer(objHandle)
    return {x,y},buff
end

function SetVisionSensorImage(...)
    debugFunc("SetVisionSensorImage",...)
    local objHandle,greyScale,img=...
    if greyScale then
        objHandle=objHandle+sim.handleflag_greyscale
    end
    return sim.setVisionSensorCharImage(objHandle,img)
end

function SetObjectPosition(...)
    debugFunc("SetObjectPosition",...)
    local objHandle,relObjHandle,pos=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.setObjectPosition(objHandle,relObjHandle,pos)
end

function GetObjectOrientation(...)
    debugFunc("GetObjectOrientation",...)
    local objHandle,relObjHandle=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.getObjectOrientation(objHandle,relObjHandle)
end

function SetObjectOrientation(...)
    debugFunc("SetObjectOrientation",...)
    local objHandle,relObjHandle,euler=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.setObjectOrientation(objHandle,relObjHandle,euler)
end

function GetObjectQuaternion(...)
    debugFunc("GetObjectQuaternion",...)
    local objHandle,relObjHandle=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.getObjectQuaternion(objHandle,relObjHandle)
end

function SetObjectQuaternion(...)
    debugFunc("SetObjectQuaternion",...)
    local objHandle,relObjHandle,quat=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.setObjectQuaternion(objHandle,relObjHandle,quat)
end

function GetObjectPose(...)
    debugFunc("GetObjectPose",...)
    local objHandle,relObjHandle=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    local pose=sim.getObjectPosition(objHandle,relObjHandle)
    local quat=sim.getObjectQuaternion(objHandle,relObjHandle)
    pose[4]=quat[1]
    pose[5]=quat[2]
    pose[6]=quat[3]
    pose[7]=quat[4]
    return pose
end

function SetObjectPose(...)
    debugFunc("SetObjectPose",...)
    local objHandle,relObjHandle,pose=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    sim.setObjectPosition(objHandle,relObjHandle,pose)
    sim.setObjectQuaternion(objHandle,relObjHandle,{pose[4],pose[5],pose[6],pose[7]})
    return 1
end

function GetObjectMatrix(...)
    debugFunc("GetObjectMatrix",...)
    local objHandle,relObjHandle=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.getObjectMatrix(objHandle,relObjHandle)
end

function SetObjectMatrix(...)
    debugFunc("SetObjectMatrix",...)
    local objHandle,relObjHandle,matr=...
    if type(relObjHandle)=='string' then
        relObjHandle=evalStr(relObjHandle)
    end
    return sim.setObjectMatrix(objHandle,relObjHandle,matr)
end

function CopyPasteObjects(...)
    debugFunc("CopyPasteObjects",...)
    local objHandles,options=...
    return sim.copyPasteObjects(objHandles,options)
end

function RemoveObjects(...)
    debugFunc("RemoveObjects",...)
    local objHandles,options=...
    local allObjs1=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    if sim.boolAnd32(options,2)>0 then
        sim.removeObject(sim.handle_all)
    else
        if sim.boolAnd32(options,1)>0 then
            for i=1,#objHandles,1 do
                local h=objHandles[i]
                if sim.isHandleValid(h)>0 then
                    local mp=sim.getModelProperty(h)
                    if sim.boolAnd32(mp,sim.modelproperty_not_model)>0 then
                        sim.removeObject(objHandles[i])
                    else
                        sim.removeModel(objHandles[i])
                    end
                end
            end
        else
            for i=1,#objHandles,1 do
                sim.removeObject(objHandles[i])
            end
        end
    end
    local allObjs2=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)
    return #allObjs1-#allObjs2
end

function CloseScene(...)
    debugFunc("CloseScene",...)
    return sim.closeScene()
end

function SetStringParameter(...)
    debugFunc("SetStringParameter",...)
    local paramId,val=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setStringParameter(paramId,val)
end

function GetStringParameter(...)
    debugFunc("GetStringParameter",...)
    local paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.getStringParameter(paramId)
end

function SetFloatParameter(...)
    debugFunc("SetFloatParameter",...)
    local paramId,val=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setFloatParameter(paramId,val)
end

function GetFloatParameter(...)
    debugFunc("GetFloatParameter",...)
    local paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.getFloatParameter(paramId)
end

function SetIntParameter(...)
    debugFunc("SetIntParameter",...)
    local paramId,val=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setInt32Parameter(paramId,val)
end

function GetIntParameter(...)
    debugFunc("GetIntParameter",...)
    local paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.getInt32Parameter(paramId)
end

function SetBoolParameter(...)
    debugFunc("SetBoolParameter",...)
    local paramId,val=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setBoolParameter(paramId,val)
end

function GetBoolParameter(...)
    debugFunc("GetBoolParameter",...)
    local paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.getBoolParameter(paramId)
end

function SetArrayParameter(...)
    debugFunc("SetArrayParameter",...)
    local paramId,val=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setArrayParameter(paramId,val)
end

function GetArrayParameter(...)
    debugFunc("GetArrayParameter",...)
    local paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.getArrayParameter(paramId)
end
function DisplayDialog(...)
    debugFunc("DisplayDialog",...)
    local titleText,mainText,dlgType,initText=...
    if type(dlgType)=='string' then
        dlgType=evalStr(dlgType)
    end
    return sim.displayDialog(titleText,mainText,dlgType,initText)
end

function GetDialogResult(...)
    debugFunc("GetDialogResult",...)
    local dlgHandle=...
    return tostring(sim.getDialogResult(dlgHandle))
end

function GetDialogInput(...)
    debugFunc("GetDialogInput",...)
    local dlgHandle=...
    return sim.getDialogInput(dlgHandle)
end

function EndDialog(...)
    debugFunc("EndDialog",...)
    local dlgHandle=...
    return sim.endDialog(dlgHandle)
end

function ExecuteScriptString(...)
    debugFunc("ExecuteScriptString",...)
    local str=...
    return evalStr(str)
end

function GetCollectionHandle(...)
    debugFunc("GetCollectionHandle",...)
    local objName=...
    if string.find(objName,'#')==nil then
        objName=objName..'#'
    end
    return sim.getCollectionHandle(objName)
end

function GetCollisionHandle(...)
    debugFunc("GetCollisionHandle",...)
    local name=...
    if string.find(name,'#')==nil then
        name=name..'#'
    end
    return sim.getCollisionHandle(name)
end

function GetDistanceHandle(...)
    debugFunc("GetDistanceHandle",...)
    local name=...
    if string.find(name,'#')==nil then
        name=name..'#'
    end
    return sim.getDistanceHandle(name)
end

function GetJointForce(...)
    debugFunc("GetJointForce",...)
    local handle=...
    return sim.getJointForce(handle)
end

function GetJointMaxForce(...)
    debugFunc("GetJointMaxForce",...)
    local handle=...
    return sim.getJointMaxForce(handle)
end

function SetJointForce(...)
    debugFunc("SetJointForce",...)
    local handle,f=...
    return sim.setJointForce(handle,f)
end

function SetJointMaxForce(...)
    debugFunc("SetJointMaxForce",...)
    local handle,f=...
    return sim.setJointMaxForce(handle,f)
end

function GetJointPosition(...)
    debugFunc("GetJointPosition",...)
    local handle=...
    return sim.getJointPosition(handle)
end

function SetJointPosition(...)
    debugFunc("SetJointPosition",...)
    local handle,p=...
    return sim.setJointPosition(handle,p)
end

function GetJointTargetPosition(...)
    debugFunc("GetJointTargetPosition",...)
    local handle=...
    return sim.getJointTargetPosition(handle)
end

function SetJointTargetPosition(...)
    debugFunc("SetJointTargetPosition",...)
    local handle,tp=...
    return sim.setJointTargetPosition(handle,tp)
end

function GetJointTargetVelocity(...)
    debugFunc("GetJointTargetVelocity",...)
    local handle=...
    return sim.getJointTargetVelocity(handle)
end

function SetJointTargetVelocity(...)
    debugFunc("SetJointTargetVelocity",...)
    local handle,tv=...
    return sim.setJointTargetVelocity(handle,tv)
end

function GetObjectChild(...)
    debugFunc("GetObjectChild",...)
    local handle,index=...
    return sim.getObjectChild(handle,index)
end

function GetObjectParent(...)
    debugFunc("GetObjectParent",...)
    local handle=...
    return sim.getObjectParent(handle)
end

function SetObjectParent(...)
    debugFunc("SetObjectParent",...)
    local handle,parentHandle,assembly,keepInPlace=...
    if assembly then
        handle=handle+sim.handleflag_assembly
    end
    return sim.setObjectParent(handle,parentHandle,keepInPlace)
end

function GetObjectsInTree(...)
    debugFunc("GetObjectsInTree",...)
    local treeBase,objType,options=...
    if type(treeBase)=='string' then
        treeBase=evalStr(treeBase)
    end
    if type(objType)=='string' then
        objType=evalStr(objType)
    end
    return sim.getObjectsInTree(treeBase,objType,options)
end

function GetObjectName(...)
    debugFunc("GetObjectName",...)
    local handle,altName=...
    if altName then
        handle=handle+sim.handleflag_altname
    end
    return sim.getObjectName(handle)
end

function GetObjectFloatParameter(...)
    debugFunc("GetObjectFloatParameter",...)
    local handle,paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    local r,retV=sim.getObjectFloatParameter(handle,paramId)
    return retV
end

function GetObjectIntParameter(...)
    debugFunc("GetObjectIntParameter",...)
    local handle,paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    local r,retV=sim.getObjectInt32Parameter(handle,paramId)
    return retV
end

function GetObjectStringParameter(...)
    debugFunc("GetObjectStringParameter",...)
    local handle,paramId=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    local r,retV=sim.getObjectStringParameter(handle,paramId)
    return retV
end

function SetObjectFloatParameter(...)
    debugFunc("SetObjectFloatParameter",...)
    local handle,paramId,v=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setObjectFloatParameter(handle,paramId,v)
end

function SetObjectIntParameter(...)
    debugFunc("SetObjectIntParameter",...)
    local handle,paramId,v=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setObjectInt32Parameter(handle,paramId,v)
end

function SetObjectStringParameter(...)
    debugFunc("SetObjectStringParameter",...)
    local handle,paramId,v=...
    if type(paramId)=='string' then
        paramId=evalStr(paramId)
    end
    return sim.setObjectStringParameter(handle,paramId,v)
end

function GetSimulationTime(...)
    debugFunc("GetSimulationTime",...)
    return sim.getSimulationTime()
end

function GetSimulationTimeStep(...)
    debugFunc("GetSimulationTimeStep",...)
    return sim.getSimulationTimeStep()
end

function GetServerTimeInMs(...)
    debugFunc("GetServerTimeInMs",...)
    return sim.getSystemTimeInMs(-2)
end

function GetSimulationState(...)
    debugFunc("GetSimulationState",...)
    local s=sim.getSimulationState()
    if s~=sim.simulation_stopped and s~=sim.simulation_paused then
        s=sim.simulation_advancing
    end
    return s
end

function EvaluateToInt(...)
    debugFunc("EvaluateToInt",...)
    local str=...
    return evalStr(str)
end

function EvaluateToStr(...)
    debugFunc("EvaluateToStr",...)
    local str=...
    return evalStr(str)
end

function GetObjects(...)
    debugFunc("GetObjects",...)
    local objType=...
    if type(objType)=='string' then
        objType=evalStr(objType)
    end
    local retVal={}
    local i=0
    local h=sim.getObjects(i,objType)
    while h>=0 do
        retVal[#retVal+1]=h
        i=i+1
        h=sim.getObjects(i,objType)
    end
    return retVal
end

function CreateDummy(...)
    debugFunc("CreateDummy",...)
    local cols={0,0,0,0,0,0,0.5,0.5,0.5,0,0,0}
    local size,color=...
    if color[1]>=0 then
        cols[1]=color[1]/255
        cols[2]=color[2]/255
        cols[3]=color[3]/255
    else
        cols[10]=-color[1]/255
        cols[11]=-color[2]/255
        cols[12]=-color[3]/255
    end
    return sim.createDummy(size,cols)
end

function GetObjectSelection(...)
    debugFunc("GetObjectSelection",...)
    local t=sim.getObjectSelection()
    if t==nil then
        t={}
    end
    return t
end

function SetObjectSelection(...)
    debugFunc("SetObjectSelection",...)
    local sel=...
    sim.removeObjectFromSelection(sim.handle_all,-1);
    return sim.addObjectToSelection(sel)
end

function GetObjectVelocity(...)
    debugFunc("GetObjectVelocity",...)
    local handle=...
    return sim.getObjectVelocity(handle+sim.handleflag_axis)
end

function LoadModelFromFile(...)
    debugFunc("LoadModelFromFile",...)
    local filename=...
    local s=sim.getObjectSelection()
    local h=sim.loadModel(filename)
    sim.removeObjectFromSelection(sim.handle_all,-1);
    if s then
        sim.addObjectToSelection(s)
    end
    return h
end

function LoadModelFromBuffer(...)
    debugFunc("LoadModelFromBuffer",...)
    local buff=...
    local s=sim.getObjectSelection()
    local h=sim.loadModel(buff)
    sim.removeObjectFromSelection(sim.handle_all,-1);
    if s then
        sim.addObjectToSelection(s)
    end
    return h
end

function LoadScene(...)
    debugFunc("LoadScene",...)
    local filename=...
    return sim.loadScene(filename)
end

-----------------------------------------
-----------------------------------------

function evalStr(str)
    local f=loadstring('return ('..str..')')
    return f()
end

function timeStr()
    local t
    if sim.getSimulationState()==sim.simulation_stopped then
        t=os.date('*t')
        t=string.format('[%02d:%02d:%02d] ',t.hour,t.min,t.sec)
    else
        local st=sim.getSimulationTime()
        t=os.date('*t',3600*23+st)
        t=string.format('[%02d:%02d:%02d.%02d] ',t.hour,t.min,t.sec,st%1)
    end
    return t
end

function Synchronous(...)
    debugFunc("Synchronous",...)
    local enable=...
    if enable and not syncMode then
        syncModeWait=true
    end
    syncMode=enable
    return true
end

function SynchronousTrigger(...)
    debugFunc("SynchronousTrigger",...)
    syncModeWait=false
    return true
end

function GetSimulationStepDone(...)
--    debugFunc("GetSimulationStepDone",...)
    local retVal={}
    retVal.simulationTime=sim.getSimulationTime()
    retVal.simulationState=tostring(sim.getSimulationState())
    retVal.simulationTimeStep=sim.getSimulationTimeStep()
    return retVal
end

function GetSimulationStepStarted(...)
--    debugFunc("GetSimulationStepStarted",...)
    local retVal={}
    retVal.simulationTime=sim.getSimulationTime()
    retVal.simulationState=tostring(sim.getSimulationState())
    retVal.simulationTimeStep=sim.getSimulationTimeStep()
    return retVal
end

function DisconnectClient(...)
    debugFunc("DisconnectClient",...)
    local clientId=...
    local val=allPublishers[clientId]
    if val then
        for topic,value in pairs(val) do
            if value.handle~=defaultPublisher then
                simB0.socketCleanup(value.handle)
                simB0.publisherDestroy(value.handle)
            end
        end
        allPublishers[clientId]=nil
    end
    local val=dedicatedSubscribers[clientId]
    if val then
        for topic,value in pairs(val) do
            simB0.socketCleanup(value.handle)
            simB0.subscriberDestroy(value.handle)
        end
        dedicatedSubscribers[clientId]=nil
    end
    allClients[clientId]=nil
end

function Ping(...)
    debugFunc("Ping",...)
    return 'Pong'
end

-----------------------------------------
-----------------------------------------

function debugFunc(funcName,...)
    lastFuncName=funcName
    if modelData and modelData.debugLevel>=3 then
        local arg=getAsString(...)
        if arg=='' then
            arg='none'
        end
        local a=timeStr().." calling function '"..funcName.."' with following arguments: "..arg
        sim.addLog(sim.verbosity_msgs,a)
    end
end

function PCALL(func,printErrors,...)
    local res,a,b,c,d,e,f,g,h,i,j,k=pcall(func,...)
    
    if modelData and modelData.debugLevel>=1 and (not res) and printErrors then
        local a=string.format(timeStr().." error while calling function '%s': %s",lastFuncName,a)
        sim.addLog(sim.verbosity_msgs,a)
    end
    
    return res,a,b,c,d,e,f,g,h,i,j,k
    
--    res,val1,val2,val3,val4,val5,val6,val7,val8=pcall(func,...)
--    if val1==nil then val1='__##LUANIL##__' end -- make sure we have 2 ret arguments
---    val=func(...)
---    print(res,val)
--    return res,val1,val2,val3,val4,val5,val6,val7,val8
end

function PACKPUBMSG(topic,res,...)
    local a={...}
    local b={topic,{res}}
    for i=1,#a,1 do
        b[2][#b[2]+1]=a[i]
    end
    return messagePack.pack(b)
end

function PACKSERVMSG(...)
    local a={...}
    return messagePack.pack(a)
end

function createNode()
    if not b0Node then
        if modelData.debugLevel>=1 then
            local a=string.format(timeStr().." creating BlueZero node '%s' and associated publisher, subscriber and service server (on channel '%s')",modelData.nodeName,modelData.channelName)
            sim.addLog(sim.verbosity_msgs,a)
        end
        modelData.currentNodeName=modelData.nodeName
        modelData.currentChannelName=modelData.channelName
        if not initStg then
            local ui
            if simUI then
                local xml = [[ <ui closeable="false" resizable="false" title="BlueZero" modal="true">
                        <label text="Looking for BlueZero resolver..." style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                        <label text="This can take several seconds." style="* {font-size: 20px; font-weight: bold; margin-left: 20px; margin-right: 20px;}"/>
                        </ui> ]]
                ui=simUI.create(xml)
            end
            if not simB0.pingResolver() then
                sim.addLog(sim.verbosity_msgs,"B0 Remote API: B0 resolver was not detected. Launching it from here...")
                sim.launchExecutable('b0_resolver','',1)
                local st=sim.getSystemTimeInMs(-1)
                while sim.getSystemTimeInMs(st)<1000 do end
            end
            if simUI then
                simUI.destroy(ui)
            end
            
            if simB0.pingResolver() then
                messagePack=require('messagePack')
                if modelData.packStrAsBin then
                    messagePack.set_string('binary')
                else
                    messagePack.set_string('string')
                end
                initStg=1
            else
                initStg=0
                sim.addLog(sim.verbosity_msgs,timeStr().." B0 resolver could not be launched.")
            end
        end

        if initStg==1 then
            b0Node=simB0.nodeCreate(modelData.nodeName)
            serviceServer=simB0.serviceServerCreate(b0Node,modelData.channelName..'SerX','serviceServer_callback')
            defaultPublisher=simB0.publisherCreate(b0Node,modelData.channelName..'PubX')
            defaultSubscriber=simB0.subscriberCreate(b0Node,modelData.channelName..'SubX','defaultSubscriber_callback')
            dedicatedSubscribers={} -- key is clientId, value is a map with: key is subscriberTopic, value is another map: handle
            allPublishers={} -- key is clientId, value is a map with: key is publisherTopic, value is another map: pubHandle, cmds=listOfRegisteredCmds 
            simB0.nodeInit(b0Node)
            allClients={}
            allSubscribers={}
        end
    end
end

function destroyNode()
    if b0Node then
        if modelData.debugLevel>=1 then
            local a=string.format(timeStr().." destroying BlueZero node '%s' and associated publisher, subscriber and service server (on channel '%s')",modelData.currentNodeName,modelData.currentChannelName)
            sim.addLog(sim.verbosity_msgs,a)
        end
        modelData.currentNodeName=nil
        modelData.currentChannelName=nil
        local clients={}
        for key,val in pairs(allClients) do
            clients[#clients+1]=key
        end
        for i=1,#clients,1 do
            DisconnectClient(clients[i])
        end
        --simB0.shutdown(b0Node)
        simB0.nodeCleanup(b0Node)
        simB0.publisherDestroy(defaultPublisher)
        simB0.subscriberDestroy(defaultSubscriber)
        simB0.serviceServerDestroy(serviceServer)
        simB0.nodeDestroy(b0Node)
    end
    allPublishers={}
    dedicatedSubscribers={}
    allClients={}
    b0Node=nil
end

function sendAndSpin(calledMoment)
    local retVal=true
    local publishSimulationStepFinished_ClientAnddata={}
    local publishSimulationStepStarted_ClientAnddata={}
    local publisherCntForClients={}
        
    if b0Node then
        -- Handle subscriber(s) and service calls:
        simB0.nodeSpinOnce(b0Node)
        for clientId,val in pairs(dedicatedSubscribers) do
            for topic,value in pairs(val) do
                local msg=''
                while simB0.socketPoll(value.handle) do
                    msg=simB0.socketRead(value.handle)
                    if not value.dropMessages then
                        dedicatedSubscriber_callback(msg)
                    end
                end
                if value.dropMessages and #msg>0 then
                    dedicatedSubscriber_callback(msg)
                end
--                simB0.socketSpinOnce(value.handle)
            end
        end

        -- Handle publishing:
        local clientsToRemove={}
        for clientId,val in pairs(allPublishers) do
            if not hasClientReachedMaxInactivityTime(clientId) then
                -- Ok, that client appears to be still active
                for topic,value in pairs(val) do
                    local publisher=value.handle
                    local triggerInterval=value.triggerInterval
                    local cmdList=value.cmds
                    for i=1,#cmdList,1 do
                        local cmd=cmdList[i]
                        if triggerInterval==0 or calledMoment==0 or (nextSimulationStepUnderway and not (sim.getSimulationState()==sim.simulation_paused)) or cmd.func=='GetSimulationStepStarted' then
                            cmd.triggerIntervalCnt=cmd.triggerIntervalCnt-1
                            if cmd.triggerIntervalCnt<=0 then
                                cmd.triggerIntervalCnt=triggerInterval
--                                local result,retVal=PCALL(_G[cmd.func],unpack(cmd.args))
--                                retVal=messagePack.pack({topic,{result,retVal}})
                                local retVal=PACKPUBMSG(topic,PCALL(_G[cmd.func],true,unpack(cmd.args)))
                                
                                if cmd.func=='GetSimulationStepDone' then
                                    publishSimulationStepFinished_ClientAnddata[clientId]={publisher,retVal} -- publish this one last! (further down)
                                else
                                    if cmd.func=='GetSimulationStepStarted' then
                                        publishSimulationStepStarted_ClientAnddata[clientId]={publisher,retVal} -- publish this one last! (further down)
                                    else
                                        publish(clientId,publisher,cmd.func,retVal,publisherCntForClients)
                                    end
                                end
                            end
                        end
                    end
                    
                end
            else
                clientsToRemove[#clientsToRemove+1]=clientId
            end
        end
        
        -- Remove publishers of inactive clients:
        for i=1,#clientsToRemove,1 do
            local clientId=clientsToRemove[i]
            DisconnectClient(clientId)
            if modelData.debugLevel>=1 then
                local a=string.format(timeStr().." destroyed all streaming functions for client '%s' after detection of inactivity",clientId)
                sim.addLog(sim.verbosity_msgs,a)
            end
        end
    end
    
    if calledMoment==1 then -- i.e. before main script
        if nextSimulationStepUnderway then
            -- Handle publishing of simulationStepFinished here (special):
            for key,value in pairs(publishSimulationStepFinished_ClientAnddata) do
                debugFunc('GetSimulationStepDone',nil)
                publish(key,value[1],'GetSimulationStepDone',value[2],publisherCntForClients)
            end
        end
    
        if syncMode then
            if syncModeWait then
                retVal=false
            else
                syncModeWait=true
            end
        end
        if retVal then
            nextSimulationStepUnderway=true
            -- Handle publishing of simulationStepStarted here (special):
            for key,value in pairs(publishSimulationStepStarted_ClientAnddata) do
                debugFunc('GetSimulationStepStarted',nil)
                publish(key,value[1],'GetSimulationStepStarted',value[2],publisherCntForClients)
            end
        else
            nextSimulationStepUnderway=false
        end
    end
    
    for publisher,client in pairs(publisherCntForClients) do
        local append
        if publisher==defaultPublisher then
            append=' (default publisher)'
        else
            append=' (dedicated publisher)'
        end
        
        local msgCnt=0
        local clientCnt=0
        local funcs=''
        for key,value in pairs(client) do
            clientCnt=clientCnt+1
            msgCnt=msgCnt+value.cnt
            if funcs=='' then
                funcs=value.funcs
            else
                funcs=funcs..'|'..value.funcs
            end
        end
        if msgCnt>0 and modelData and modelData.debugLevel>=2 then
            local a=string.format(timeStr().." published %i message(s) to %i client(s): %s",msgCnt,clientCnt,funcs)
            sim.addLog(sim.verbosity_msgs,a)
        end
    end
    
    return retVal
end

function publish(clientId,publisher,func,retVal,publisherCntForClients)
    simB0.publisherPublish(publisher,retVal)
    
    if not publisherCntForClients[publisher] then
        publisherCntForClients[publisher]={}
    end
    
    if not publisherCntForClients[publisher][clientId] then
        publisherCntForClients[publisher][clientId]={cnt=1,funcs=func}
    else
        publisherCntForClients[publisher][clientId].cnt=publisherCntForClients[publisher][clientId].cnt+1
        publisherCntForClients[publisher][clientId].funcs=publisherCntForClients[publisher][clientId].funcs..'|'..func
    end
end

function updateClientLastActivityTime(clientId)
    if not allClients[clientId] then
        allClients[clientId]={maxInactivityTimeMs=60*1000}
    end
    local val=allClients[clientId]
    val.lastActivityTimeMs=sim.getSystemTimeInMs(-1)
end

function setClientMaxInactivityTime(clientId,maxInactivityTime)
    local val=allClients[clientId]
    val.maxInactivityTimeMs=maxInactivityTime*1000
end

function hasClientReachedMaxInactivityTime(clientId)
    local val=allClients[clientId]
    if val.maxInactivityTimeMs==0 then
        return false
    end
    return sim.getSystemTimeInMs(val.lastActivityTimeMs)>val.maxInactivityTimeMs
end

function serviceServer_callback(receiveMsg)
    local retVal=PACKSERVMSG(true)
 --   local result=true
 --   local data=true
    receiveMsg=messagePack.unpack(receiveMsg)
    local funcName=receiveMsg[1][1]
    local clientId=receiveMsg[1][2]
    local topic=receiveMsg[1][3]
    local task=receiveMsg[1][4] -- 0=normal serviceCall, 1=received on default subscriber, 2=register streaming cmd on default publisher, 3=received on dedicated subscriber, 4=register streaming cmd on dedicated publisher
    local funcArgs=receiveMsg[2]
    updateClientLastActivityTime(clientId)

    if not handlePublisherSetupFunctions(task,funcName,clientId,topic,funcArgs) then
        if funcName=='createSubscriber' then
            local subscr=simB0.subscriberCreate(b0Node,funcArgs[1],'dedicatedSubscriber_callback',false,true)
       --     simB0.socketSetOption(subscr,'conflate',1)
            simB0.socketInit(subscr);
            if not dedicatedSubscribers[clientId] then
                dedicatedSubscribers[clientId]={}
            end
            dedicatedSubscribers[clientId][funcArgs[1]]={handle=subscr,dropMessages=funcArgs[2]}
            if modelData.debugLevel>=1 then
                local a=string.format(timeStr().." creating dedicated subscriber for client '%s' with topic '%s'",clientId,funcArgs[1])
                sim.addLog(sim.verbosity_msgs,a)
            end
        elseif funcName=='inactivityTolerance' then
            setClientMaxInactivityTime(clientId,funcArgs[1])
            if modelData.debugLevel>=2 then
                local a=string.format(timeStr().." setting max. inactivity tolerance for client '%s'",clientId)
                sim.addLog(sim.verbosity_msgs,a)
            end
        else
            retVal=PACKSERVMSG(PCALL(_G[funcName],true,unpack(funcArgs)))
    --        result,data=PCALL(_G[funcName],unpack(funcArgs))
            if modelData.debugLevel>=2 then
                local a=string.format(timeStr().." called function for client '%s': %s (service call)",clientId,receiveMsg[1][1])
                sim.addLog(sim.verbosity_msgs,a)
            end
        end
    end
--    return messagePack.pack({result,data})
    return retVal
end

function handlePublisherSetupFunctions(task,funcName,clientId,topic,funcArgs)
    if task==2 then
        -- We want to register a command to be constantly executed on the default publisher:
        if not allPublishers[clientId] then
            allPublishers[clientId]={}
        end
        if not allPublishers[clientId][topic] then
            allPublishers[clientId][topic]={handle=defaultPublisher,cmds={},triggerInterval=1}
        end
        local val=allPublishers[clientId][topic]
        val.cmds[#val.cmds+1]={func=funcName,args=funcArgs,triggerIntervalCnt=1}
        if modelData.debugLevel>=1 then
            local a=string.format(timeStr().." registering streaming function '%s' for client '%s' on topic '%s' (default publisher)",funcName,clientId,topic)
            sim.addLog(sim.verbosity_msgs,a)
        end
    elseif task==4 then
        -- We want to register a command to be constantly executed on a dedicated publisher:
        if allPublishers[clientId] and  allPublishers[clientId][topic] then
            local val=allPublishers[clientId][topic]
            allCmds=val.cmds
            allCmds[#allCmds+1]={func=funcName,args=funcArgs,triggerIntervalCnt=1}
            if modelData.debugLevel>=1 then
                local a=string.format(timeStr().." registering streaming function '%s' for client '%s' on topic '%s' (dedicated publisher)",funcName,clientId,topic)
                sim.addLog(sim.verbosity_msgs,a)
            end
        end
    else
        if funcName=='createPublisher' then
            local pub=simB0.publisherCreate(b0Node,funcArgs[1],false,true)
        --    simB0.socketSetOption(pub,'conflate',1)
            simB0.socketInit(pub);
            if not allPublishers[clientId] then
                allPublishers[clientId]={}
            end
            local targetTopic=funcArgs[1]
            local trigInterv=funcArgs[2]
            allPublishers[clientId][targetTopic]={handle=pub,cmds={},triggerInterval=trigInterv}
            if modelData.debugLevel>=1 then
                local a=string.format(timeStr().." creating dedicated publisher for client '%s' with topic '%s'",clientId,targetTopic)
                sim.addLog(sim.verbosity_msgs,a)
            end
            return true
        elseif funcName=='setDefaultPublisherPubInterval' then
            if not allPublishers[clientId] then
                allPublishers[clientId]={}
            end
            local targetTopic=funcArgs[1]
            local trigInterv=funcArgs[2]
            if not allPublishers[clientId][targetTopic] then
                allPublishers[clientId][targetTopic]={handle=defaultPublisher,cmds={},triggerInterval=trigInterv}
            end
            if modelData.debugLevel>=2 then
                local a=string.format(timeStr().." setting default publisher interval for client '%s' with topic '%s'",clientId,targetTopic)
                sim.addLog(sim.verbosity_msgs,a)
            end
            return true
        elseif funcName=='stopDefaultPublisher' or funcName=='stopPublisher' then
            local topic=funcArgs[1]
            if allPublishers[clientId] then
                if allPublishers[clientId][topic] then
                    local nn='default'
                    if funcName=='stopPublisher' then
                        simB0.socketCleanup(allPublishers[clientId][topic].handle)
                        simB0.publisherDestroy(allPublishers[clientId][topic].handle)
                        nn='dedicated'
                    end
                    local cmds=allPublishers[clientId][topic].cmds
                    allPublishers[clientId][topic]=nil
                    if modelData.debugLevel>=1 then
                        local cm=''
                        if #cmds>0 then
                            cm='. Following streaming functions on that topic will be unregistered:'
                            for i=1,#cmds,1 do
                                if i==1 then
                                    cm=cm..' '
                                else
                                    cm=cm..', '
                                end
                                cm=cm..cmds[i].func
                            end
                        end
                        local a=string.format(timeStr().." stopping %s publisher for client '%s' with topic '%s'. All Streaming functions on that topic will be unregistered%s",nn,clientId,topic,cm)
                        sim.addLog(sim.verbosity_msgs,a)
                    end
                end
            end
            return true
        elseif funcName=='stopSubscriber' then
            local topic=funcArgs[1]
            if dedicatedSubscribers[clientId] then
                if dedicatedSubscribers[clientId][topic] then
                    simB0.socketCleanup(dedicatedSubscribers[clientId][topic].handle)
                    simB0.subscriberDestroy(dedicatedSubscribers[clientId][topic].handle)
                    dedicatedSubscribers[clientId][topic]=nil
                    if modelData.debugLevel>=1 then
                        local a=string.format(timeStr().." stopping dedicated subscriber for client '%s' with topic '%s'",clientId,topic)
                        sim.addLog(sim.verbosity_msgs,a)
                    end
                end
            end
            return true
        end
    end
    
    return false
end

function defaultSubscriber_callback(msg)
    msg=messagePack.unpack(msg)
    local funcName=msg[1][1]
    local clientId=msg[1][2]
    local task=msg[1][4] -- 0=normal serviceCall, 1=received on default subscriber, 2=register streaming cmd on default publisher, 3=received on dedicated subscriber, 4=register streaming cmd on dedicated publisher
    local topic=msg[1][3]
    local funcArgs=msg[2]
    updateClientLastActivityTime(clientId)
    
    -- We simply want to execute the function and forget (no return)
    if not handlePublisherSetupFunctions(task,funcName,clientId,topic,funcArgs) then
        PCALL(_G[funcName],true,unpack(funcArgs))
        if modelData.debugLevel>=2 then
            local a=string.format(timeStr().." called function for client '%s': %s (default subscriber)",clientId,funcName)
            sim.addLog(sim.verbosity_msgs,a)
        end
    end
end    
    
    
function dedicatedSubscriber_callback(msg)
    msg=messagePack.unpack(msg)
    local funcName=msg[1][1]
    local clientId=msg[1][2]
    local topic=msg[1][3]
    local funcArgs=msg[2]
    updateClientLastActivityTime(clientId)
    -- We simply want to execute the function and forget (no return)
    PCALL(_G[funcName],true,unpack(funcArgs))
    if modelData.debugLevel>=2 then
        local a=string.format(timeStr().." called function for client '%s': %s (dedicated subscriber)",clientId,funcName)
        sim.addLog(sim.verbosity_msgs,a)
    end
end    

function onConfigNodeNameChanged(ui,id,newVal)
    if #newVal>2 then
        local newValue=''
        for i=1,#newVal,1 do
            local v=newVal:sub(i,i)
            if (v>='0' and v<='9') or (v>='a' and v<='z') or (v>='A' and v<='Z') or v=='_' or v=='-' then
                newValue=newValue..v
            else
                newValue=newValue..'_'
            end
        end
        configUiData.nodeName=newValue
    end
    simUI.setEditValue(configUiData.dlg,1,configUiData.nodeName)
end

function onConfigChannelNameChanged(ui,id,newVal)
    if #newVal>2 then
        local newValue=''
        for i=1,#newVal,1 do
            local v=newVal:sub(i,i)
            if (v>='0' and v<='9') or (v>='a' and v<='z') or (v>='A' and v<='Z') or v=='_' or v=='-' then
                newValue=newValue..v
            else
                newValue=newValue..'_'
            end
        end
        configUiData.channelName=newValue
    end
    simUI.setEditValue(configUiData.dlg,2,configUiData.channelName)
end

function onDebugLevelChanged(uiHandle,id,newIndex)
    configUiData.debugLevel=newIndex
    modelData.debugLevel=newIndex
    sim.writeCustomDataBlock(model,modelTag,sim.packTable(modelData))
end

function updateDebugLevelCombobox()
    local items={'none','basic','extended','full'}
    simUI.setComboboxItems(configUiData.dlg,3,items,modelData.debugLevel)
end


function onSimOnlyChanged(ui,id,newval)
    configUiData.duringSimulationOnly=not configUiData.duringSimulationOnly
    modelData.duringSimulationOnly=not modelData.duringSimulationOnly
    sim.writeCustomDataBlock(model,modelTag,sim.packTable(modelData))
    if modelData.duringSimulationOnly then
        destroyNode()
    else
        createNode()
    end
end

function onPackStrAsBinChanged(ui,id,newval)
    configUiData.packStrAsBin=not configUiData.packStrAsBin
    modelData.packStrAsBin=not modelData.packStrAsBin
    sim.writeCustomDataBlock(model,modelTag,sim.packTable(modelData))
    if modelData.packStrAsBin then
        messagePack.set_string('binary')
    else
        messagePack.set_string('string')
    end
end

function onConfigRestartNode(ui,id,newVal)
    modelData.nodeName=configUiData.nodeName
    modelData.channelName=configUiData.channelName
    sim.writeCustomDataBlock(model,modelTag,sim.packTable(modelData))
    if not modelData.duringSimulationOnly then
        destroyNode()
        createNode()
    end
end

function createConfigDlg()
    if simUI then
        if not configUiData then
            local xml = [[
            <ui title="BlueZero-based remote API, server-side configuration" closeable="false" resizable="false" activate="false">
            <group layout="form" flat="true">
            <label text="Node name"/>
            <edit on-editing-finished="onConfigNodeNameChanged" id="1"/>
            <label text="Channel name"/>
            <edit on-editing-finished="onConfigChannelNameChanged" id="2"/>
            <label text=""/>
            <button text="Restart node with above names" checked="false" on-click="onConfigRestartNode" />
            
            <label text="Pack strings as binary"/>
            <checkbox text="" on-change="onPackStrAsBinChanged" id="4" />
            <label text="Enabled during simulation only"/>
            <checkbox text="" on-change="onSimOnlyChanged" id="5" />
            <label text="Debug level"/>
            <combobox id="3" on-change="onDebugLevelChanged"></combobox>
            </group>
            </ui>
            ]]
            configUiData={}
            configUiData.dlg=simUI.create(xml)
            if previousConfigDlgPos then
                simUI.setPosition(configUiData.dlg,previousConfigDlgPos[1],previousConfigDlgPos[2],true)
            end
            configUiData.nodeName=modelData.nodeName
            configUiData.channelName=modelData.channelName
            configUiData.debugLevel=modelData.debugLevel
            configUiData.packStrAsBin=modelData.packStrAsBin
            configUiData.duringSimulationOnly=modelData.duringSimulationOnly
            simUI.setEditValue(configUiData.dlg,1,configUiData.nodeName)
            simUI.setEditValue(configUiData.dlg,2,configUiData.channelName)
            simUI.setCheckboxValue(configUiData.dlg,4,configUiData.packStrAsBin and 2 or 0)
            simUI.setCheckboxValue(configUiData.dlg,5,configUiData.duringSimulationOnly and 2 or 0)
            updateDebugLevelCombobox()
        end
    end
end

function removeConfigDlg()
    if simUI then
        if configUiData then
            local x,y=simUI.getPosition(configUiData.dlg)
            previousConfigDlgPos={x,y}
            simUI.destroy(configUiData.dlg)
            configUiData=nil
        end
    end
end

function sysCall_init()
    local res
    res,model=PCALL(sim.getObjectAssociatedWithScript,false,sim.handle_self) -- if call made directly, will fail with add-on script
    local abort=false
    if not res or model==-1 then
        -- We are running this script via an Add-On script
        
        model=-1
        modelData={nodeName='b0RemoteApi_CoppeliaSim-addOn',channelName='b0RemoteApiAddOn',debugLevel=1,packStrAsBin=false,duringSimulationOnly=false}
    else
        -- We are probably running this script via a customization script
        modelTag='b0-remoteApi'
--        sim.writeCustomDataBlock(model,modelTag,sim.packTable({nodeName='b0RemoteApi_CoppeliaSim',channelName='b0RemoteApi',debugLevel=1,packStrAsBin=false,duringSimulationOnly=false}))
        
        local objs=sim.getObjectsWithTag(modelTag,true)
        if #objs>1 then
            sim.removeModel(model)
            sim.removeObjectFromSelection(sim.handle_all)
            objs=sim.getObjectsWithTag(modelTag,true)
            sim.addObjectToSelection(sim.handle_single,objs[1])
            abort=true
        else
            modelData=sim.unpackTable(sim.readCustomDataBlock(model,modelTag))
        end
    end
    syncMode=false
    if not abort then
        createNode()
    end
end

function sysCall_cleanup()
    destroyNode()
    removeConfigDlg()
end

function sysCall_nonSimulation()
    local s=sim.getObjectSelection()
    if s and #s==1 and s[1]==model then
        createConfigDlg()
    else
        removeConfigDlg()
    end
    sendAndSpin(0)
end

function sysCall_beforeMainScript()
    if not sendAndSpin(1) then
        return {doNotRunMainScript=true}
    end
end

function sysCall_suspended()
    sendAndSpin(2)
end

function sysCall_beforeSimulation()
    removeConfigDlg()
    if modelData.duringSimulationOnly then
        createNode()
    end
end

function sysCall_afterSimulation()
    if modelData.duringSimulationOnly then
        destroyNode()
    end
    syncMode=false
end

function sysCall_beforeInstanceSwitch()
    if model>=0 then
        destroyNode()
        removeConfigDlg()
    end
end

function sysCall_afterInstanceSwitch()
    if model>=0 then
        if not modelData.duringSimulationOnly then
            createNode()
        end
        createNode()
    end
end

function sysCall_addOnScriptSuspend()
    destroyNode()
end

function sysCall_addOnScriptResume()
    if not modelData.duringSimulationOnly then
        createNode()
    end
end
