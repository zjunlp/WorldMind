model.robotPlot={}

function model.robotPlot.setData(dataFromRagnar,plotId)
    if model.robotPlot.ui then
        if plotId==1 then
            for i=1,4,1 do
                local label='axis'..i
                simUI.clearCurve(model.robotPlot.ui,1,label)
                if #dataFromRagnar.timeStamps>0 then
                    simUI.addCurveTimePoints(model.robotPlot.ui,1,label,dataFromRagnar.timeStamps,dataFromRagnar.motorAngles[i])
                end
            end
            simUI.rescaleAxesAll(model.robotPlot.ui,1,false,false)
            simUI.replot(model.robotPlot.ui,1)
        end
        if plotId==2 then
            for i=1,4,1 do
                local label='axis'..i
                simUI.clearCurve(model.robotPlot.ui,2,label)
                if #dataFromRagnar.timeStamps>0 then
                    simUI.addCurveTimePoints(model.robotPlot.ui,2,label,dataFromRagnar.timeStamps,dataFromRagnar.motorErrors[i])
                end
            end
            simUI.rescaleAxesAll(model.robotPlot.ui,2,false,false)
            simUI.replot(model.robotPlot.ui,2)
        end
        if plotId==3 then
            simUI.clearCurve(model.robotPlot.ui,3,'X')
            simUI.clearCurve(model.robotPlot.ui,3,'Y')
            simUI.clearCurve(model.robotPlot.ui,3,'Z')
            simUI.clearCurve(model.robotPlot.ui,3,'Rot')
            simUI.clearCurve(model.robotPlot.ui,3,'Gripper close')
            simUI.clearCurve(model.robotPlot.ui,3,'Gripper open')
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'X',dataFromRagnar.timeStamps,dataFromRagnar.platformPose[1])
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'Y',dataFromRagnar.timeStamps,dataFromRagnar.platformPose[2])
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'Z',dataFromRagnar.timeStamps,dataFromRagnar.platformPose[3])
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'Rot',dataFromRagnar.timeStamps,dataFromRagnar.platformPose[6])
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'Gripper close',dataFromRagnar.gripperClose.t,dataFromRagnar.gripperClose.v)
            simUI.addCurveTimePoints(model.robotPlot.ui,3,'Gripper open',dataFromRagnar.gripperOpen.t,dataFromRagnar.gripperOpen.v)
            simUI.rescaleAxesAll(model.robotPlot.ui,3,false,false)
 --           simUI.growPlotYRange(model.robotPlot.ui,3,min,max)
            simUI.replot(model.robotPlot.ui,3)
        end
    end
end

function model.robotPlot.startShowing()
    if savedJoints and not model.robotPlot.ui then
        local xml=[[<tabs id="77">
                <tab title="Axes angles">
                <plot id="1" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
                <tab title="Axes angular errors">
                <plot id="2" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
                <tab title="Platform position">
                <plot id="3" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                </tab>
            </tabs>]]
        local prevPos,prevSize=simBWF.readSessionPersistentObjectData(model.handle,"ragnarPlotPosAndSize"..model.simOrRealIndex)
        if not prevPos then
            prevPos="bottomRight"
        end
        model.robotPlot.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle),prevPos,true,"model.robotPlot.stopShowing_callback",false,true,false,nil,prevSize)
        simUI.setPlotLabels(model.robotPlot.ui,1,"time (seconds)","degrees")
        simUI.setPlotLabels(model.robotPlot.ui,2,"time (seconds)","degrees")
        simUI.setPlotLabels(model.robotPlot.ui,3,"time (seconds)","millimeters and degrees")
        if not model.robotPlot.tabIndex then
            model.robotPlot.tabIndex=0
        end
        simUI.setCurrentTab(model.robotPlot.ui,77,model.robotPlot.tabIndex,true)

        local curveStyle=simUI.curve_style.line
        local scatterShape={scatter_shape=simUI.curve_scatter_shape.none,scatter_size=5,line_size=1,add_to_legend=true,selectable=true,track=false}
        simUI.addCurve(model.robotPlot.ui,1,simUI.curve_type.time,'axis1',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,1,simUI.curve_type.time,'axis2',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,1,simUI.curve_type.time,'axis3',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,1,simUI.curve_type.time,'axis4',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(model.robotPlot.ui,1,true)
        simUI.setMouseOptions(model.robotPlot.ui,1,false,false,false,false)
        simUI.addCurve(model.robotPlot.ui,2,simUI.curve_type.time,'axis1',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,2,simUI.curve_type.time,'axis2',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,2,simUI.curve_type.time,'axis3',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,2,simUI.curve_type.time,'axis4',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(model.robotPlot.ui,2,true)
        simUI.setMouseOptions(model.robotPlot.ui,2,false,false,false,false)
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'X',{255,0,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'Y',{0,255,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'Z',{0,128,255},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'Rot',{255,255,0},curveStyle,scatterShape)
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'Gripper close',{255,0,255},simUI.curve_style.scatter,{scatter_shape=simUI.curve_scatter_shape.circle,scatter_size=10,line_size=1,add_to_legend=true,selectable=true,track=false})
        simUI.addCurve(model.robotPlot.ui,3,simUI.curve_type.time,'Gripper open',{0,255,255},simUI.curve_style.scatter,{scatter_shape=simUI.curve_scatter_shape.circle,scatter_size=10,line_size=1,add_to_legend=true,selectable=true,track=false})
        simUI.setLegendVisibility(model.robotPlot.ui,3,true)
        simUI.setMouseOptions(model.robotPlot.ui,3,false,false,false,false)
    end
end

function model.robotPlot.stopShowing_callback()
    model.robotPlot.wasClosed=true
    model.robotPlot.stopShowing()
end

function model.robotPlot.stopShowing()
    if model.robotPlot.ui then
        local x,y=simUI.getPosition(model.robotPlot.ui)
        local xs,ys=simUI.getSize(model.robotPlot.ui)
        simBWF.writeSessionPersistentObjectData(model.handle,"ragnarPlotPosAndSize"..model.simOrRealIndex,{x,y},{xs,ys})
        model.robotPlot.tabIndex=simUI.getCurrentTab(model.robotPlot.ui,77)
        simUI.destroy(model.robotPlot.ui)
        model.robotPlot.ui=nil
    end
end

