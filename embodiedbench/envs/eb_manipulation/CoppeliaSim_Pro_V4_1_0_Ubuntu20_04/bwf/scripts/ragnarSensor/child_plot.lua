model.plot={}

function model.plot.updateData(triggerData,plotId)
    if model.plot.ui then
        if plotId==1 then
            simUI.clearCurve(model.plot.ui,1,'l')
            simUI.clearCurve(model.plot.ui,1,'h')
            simUI.clearCurve(model.plot.ui,1,'trigger')
            local startT=0
            if triggerData.lastTime-10>0 then
                startT=triggerData.lastTime-10
            end
            local trig={}
            for i=1,#triggerData.triggerTimes,1 do
                trig[i]=4
            end
            simUI.addCurveTimePoints(model.plot.ui,1,'l',{startT,triggerData.lastTime},{0,0})
            simUI.addCurveTimePoints(model.plot.ui,1,'h',{startT,triggerData.lastTime},{4.5,4.5})
            if #trig>0 then
                simUI.addCurveTimePoints(model.plot.ui,1,'trigger',triggerData.triggerTimes,trig)
            end
            simUI.rescaleAxesAll(model.plot.ui,1,false,false)
            simUI.replot(model.plot.ui,1)
        end
    end
end

function model.plot.showPlot()
    if not model.plot.ui then
        local xml=[[
                <plot id="1" max-buffer-size="1000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150" y-ticks="false" y-tick-labels="false"/>
                ]]
        if not model.plot.previousPos then
            model.plot.previousPos="bottomRight"
        end
        model.plot.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle),model.plot.previousPos,true,"model.plot.closePlot_callback",false,true,false,nil,model.plot.previousSize)
        simUI.setPlotLabels(model.plot.ui,1,"time (seconds)","trigger state")

        local curveStyle=simUI.curve_style.line
        local scatterShape={scatter_shape=simUI.curve_scatter_shape.none,scatter_size=1,line_size=1,add_to_legend=false,selectable=false,track=false}
        simUI.addCurve(model.plot.ui,1,simUI.curve_type.time,'l',{255,255,255},curveStyle,scatterShape)
        simUI.addCurve(model.plot.ui,1,simUI.curve_type.time,'h',{64,64,64},curveStyle,scatterShape)
        simUI.addCurve(model.plot.ui,1,simUI.curve_type.time,'trigger',{255,255,0},simUI.curve_style.scatter,{scatter_shape=simUI.curve_scatter_shape.circle,scatter_size=10,line_size=1,add_to_legend=true,selectable=true,track=false})
        simUI.setLegendVisibility(model.plot.ui,1,true)
 --       simUI.YLabel(model.plot.ui,1,''
        simUI.setMouseOptions(model.plot.ui,1,false,false,false,false)
    end
end

function model.plot.closePlot_callback()
    model.plot.wasClosed=true
    model.plot.closePlot()
end

function model.plot.closePlot()
    if model.plot.ui then
        local x,y=simUI.getPosition(model.plot.ui)
        model.plot.previousPos={x,y}
        local x,y=simUI.getSize(model.plot.ui)
        model.plot.previousSize={x,y}
        simUI.destroy(model.plot.ui)
        model.plot.ui=nil
    end
end
