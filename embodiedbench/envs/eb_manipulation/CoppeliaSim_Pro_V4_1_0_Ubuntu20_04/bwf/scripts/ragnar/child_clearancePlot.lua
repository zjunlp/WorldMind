model.clearancePlot={}

function model.clearancePlot.setData(clearanceData,plotId)
    if model.clearancePlot.ui then
        if plotId==1 then
            simUI.clearCurve(model.clearancePlot.ui,1,'Clearance')
            if #clearanceData.times>0 then
                simUI.addCurveTimePoints(model.clearancePlot.ui,1,'Clearance',clearanceData.times,clearanceData.clearances)
            end
            simUI.rescaleAxesAll(model.clearancePlot.ui,1,false,false)
            simUI.replot(model.clearancePlot.ui,1)
        end
    end
end

function model.clearancePlot.startShowing()
    if not model.clearancePlot.ui then
        local xml=[[
                <plot id="1" max-buffer-size="100000" cyclic-buffer="false" background-color="25,25,25" foreground-color="150,150,150"/>
                ]]
        local prevPos,prevSize=simBWF.readSessionPersistentObjectData(model.handle,"ragnarClearancePlotPosAndSize")
        if not prevPos then
            prevPos="bottomRight"
        end
        model.clearancePlot.ui=simBWF.createCustomUi(xml,simBWF.getObjectAltName(model.handle).." Clearance",prevPos,true,"model.clearancePlot.stopShowing_callback",false,true,false,nil,prevSize)
        simUI.setPlotLabels(model.clearancePlot.ui,1,"time (seconds)","meters")

        local curveStyle=simUI.curve_style.line
        local scatterShape={scatter_shape=simUI.curve_scatter_shape.none,scatter_size=5,line_size=1,add_to_legend=true,selectable=true,track=false}
        simUI.addCurve(model.clearancePlot.ui,1,simUI.curve_type.time,'Clearance',{255,255,0},curveStyle,scatterShape)
        simUI.setLegendVisibility(model.clearancePlot.ui,1,true)
        simUI.setMouseOptions(model.clearancePlot.ui,1,false,false,false,false)
    end
end

function model.clearancePlot.stopShowing_callback()
    model.clearancePlot.wasClosed=true
    model.clearancePlot.stopShowing()
end

function model.clearancePlot.stopShowing()
    if model.clearancePlot.ui then
        local x,y=simUI.getPosition(model.clearancePlot.ui)
        local xs,ys=simUI.getSize(model.clearancePlot.ui)
        simBWF.writeSessionPersistentObjectData(model.handle,"ragnarClearancePlotPosAndSize",{x,y},{xs,ys})
        simUI.destroy(model.clearancePlot.ui)
        model.clearancePlot.ui=nil
    end
end
