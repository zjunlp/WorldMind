local simUI={}

--@fun insertTableRow insert a row in a table widget
--@arg int ui the ui handle
--@arg int widget the widget identifier
--@arg int index the index (0-based) where the new row will appear
function simUI.insertTableRow(ui,widget,index)
    local rows=simUI.getRowCount(ui,widget)
    local cols=simUI.getColumnCount(ui,widget)
    simUI.setRowCount(ui,widget,rows+1)
    for row=rows-1,index+1,-1 do
        for col=0,cols-1 do
            simUI.setItem(ui,widget,row,col,simUI.getItem(ui,widget,row-1,col))
        end
    end
end

--@fun removeTableRow remove a row from a table widget
--@arg int ui the ui handle
--@arg int widget the widget identifier
--@arg int index the row index (0-based) to remove
function simUI.removeTableRow(ui,widget,index)
    local rows=simUI.getRowCount(ui,widget)
    local cols=simUI.getColumnCount(ui,widget)
    for row=index,rows-2 do
        for col=0,cols-1 do
            simUI.setItem(ui,widget,row,col,simUI.getItem(ui,widget,row+1,col))
        end
    end
    simUI.setRowCount(ui,widget,rows-1)
end

--@fun insertTableColumn insert a column in a table widget
--@arg int ui the ui handle
--@arg int widget the widget identifier
--@arg int index the index (0-based) where the new column will appear
function simUI.insertTableColumn(ui,widget,index)
    local rows=simUI.getRowCount(ui,widget)
    local cols=simUI.getColumnCount(ui,widget)
    simUI.setColumnCount(ui,widget,cols+1)
    for col=cols-1,index+1,-1 do
        for row=0,rows-1 do
            simUI.setItem(ui,widget,row,col,simUI.getItem(ui,widget,row,col-1))
        end
    end
end

--@fun removeTableColumn remove a column from a table widget
--@arg int ui the ui handle
--@arg int widget the widget identifier
--@arg int index the column index (0-based) to remove
function simUI.removeTableColumn(ui,widget,index)
    local rows=simUI.getRowCount(ui,widget)
    local cols=simUI.getColumnCount(ui,widget)
    for col=index,cols-2 do
        for row=0,rows-1 do
            simUI.setItem(ui,widget,row,col,simUI.getItem(ui,widget,row,col+1))
        end
    end
    simUI.setColumnCount(ui,widget,cols-1)
end

--@fun setScene3DNodeParam polymorphic version of the onSetScene3DNodeXXXParam() functions
--@arg int ui the ui handle
--@arg int widget the widget identifier
--@arg int nodeId the node id
--@arg string paramName the parameter name
--@arg anything paramValue the parameter value
function simUI.setScene3DNodeParam(ui,widget,nodeId,paramName,paramValue)
    if type(paramValue)=='number' then
        if math.floor(paramValue)==paramValue then
            simUI.setScene3DNodeIntParam(ui,widget,nodeId,paramName,paramValue)
        else
            simUI.setScene3DNodeFloatParam(ui,widget,nodeId,paramName,paramValue)
        end
    elseif type(paramValue)=='string' then
        simUI.setScene3DNodeStringParam(ui,widget,nodeId,paramName,paramValue)
    elseif type(paramValue)=='table' then
        if #paramValue==2 then
            simUI.setScene3DNodeParam(ui,widget,nodeId,paramName,paramValue[1],paramValue[2])
        elseif #paramValue==3 then
            simUI.setScene3DNodeParam(ui,widget,nodeId,paramName,paramValue[1],paramValue[2],paramValue[3])
        elseif #paramValue==4 then
            simUI.setScene3DNodeParam(ui,widget,nodeId,paramName,paramValue[1],paramValue[2],paramValue[3],paramValue[4])
        end
    else
        error(string.format('unsupported value type: %s', type(paramValue)))
    end
end

return simUI
