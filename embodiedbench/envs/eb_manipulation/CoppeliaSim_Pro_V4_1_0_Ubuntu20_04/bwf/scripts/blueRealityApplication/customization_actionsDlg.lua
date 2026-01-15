model.actions={}

function model.actions.saveJsonClick_callback()
    if variousActionDlgId then
        simUI.destroy(variousActionDlgId)
        variousActionDlgId=nil
    end
    local pathAndName=sim.fileDialog(sim.filedlg_type_save,"JSON File Save","","","JSON file","JSON")
--    pathAndName ="CurrentScene.JSON"
    if pathAndName and #pathAndName>0 then
        local data={}
        data.fileName=pathAndName
        simBWF.query('make_JSON_file',data)
    end
end

-------------------------------------------------------
-------------------------------------------------------

function model.actions.quoteRequest_executeIfNeeded()
    if quoteRequest then
        if quoteRequest.counter>0 then
            quoteRequest.counter=quoteRequest.counter-1
        else
            local res,code,response_status_line,data=model.actions.sendRequest(quoteRequest.payload,'http://brcommunicator.azurewebsites.net/api/quote')
            if res and code==200 then
                local filename='Quote_'..os.date()..'.docx'
                filename=string.gsub(filename,":","_")
                filename=string.gsub(filename," ","_")
                filename=sim.fileDialog(sim.filedlg_type_save,'save quote...','',filename,'MS Doument','docx')
                if filename then
                    local f = assert(io.open(filename, 'wb')) -- open in "binary" mode
                    f:write(data)
                    f:close()
                    simBWF.openFile(filename)
                end
            else
                -- code contains the error msg if res is nil. Otherwise, it contains a status code
                local msg="Failed to retrieve the quote information.\n"
                if not res then
                    msg=msg.."Status code is: "..code
                else
                    msg=msg.."Error message is: "..res
                end
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"Quote inquiry",msg)
            end
            simUI.destroy(quoteRequest.ui)
            quoteRequest=nil
        end
    end
end

function model.actions.userIdPricing_callback(uiHandle,id,newValue)
    remoteRequestUi.userId=newValue
    simUI.setEditValue(remoteRequestUi.ui,1,remoteRequestUi.userId)
end

function model.actions.powerSupplyComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.power=powerPricing_comboboxItems[newIndex+1][1]
end

function model.actions.connectorTypeComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.connector=connectorType_comboboxItems[newIndex+1][1]
end

function model.actions.valveTypeComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.valve=valveType_comboboxItems[newIndex+1][1]
end

function model.actions.customGrippersPricing_callback(uiHandle,id,newValue)
    local numGrippers=tonumber(newValue)
    if numGrippers then
        if numGrippers<20 then
            remoteRequestUi.numGrippers=newValue
        else
            remoteRequestUi.numGrippers=20
        end
    end
    simUI.setEditValue(remoteRequestUi.ui,18,simBWF.format('%d',remoteRequestUi.numGrippers))
end

function model.actions.frameTypeComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.frame=frameType_comboboxItems[newIndex+1][1]
end

function model.actions.frameLifterCheckChangePricing_callback(uiHandle,id,newValue)
    if newValue~=0 then
        remoteRequestUi.frameLifter=true
    else
        remoteRequestUi.frameLifter=false
    end
end

function model.actions.lineControlComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.lineControl=lineControl_comboboxItems[newIndex+1][1]
end

function model.actions.brdLicenseCheckChangePricing_callback(uiHandle,id,newValue)
    if newValue~=0 then
        remoteRequestUi.brdLicense=true
    else
        remoteRequestUi.brdLicense=false
    end
end

function model.actions.broLicenseComboChangePricing_callback(uiHandle,id,newValue)
    local numGrippers=tonumber(newValue)
    if numGrippers then
        if numGrippers<20 then
            remoteRequestUi.broLicense=newValue
        else
            remoteRequestUi.broLicense=20
        end
    end
    simUI.setEditValue(remoteRequestUi.ui,16,simBWF.format('%d',remoteRequestUi.broLicense))
end

function model.actions.oeeLicenseCheckChangePricing_callback(uiHandle,id,newValue)
    if newValue~=0 then
        remoteRequestUi.oeeLicense=true
    else
        remoteRequestUi.oeeLicense=false
    end
end

function model.actions.shippingComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.shipping=shipping_comboboxItems[newIndex+1][1]
end

function model.actions.destinationComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.destination=destination_comboboxItems[newIndex+1][1]
end

function model.actions.documentationComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.documentation=documentation_comboboxItems[newIndex+1][1]
end

function model.actions.quoteTypeComboChangePricing_callback(uiHandle,id,newIndex)
    remoteRequestUi.quote=quote_comboboxItems[newIndex+1][1]
end

function model.actions.cancelRequestDlg_callback()
    simUI.destroy(remoteRequestUi.ui)
    remoteRequestUi.ui=nil
end

function model.actions.okQuote_callback()
    simUI.destroy(remoteRequestUi.ui)
    remoteRequestUi.ui=nil

    local data=model.actions.getSceneContentData()
    data.sceneConfig.sceneImage=model.actions.getSceneScreenShot(false)
    data.sceneConfig.sceneImageFromTop=model.actions.getSceneScreenShot(true)

    quoteRequest={}
    quoteRequest.payload=model.json.encode(data,{indent=true})
--    quoteRequest.requestAuxConsole=sim.auxiliaryConsoleOpen('Quote request',500,4,{100,100},{800,800},nil,{1,0.95,0.95})
--    sim.auxiliaryConsolePrint(quoteRequest.requestAuxConsole,quoteRequest.payload)

    local xml =[[
            <label text="Please wait a few seconds..."  style="* {qproperty-alignment: AlignCenter; min-width: 300px; min-height: 100px;}"/>
    ]]
    quoteRequest.ui=simBWF.createCustomUi(xml,'Quote request','center',false,nil,true,false,false)
    quoteRequest.counter=3
end

-------------------------------------------------------
-------------------------------------------------------

function model.actions.roiRequest_executeIfNeeded()
    if roiRequest then
        if roiRequest.counter>0 then
            roiRequest.counter=roiRequest.counter-1
        else
            local res,code,response_status_line,data=model.actions.sendRequest(roiRequest.payload,'http://brcommunicator.azurewebsites.net/api/roi')
            if res and code==200 then
                local filename='ROI_'..os.date()..'.xlsx'
                filename=string.gsub(filename,":","_")
                filename=string.gsub(filename," ","_")
                local f = assert(io.open(filename, 'wb')) -- open in "binary" mode
                f:write(data)
                f:close()
                simBWF.openFile(filename)
            else
                -- code contains the error msg if res is nil. Otherwise, it contains a status code
                local msg="Failed to retrieve the ROI information.\n"
                if not res then
                    msg=msg.."Status code is: "..code
                else
                    msg=msg.."Error message is: "..res
                end
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"ROI inquiry",msg)
            end
            simUI.destroy(roiRequest.ui)
            roiRequest=nil
        end
    end
end

function model.actions.closeROI_callback()
    simUI.destroy(roiInfo.ui)
    roiInfo.ui=nil
end

function model.actions.workersCurrentROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.current.number_of_workers=math.floor(newValue*10)/10
    end
    model.actions.refreshRoiDlg()
end

function model.actions.hourlyCostCurrentROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.current.burdened_hourly_cost=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.outputRateCurrentROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.current.output_rate_parts_per_minute=math.floor(newValue)
    end
    model.actions.refreshRoiDlg()
end

function model.actions.failureRateCurrentROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100 then newValue=100 end
        roiInfo.current.quality_failure_rate=math.floor(newValue)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.workersBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.bwf.number_of_workers=math.floor(newValue*10)/10
    end
    model.actions.refreshRoiDlg()
end

function model.actions.hourlyCostBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.bwf.burdened_hourly_cost=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.outputRateBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.bwf.output_rate_parts_per_minute=math.floor(newValue)
    end
    model.actions.refreshRoiDlg()
end

function model.actions.failureRateBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100 then newValue=100 end
        roiInfo.bwf.quality_failure_rate=math.floor(newValue)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.shiftsPerDayBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>5 then newValue=5 end
        roiInfo.bwf.shifts_per_day=math.floor(newValue)
    end
    model.actions.refreshRoiDlg()
end

function model.actions.hoursPerShiftBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>12 then newValue=12 end
        roiInfo.bwf.hours_per_shift=math.floor(newValue*10)/10
    end
    model.actions.refreshRoiDlg()
end

function model.actions.prodDaysPerYearBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>365 then newValue=365 end
        roiInfo.bwf.production_days_per_year=math.floor(newValue)
    end
    model.actions.refreshRoiDlg()
end

function model.actions.reworkCostBwfROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>1000 then newValue=1000 end
        roiInfo.bwf.cost_to_rework_quality_failures=math.floor(newValue*1000)/1000
    end
    model.actions.refreshRoiDlg()
end

function model.actions.dualGripperCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>10000 then newValue=10000 end
--        roiInfo.unit_cost_1=math.floor(newValue*100)/100
        roiInfo.dual_gripper_cost=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.lipBaseCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>10000 then newValue=10000 end
--        roiInfo.unit_cost_2=math.floor(newValue*100)/100
        roiInfo.lip_base_cost=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.swivelAdaptorCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>10000 then newValue=10000 end
--        roiInfo.unit_cost_3=math.floor(newValue*100)/100
        roiInfo.swivel_adaptor_cost=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.depreciationROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>20 then newValue=20 end
        roiInfo.depreciation=math.floor(newValue*10)/10
    end
    model.actions.refreshRoiDlg()
end

function model.actions.financingCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100 then newValue=100 end
        roiInfo.financing_cost=math.floor(newValue)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.otherEquipmentCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>10000000 then newValue=10000000 end
        roiInfo.cost_of_other_equipment=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.shippingCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100000 then newValue=100000 end
        roiInfo.shipping_and_installation=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.sparePartsCostROI_callback(uiHandle,id,newValue)
    newValue=tonumber(newValue)
    if newValue then
        if newValue<0 then newValue=0 end
        if newValue>100000 then newValue=100000 end
        roiInfo.spare_parts_purchase=math.floor(newValue*100)/100
    end
    model.actions.refreshRoiDlg()
end

function model.actions.refreshRoiDlg()
    local sel=simBWF.getSelectedEditWidget(roiInfo.ui)
    simUI.setEditValue(roiInfo.ui,1,simBWF.format("%.1f",roiInfo.current.number_of_workers))
    simUI.setEditValue(roiInfo.ui,2,simBWF.format("%.2f",roiInfo.current.burdened_hourly_cost))
    simUI.setEditValue(roiInfo.ui,3,simBWF.format("%.0f",roiInfo.current.output_rate_parts_per_minute))
    simUI.setEditValue(roiInfo.ui,4,simBWF.format("%.0f",roiInfo.current.quality_failure_rate*100))

    simUI.setEditValue(roiInfo.ui,5,simBWF.format("%.1f",roiInfo.bwf.number_of_workers))
    simUI.setEditValue(roiInfo.ui,6,simBWF.format("%.2f",roiInfo.bwf.burdened_hourly_cost))
    simUI.setEditValue(roiInfo.ui,7,simBWF.format("%.0f",roiInfo.bwf.output_rate_parts_per_minute))
    simUI.setEditValue(roiInfo.ui,8,simBWF.format("%.0f",roiInfo.bwf.quality_failure_rate*100))
    simUI.setEditValue(roiInfo.ui,9,simBWF.format("%.0f",roiInfo.bwf.shifts_per_day))
    simUI.setEditValue(roiInfo.ui,10,simBWF.format("%.1f",roiInfo.bwf.hours_per_shift))
    simUI.setEditValue(roiInfo.ui,11,simBWF.format("%.0f",roiInfo.bwf.production_days_per_year))
    simUI.setEditValue(roiInfo.ui,12,simBWF.format("%.3f",roiInfo.bwf.cost_to_rework_quality_failures))

    simUI.setEditValue(roiInfo.ui,13,simBWF.format("%.2f",roiInfo.dual_gripper_cost))
    simUI.setEditValue(roiInfo.ui,14,simBWF.format("%.2f",roiInfo.lip_base_cost))
    simUI.setEditValue(roiInfo.ui,15,simBWF.format("%.2f",roiInfo.swivel_adaptor_cost))


    simUI.setEditValue(roiInfo.ui,16,simBWF.format("%.1f",roiInfo.depreciation))
    simUI.setEditValue(roiInfo.ui,17,simBWF.format("%.0f",roiInfo.financing_cost*100))
    simUI.setEditValue(roiInfo.ui,19,simBWF.format("%.2f",roiInfo.cost_of_other_equipment))
    simUI.setEditValue(roiInfo.ui,20,simBWF.format("%.2f",roiInfo.shipping_and_installation))
    simUI.setEditValue(roiInfo.ui,21,simBWF.format("%.2f",roiInfo.spare_parts_purchase))

    simBWF.setSelectedEditWidget(roiInfo.ui,sel)
end

function model.actions.generateDefaultRoiInfoIfNeeded()
    roiInfo={}
    roiInfo.current={}
    roiInfo.current.number_of_workers=4
    roiInfo.current.burdened_hourly_cost=17
    roiInfo.current.output_rate_parts_per_minute=330
    roiInfo.current.quality_failure_rate=0

    roiInfo.bwf={}
    roiInfo.bwf.number_of_workers=0.5
    roiInfo.bwf.burdened_hourly_cost=17
    roiInfo.bwf.output_rate_parts_per_minute=330
    roiInfo.bwf.quality_failure_rate=0

    roiInfo.bwf.shifts_per_day=3
    roiInfo.bwf.hours_per_shift=8
    roiInfo.bwf.production_days_per_year=300
    roiInfo.bwf.cost_to_rework_quality_failures=0.05

    roiInfo.dual_gripper_cost=435
    roiInfo.lip_base_cost=800
    roiInfo.swivel_adaptor_cost=0

    roiInfo.depreciation=7
    roiInfo.financing_cost=0.05
    roiInfo.cost_of_other_equipment=20000
    roiInfo.shipping_and_installation=10000
    roiInfo.spare_parts_purchase=5000
end

function model.actions.roiSettingsDlg()
    local xml =[[
    <tabs id="78">
    <tab title="Production environment">
            <group layout="form" flat="false">
                <label text="Current" style="* {font-weight: bold; min-width: 250px;}"/>  <label style="* {max-width: 50px;}"/>

                <label text="Number of workers"/>
                <edit on-editing-finished="model.actions.workersCurrentROI_callback" id="1"/>

                <label text="Burdened hourly cost"/>
                <edit on-editing-finished="model.actions.hourlyCostCurrentROI_callback" id="2"/>

                <label text="Output rate (parts per min.)"/>
                <edit on-editing-finished="model.actions.outputRateCurrentROI_callback" id="3"/>

                <label text="Quality failure rate (%)"/>
                <edit on-editing-finished="model.actions.failureRateCurrentROI_callback" id="4"/>
            </group>

            <group layout="form" flat="false">
                <label text="With BWF" style="* {font-weight: bold; min-width: 250px;}"/>  <label style="* {max-width: 50px;}"/>

                <label text="Number of workers"/>
                <edit on-editing-finished="model.actions.workersBwfROI_callback" id="5"/>

                <label text="Burdened hourly cost"/>
                <edit on-editing-finished="model.actions.hourlyCostBwfROI_callback" id="6"/>

                <label text="Output rate (parts per min.)"/>
                <edit on-editing-finished="model.actions.outputRateBwfROI_callback" id="7"/>

                <label text="Quality failure rate (%)"/>
                <edit on-editing-finished="model.actions.failureRateBwfROI_callback" id="8"/>

                <label text="Shifts per day"/>
                <edit on-editing-finished="model.actions.shiftsPerDayBwfROI_callback" id="9"/>

                <label text="Hours per shift"/>
                <edit on-editing-finished="model.actions.hoursPerShiftBwfROI_callback" id="10"/>

                <label text="Production days per year"/>
                <edit on-editing-finished="model.actions.prodDaysPerYearBwfROI_callback" id="11"/>

                <label text="Cost to rework quality failures"/>
                <edit on-editing-finished="model.actions.reworkCostBwfROI_callback" id="12"/>
            </group>
    </tab>
    <tab title="Other">
            <group layout="form" flat="false">
                <label text="Consumables" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Dual gripper cost"/>
                <edit on-editing-finished="model.actions.dualGripperCostROI_callback" id="13"/>

                <label text="Lip base cost"/>
                <edit on-editing-finished="model.actions.lipBaseCostROI_callback" id="14"/>

                <label text="Swivel adapter cost"/>
                <edit on-editing-finished="model.actions.swivelAdaptorCostROI_callback" id="15"/>
            </group>

            <group layout="form" flat="false">
                <label text="Other" style="* {font-weight: bold;}"/>  <label text=""/>

                <label text="Depreciation in life of (years)"/>
                <edit on-editing-finished="model.actions.depreciationROI_callback" id="16"/>

                <label text="Financing cost (%)"/>
                <edit on-editing-finished="model.actions.financingCostROI_callback" id="17"/>

                <label text="Cost of other equipment"/>
                <edit on-editing-finished="model.actions.otherEquipmentCostROI_callback" id="19"/>

                <label text="Cost of ship. & installation"/>
                <edit on-editing-finished="model.actions.shippingCostROI_callback" id="20"/>

                <label text="Cost of spare parts"/>
                <edit on-editing-finished="model.actions.sparePartsCostROI_callback" id="21"/>
            </group>
    </tab>
    </tabs>
    <button text="Close" on-click="model.actions.closeROI_callback" />
    ]]
    if not roiInfo then
        model.actions.generateDefaultRoiInfoIfNeeded()
    end
    roiInfo.ui=simBWF.createCustomUi(xml,'ROI Calculation Request','center',false,nil,true,false,true)
    model.actions.refreshRoiDlg()
end

function model.actions.okRoi_callback()
    simUI.destroy(remoteRequestUi.ui)
    remoteRequestUi.ui=nil

    local data=model.actions.getSceneContentData()
    for key,value in pairs(roiInfo) do
        data[key]=value
    end

    roiRequest={}
    roiRequest.payload=model.json.encode(data,{indent=true})
--    roiRequest.requestAuxConsole=sim.auxiliaryConsoleOpen('ROI request',500,4,{100,100},{800,800},nil,{1,0.95,0.95})
--    sim.auxiliaryConsolePrint(roiRequest.requestAuxConsole,roiRequest.payload)

    local xml =[[
            <label text="Please wait a few seconds..."  style="* {qproperty-alignment: AlignCenter; min-width: 300px; min-height: 100px;}"/>
    ]]
    roiRequest.ui=simBWF.createCustomUi(xml,'ROI request','center',false,nil,true,false,false)
    roiRequest.counter=3
end


-------------------------------------------------------
-------------------------------------------------------

function model.actions.sopRequest_executeIfNeeded()
    if sopRequest then
        if sopRequest.counter>0 then
            sopRequest.counter=sopRequest.counter-1
        else
            local res,code,response_status_line,data=model.actions.sendRequest(sopRequest.payload,'http://brcommunicator.azurewebsites.net/api/sop')
            if res and code==200 then
                local sopConsole=sim.auxiliaryConsoleOpen('Production order',500,4,{100,100},{1000,400},nil,{1,1,1})
                local sopData=model.json.decode(data)
                for i=1,#sopData/2,1 do
                    local txt='Ragnar '..i..'\n'
                    txt=txt..'    Serial: '..sopData[2*(i-1)+1]..'\n'
                    txt=txt..'    QR code URL: '..sopData[2*(i-1)+2]..'\n'
                    sim.auxiliaryConsolePrint(sopConsole,txt)
                end
            else
                -- code contains the error msg if res is nil. Otherwise, it contains a status code
                local msg="Failed to retrieve the SOP information.\n"
                if not res then
                    msg=msg.."Status code is: "..code
                else
                    msg=msg.."Error message is: "..res
                end
                sim.msgBox(sim.msgbox_type_warning,sim.msgbox_buttons_ok,"SOP inquiry",msg)
            end
            simUI.destroy(sopRequest.ui)
            sopRequest=nil
        end
    end
end

function model.actions.okSop_callback()
    simUI.destroy(remoteRequestUi.ui)
    remoteRequestUi.ui=nil

    local data=model.actions.getSceneContentData()

    sopRequest={}
    sopRequest.payload=model.json.encode(data,{indent=true})
--    sopRequest.requestAuxConsole=sim.auxiliaryConsoleOpen('SOP request',500,4,{100,100},{800,800},nil,{1,0.95,0.95})
--    sim.auxiliaryConsolePrint(sopRequest.requestAuxConsole,sopRequest.payload)

    local xml =[[
            <label text="Please wait a few seconds..."  style="* {qproperty-alignment: AlignCenter; min-width: 300px; min-height: 100px;}"/>
    ]]
    sopRequest.ui=simBWF.createCustomUi(xml,'SOP request','center',false,nil,true,false,false)
    sopRequest.counter=3
end

-------------------------------------------------------
-------------------------------------------------------

function model.actions.sendRequest(payload,path)
    local response_body = { }
    model.http.TIMEOUT=60 -- default is 60

    local res, code, response_headers, response_status_line = model.http.request
    {
        url = path,
        method = "POST",
        headers =
        {
          ["Content-Type"] = "application/json",
          ["Content-Length"] = payload:len()
        },
        source = model.ltn12.source.string(payload),
        sink = model.ltn12.sink.table(response_body)
    }
    return res,code,response_status_line,table.concat(response_body)
end

function model.actions.getSceneScreenShot(fromTop)
    local rgb=nil
    local res= {1280,780} -- {640,480}
    if not fromTop then
        if sim.getObjectHandle('DefaultCamera@silentError') ~=-1 then
            camera = sim.getObjectHandle('DefaultCamera@silentError')
        else
            camera = sim.getObjectHandle('Camera@silentError')
        end
        local vs=sim.createVisionSensor(1+2+128,{res[1],res[2],0,0},{0.1,50,60*math.pi/180,0.1,0.1,0.1,255,255,255,0,0})
        sim.setObjectOrientation(vs,camera,{0,0,0})
        sim.setObjectPosition(vs,camera,{0,0,0})
        sim.handleVisionSensor(vs)
        rgb=sim.getVisionSensorCharImage(vs)
        sim.removeObject(vs)
    end
    if fromTop then
        local fl=sim.getObjectHandle('Floor')
        local prop=sim.getModelProperty(fl)
        sim.setModelProperty(fl,sim.modelproperty_not_visible)
        local parentless=sim.getObjectsInTree(sim.handle_scene,sim.handle_all,2)
        local minMaxX={9999,-9999}
        local minMaxY={9999,-9999}
        for po=1,#parentless,1 do
            local p=sim.getModelProperty(parentless[po])
            local isVisibleModel=sim.boolAnd32(p,sim.modelproperty_not_model+sim.modelproperty_not_visible)==0
            if isVisibleModel then
                local shapes=sim.getObjectsInTree(parentless[po],sim.object_shape_type,0)
                for sc=1,#shapes,1 do
                    local sp=sim.getObjectSpecialProperty(shapes[sc])
                    if sim.boolAnd32(sp,sim.objectspecialproperty_renderable)>0 then
                        local vertices=sim.getShapeMesh(shapes[sc])
                        local m=sim.getObjectMatrix(shapes[sc],-1)
                        for i=0,#vertices/3-1,1 do
                            local v={vertices[3*i+1],vertices[3*i+2],vertices[3*i+3]}
                            v=sim.multiplyVector(m,v)
                            if minMaxX[1]>v[1] then
                                minMaxX[1]=v[1]
                            end
                            if minMaxX[2]<v[1] then
                                minMaxX[2]=v[1]
                            end
                            if minMaxY[1]>v[2] then
                                minMaxY[1]=v[2]
                            end
                            if minMaxY[2]<v[2] then
                                minMaxY[2]=v[2]
                            end
                        end
                    end
                end
            end
        end
        local extX=minMaxX[2]-minMaxX[1]
        local extY=(minMaxY[2]-minMaxY[1])*1/0.75

        local vs=sim.createVisionSensor(1+128,{res[1],res[2],0,0},{0.1,10,math.max(extX,extY),0.1,0.1,0.1,255,255,255,0,0})
        sim.setObjectOrientation(vs,-1,{180*math.pi/180,0,0})
        sim.setObjectPosition(vs,-1,{(minMaxX[1]+minMaxX[2])/2,(minMaxY[1]+minMaxY[2])/2,5})
        sim.handleVisionSensor(vs)
        sim.setModelProperty(fl,prop)
        rgb=sim.getVisionSensorCharImage(vs)
        sim.removeObject(vs)
    end
    if rgb then
        local pngData=sim.saveImage(rgb,res,0,'.png',-1)
        pngData=sim.transformBuffer(pngData,sim.buffer_uint8,1,0,sim.buffer_base64)
        return pngData
    end
end

function model.actions.getSceneContentData()
    local objects={}
    objects.robots={}
    objects.visionSystems={}
    objects.conveyors={}
    local tagsAndCategories={{simBWF.modelTags.RAGNAR,objects.robots},{simBWF.modelTags.VISIONWINDOW,objects.visionSystems},{simBWF.modelTags.CONVEYOR,objects.conveyors}}
    for i=1,#tagsAndCategories,1 do
        local objs=sim.getObjectsWithTag(tagsAndCategories[i][1],true)
        for j=1,#objs,1 do
            local ob=simBWF.callCustomizationScriptFunction('model.ext.getItemData_pricing',objs[j])
            tagsAndCategories[i][2][#tagsAndCategories[i][2]+1]=ob
        end
    end
    objects.sceneConfig={}
    objects.sceneConfig.type='user_input'
    objects.sceneConfig.client_id=remoteRequestUi.userId
    objects.sceneConfig.projectName=sim.getStringParameter(sim.stringparam_scene_name)
    objects.sceneConfig.sceneSerializationNo=sim.getStringParameter(sim.stringparam_scene_unique_id)
    objects.sceneConfig.power_supply=remoteRequestUi.power
    objects.sceneConfig.connector_type=remoteRequestUi.connector
    objects.sceneConfig.value_type=remoteRequestUi.valve
    objects.sceneConfig.num_grippers=remoteRequestUi.numGrippers
    objects.sceneConfig.frame_type=remoteRequestUi.frame
    objects.sceneConfig.frame_lifter=remoteRequestUi.frameLifter
    objects.sceneConfig.conveyor_control="sensor_based"
    objects.sceneConfig.line_control=remoteRequestUi.lineControl
    objects.sceneConfig.brd_license=remoteRequestUi.brdLicense
    objects.sceneConfig.bro_license=remoteRequestUi.broLicense
    objects.sceneConfig.oee_license=remoteRequestUi.oeeLicense
    objects.sceneConfig.shipping_type=remoteRequestUi.shipping
    objects.sceneConfig.shipping_destination=remoteRequestUi.destination
    objects.sceneConfig.robot_documentation=remoteRequestUi.documentation
    objects.sceneConfig.quote_type=remoteRequestUi.quote
    return objects
end

function model.actions.remoteRequestDlg(requestType) -- 0=generate quote, 1=compute roi, 2=generate SOP
    if variousActionDlgId then
        simUI.destroy(variousActionDlgId)
        variousActionDlgId=nil
    end

    model.actions.generateDefaultRoiInfoIfNeeded()
    local xml =[[
        <group layout="form" flat="true">
            <label text="Client ID"/>
            <edit on-editing-finished="model.actions.userIdPricing_callback" id="1"/>

            <label text="Power supply"/>
            <combobox id="5" on-change="model.actions.powerSupplyComboChangePricing_callback"></combobox>

            <label text="Connector type"/>
            <combobox id="6" on-change="model.actions.connectorTypeComboChangePricing_callback"></combobox>

            <label text="Valve type"/>
            <combobox id="11" on-change="model.actions.valveTypeComboChangePricing_callback"></combobox>

            <label text="Number of custom grippers"/>
            <edit on-editing-finished="model.actions.customGrippersPricing_callback" id="18"/>

            <label text="Frame type"/>
            <combobox id="12" on-change="model.actions.frameTypeComboChangePricing_callback"></combobox>

            <label text="Frame lifter"/>
            <checkbox text="" on-change="model.actions.frameLifterCheckChangePricing_callback" id="13" />

            <label text="Line control"/>
            <combobox id="7" on-change="model.actions.lineControlComboChangePricing_callback"></combobox>

            <label text="Blue REALITY Designer license"/>
            <checkbox text="" on-change="model.actions.brdLicenseCheckChangePricing_callback" id="15" />

            <label text="Blue REALITY Operator license(s)"/>
            <edit on-editing-finished="model.actions.broLicenseComboChangePricing_callback" id="16"/>

            <label text="OEE license"/>
            <checkbox text="" on-change="model.actions.oeeLicenseCheckChangePricing_callback" id="17" />

            <label text="Shipping"/>
            <combobox id="8" on-change="model.actions.shippingComboChangePricing_callback"></combobox>

            <label text="Destination"/>
            <combobox id="9" on-change="model.actions.destinationComboChangePricing_callback"></combobox>

            <label text="Documentation"/>
            <combobox id="10" on-change="model.actions.documentationComboChangePricing_callback"></combobox>

            <label text="Quote type"/>
            <combobox id="14" on-change="model.actions.quoteTypeComboChangePricing_callback"></combobox> ]]

            if requestType==1 then
                xml=xml..[[
                    <label text="ROI input"/>
                    <button text="Edit" on-click="model.actions.roiSettingsDlg"/>
                    ]]
            end
            xml=xml..[[
                </group>
                    <group layout="form" flat="true">
                    <button text="Cancel" on-click="model.actions.cancelRequestDlg_callback" id="3" />
                    ]]
            if requestType==0 then
                xml=xml..'<button text="OK" on-click="model.actions.okQuote_callback" id="4" />'
            end
            if requestType==1 then
                xml=xml..'<button text="OK" on-click="model.actions.okRoi_callback" id="4" />'
            end
            if requestType==2 then
                xml=xml..'<button text="OK" on-click="model.actions.okSop_callback" id="4" />'
            end
            xml=xml..'</group>'

    if not remoteRequestUi then
        remoteRequestUi={}
        remoteRequestUi.userId='001'
        remoteRequestUi.power='220V'
        remoteRequestUi.connector='Phoenix'
        remoteRequestUi.valve='SMC'
        remoteRequestUi.numGrippers=0
        remoteRequestUi.frame='Linear'
        remoteRequestUi.frameLifter=false
        remoteRequestUi.lineControl='upstream'
        remoteRequestUi.brdLicense=true
        remoteRequestUi.broLicense=0
        remoteRequestUi.oeeLicense=true
        remoteRequestUi.shipping='air'
        remoteRequestUi.destination='Europe'
        remoteRequestUi.documentation='English'
        remoteRequestUi.quote='Compact'
    end
    if requestType==0 then
        remoteRequestUi.ui=simBWF.createCustomUi(xml,'Generate quote','center',false,nil,true,false,true)
    end
    if requestType==1 then
        remoteRequestUi.ui=simBWF.createCustomUi(xml,'Compute ROI','center',false,nil,true,false,true)
    end
    if requestType==2 then
        remoteRequestUi.ui=simBWF.createCustomUi(xml,'Generate production order','center',false,nil,true,false,true)
    end

    simUI.setEditValue(remoteRequestUi.ui,1,remoteRequestUi.userId)
    powerPricing_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,5,{{'110V',1},{'220V',2}},{},remoteRequestUi.power,false,{})
    connectorType_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,6,{{'Phoenix',1},{'Harting',2}},{},remoteRequestUi.connector,false,{})
    valveType_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,11,{{'SMC',1},{'Festo',2}},{},remoteRequestUi.valve,false,{})
    simUI.setEditValue(remoteRequestUi.ui,18,simBWF.format("%d",remoteRequestUi.numGrippers))
    frameType_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,12,{{'Linear',1},{'Compact',2}},{},remoteRequestUi.frame,false,{})
    simUI.setCheckboxValue(remoteRequestUi.ui,13,remoteRequestUi.frameLifter and 2 or 0,true)
    lineControl_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,7,{{'downstream',1},{'upstream',2}},{},remoteRequestUi.lineControl,false,{})
    simUI.setCheckboxValue(remoteRequestUi.ui,15,remoteRequestUi.brdLicense and 2 or 0,true)
    simUI.setEditValue(remoteRequestUi.ui,16,simBWF.format("%d",remoteRequestUi.broLicense))
    simUI.setCheckboxValue(remoteRequestUi.ui,17,remoteRequestUi.oeeLicense and 2 or 0,true)
    shipping_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,8,{{'land',1},{'air',2}},{},remoteRequestUi.shipping,false,{})
    destination_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,9,{{'Europe',1},{'Asia',2},{'USA',3}},{},remoteRequestUi.destination,false,{})
    documentation_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,10,{{'English',1},{'Chinese',2},{'Danish',3},{'German',4}},{},remoteRequestUi.documentation,false,{})
    quote_comboboxItems=simBWF.populateCombobox(remoteRequestUi.ui,14,{{'Compact',1},{'Detailed',2}},{},remoteRequestUi.quote,false,{})
end

function model.actions.quoteRequestDlg()
    if not quoteRequest then
        model.actions.remoteRequestDlg(0)
    end
end

function model.actions.roiRequestDlg()
    if not roiRequest then
        model.actions.remoteRequestDlg(1)
    end
end

function model.actions.sopRequestDlg()
    if not sopRequest then
        model.actions.remoteRequestDlg(2)
    end
end

function model.actions.onCloseVariousActionDlg()
    simUI.destroy(variousActionDlgId)
    variousActionDlgId=nil
end

function model.actions.variousActionDlg()
        local xml =[[
                <button text="Save configuration file" on-click="model.actions.saveJsonClick_callback" style="* {min-width: 300px;}" />
                <button text="Generate quote" on-click="model.actions.quoteRequestDlg" style="* {min-width: 300px;}" />
                <button text="Compute ROI" on-click="model.actions.roiRequestDlg" style="* {min-width: 300px;}" />
                <button text="Generate SOP" on-click="model.actions.sopRequestDlg" style="* {min-width: 300px;}" />
        ]]
        variousActionDlgId=simBWF.createCustomUi(xml,'Actions','center',true,"model.actions.onCloseVariousActionDlg",true,false,true)
end
