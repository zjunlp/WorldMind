function sysCall_init()
    model.codeVersion=1
    
    model.dlg.init()

    model.selectedObj=-1
end

function sysCall_nonSimulation()
    model.dlg.showOrHideDlgIfNeeded()
end


function sysCall_afterSimulation()
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
    model.dlg.cleanup()
end
