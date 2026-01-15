local simIM={}

function simIM.numActiveHandles()
    local h=simIM.handles()
    return #h
end

return simIM
