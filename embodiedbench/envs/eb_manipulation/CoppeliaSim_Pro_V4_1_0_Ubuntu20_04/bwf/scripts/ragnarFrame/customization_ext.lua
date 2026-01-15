model.ext={}

function model.ext.adjustFrame(frameState,width,height,doorState)
    -- nil for args that should stay same
    -- frameState: 0=not present, 1=present, 2=hidden
    -- doorState: 0=closed, 1=open, 2=hidden
    model.adjustFrame(frameState,width,height,doorState)
end
