require 'utils'

LOG={
    TRACE={str='TRACE', level=1},
    DEBUG={str='DEBUG', level=2},
    INFO={str='INFO', level=3},
    WARN={str='WARNING', level=4},
    ERROR={str='ERROR', level=5},
}
loglevel=LOG.ERROR

function log(l,fmt,...)
    if l.level>=loglevel.level then
        local txt=string.format('[%s] ',l.str)..string.formatex(fmt,unpack(arg))
        print(txt)
    end
end

function setloglevel(lvl)
    loglevel=lvl
end