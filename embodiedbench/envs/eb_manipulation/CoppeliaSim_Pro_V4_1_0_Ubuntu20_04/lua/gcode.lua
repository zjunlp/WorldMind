require 'utils'
require 'logger'

GCodeInterpreter = {
    verbose=false,
    warnAboutUnimplementedCommands=true,
    unitMultiplier=0.001,
    absolute=true,
    rapid=false,
    -- currentPos={0,0,0},
    -- targetPos={0,0,0},
    -- currentOrient={0,0,0},
    -- targetOrient={0,0,0},
    currentM={1,0,0,0,0,1,0,0,0,0,1,0},
    targetM={1,0,0,0,0,1,0,0,0,0,1,0},
    speed=0,
    lastMotion=1,
    motion=0, -- 1=linear, 2=cw, 3=ccw
    center={0,0,0}, -- if useCenter==true
    radius=0, -- if useCenter==false
    pathResolution=150,
    useCenter=true,
    lineNumber=0,
    wordNumber=0,
    pathNumber=0,
    param=0,
    visualizationMethod=1, -- 0=none, 1=drawing objects, 2=path objects
    greenLineContainer=-1,
    redLineContainer=-1,
    bluePointContainer=-1,
    pathItems={}, -- {duration,{pathPoints}},{duration,{pathPoints}}, etc., with pathPoints={pos1,pos2,pos3,orient1,orient2,orient3}

    createLinearPath=function(self,from,to)
        asserttable(from,'from',3,'number')
        asserttable(to,'to',3,'number')

        local d=math.hypotn(from,to)
        local n=math.max(1,math.floor(d*self.pathResolution))
        local points={}
        for i=0,n do
            local tau=i/n
            local point={}
            for j=1,3 do table.insert(point, from[j]*(1-tau)+to[j]*tau) end
            table.insert(points, point)
        end
        return points,d
    end,

    createCircularPath=function(self,from,to,direction,centerOrRadius)
        asserttable(from,'from',3,'number')
        asserttable(to,'to',3,'number')
        assertmember(direction,{-1,1},'direction')

        local t=type(centerOrRadius)
        if t=='number' then
            return self:createCircularPathWithRadius(from,to,direction,centerOrRadius)
        elseif t=='table' then
            return self:createCircularPathWithCenter(from,to,direction,centerOrRadius)
        else
            error('centerOrRadius must be a table or a number')
        end
    end,

    createCircularPathWithCenter=function(self,from,to,direction,center)
        asserttable(from,'from',3,'number')
        asserttable(to,'to',3,'number')
        assertmember(direction,{-1,1},'direction')
        asserttable(center,'center',3,'number')

        if math.abs(from[3]-to[3])>0.0001 then
            log(LOG.DEBUG,'createCircularPathWithCenter: from=%s to=%s center=%s r1=%f r2=%f',from,to,center,math.hypotn(from,center),math.hypotn(to,center))
            error('from/to points do not have the same Z')
        end

        -- compute start/end radiuses:
        local r1=math.hypotn(from,center)
        local r2=math.hypotn(to,center)
        if math.abs(r1-r2) > 0.0025 then
            error('start and end radius are not the same: error='..math.abs(r1-r2))
        end

        -- compute start/end angles:
        local As=math.atan2(from[2]-center[2],from[1]-center[1])
        local Ae=math.atan2(to[2]-center[2],to[1]-center[1])

        -- compute distance in radians:
        local angular_distance=0
        if direction>0 and As<Ae then
            angular_distance=Ae-As
        elseif direction>0 and As>Ae then
            angular_distance=2*math.pi-(As-Ae)
        elseif direction<0 and As<Ae then
            angular_distance=2*math.pi-(Ae-As)
        elseif direction<0 and As>Ae then
            angular_distance=As-Ae
        else
            error('WTF?')
        end

        -- linear distance:
        local d=angular_distance*r1

        -- circular (i.e. polar) interpolation:
        local n=math.max(1,math.floor(d*self.pathResolution))
        local da=angular_distance/n
        local points={}
        for i=0,n do
            local a=As+direction*da*i
            local point={center[1]+r1*math.cos(a),center[2]+r1*math.sin(a),from[3]}
            table.insert(points, point)
        end
        return points,d
    end,

    createCircularPathWithRadius=function(self,from,to,direction,radius)
        asserttable(from,'from',3,'number')
        asserttable(to,'to',3,'number')
        assertmember(direction,{-1,1},'direction')
        assertnumber(radius,'radius')

        local r=radius
        local x1=from[1]
        local y1=from[2]
        local x2=to[1]
        local y2=to[2]
        local z=from[3]

        -- find the centers of the two circles passing thru (x1,y1) and (x2,y2):
        local x3=(x1+x2)/2
        local y3=(y1+y2)/2
        local d=math.hypotn({x1,y1},{x2,y2})
        local xA=x3+math.sqrt(r*r-d*d/4)*(y1-y2)/d
        local yA=y3+math.sqrt(r*r-d*d/4)*(x2-x1)/d
        local xB=x3-math.sqrt(r*r-d*d/4)*(y1-y2)/d
        local yB=y3-math.sqrt(r*r-d*d/4)*(x2-x1)/d

        if ((x2-x1)*(yA-y1)-(y2-y1)*(xA-x1))>0 then
            xL,yL,xR,yR=xA,yA,xB,yB
        else
            xL,yL,xR,yR=xB,yB,xA,yA
        end

        if direction>0 then
            return self:createCircularPathWithCenter(from,to,direction,{xL,yL,z})
        else
            return self:createCircularPathWithCenter(from,to,direction,{xR,yR,z})
        end
    end,

    onBeginProgram=function(self,program)
    end,

    onEndProgram=function(self,program)
        log(LOG.INFO,'parsed %d words in %d lines',self.wordNumber,self.lineNumber)
    end,

    runProgram=function(self,program,visualizePath)
        assertstring(program,'program')

        self:onBeginProgram(program)
        self.pathItems={}
        self.greenLineContainer=nil
        self.redLineContainer=nil
        self.bluePointContainer=nil
        if visualizePath then
            self.visualizationMethod=1
        else
            self.visualizationMethod=0
        end
        if  self.visualizationMethod==1 then
            self.greenLineContainer=sim.addDrawingObject(sim.drawing_lines,1,0,-1,99999999,{0,0,0},nil,nil,{0,1,0})
            self.redLineContainer=sim.addDrawingObject(sim.drawing_lines,1,0,-1,99999999,{0,0,0},nil,nil,{1,0,0})
            self.bluePointContainer=sim.addDrawingObject(sim.drawing_spherepoints,0.0025,0,-1,99999999,{0,0,0},nil,nil,{0,0,1})
        end

        local lines=string.splitlines(program)
        for i=1,#lines do
            self.lineNumber=self.lineNumber+1
            self:runLine(lines[i])
        end
        
        self:onEndProgram(program)
        return self.pathItems, self.greenLineContainer, self.redLineContainer, self.bluePointContainer
    end,

    onBeginLine=function(self,line)
        log(LOG.TRACE,'>>>>>>>>  %s',line)
    end,

    onEndLine=function(self,line)
        self:executeMotion()
    end,

    runLine=function(self,line)
        assertstring(line,'line')

        self.center={0,0,0}
        self.radius=0
        self.motion=0

        self:onBeginLine(line)

        local handler=function(address,value)
            self.wordNumber=self.wordNumber+1
            local valueNum=tonumber(value)
            local f=address:upper()
            local f1=f..valueNum
            if self[f1]~=nil then
                self[f1](self)
            elseif self[f]~=nil then
                self[f](self,valueNum)
            else
                log(LOG.WARN,'command '..address..valueNum..' not implemented')
            end
        end

        local comment=false
        local comment2=false
        local addr=nil
        local val=''
        for ch in line:gmatch('.') do
            if ch==';' then comment2=true
            elseif ch=='(' then comment=true
            elseif ch==')' then comment=false
            elseif not (ch==' ' or ch=='\t' or comment or comment2) then
                if ch:match('%a') then
                    if addr~=nil and val~='' then handler(addr,val) end
                    addr,val=ch,''
                elseif addr==nil then
                    error('unexpected "'..ch..'" while waiting for an address')
                else
                    val=val..ch
                end 
            end
        end
        if addr~=nil and val~='' then handler(addr,val) end

        self.lastMotion=self.motion

        self:onEndLine(line)
    end,

    executeMotion=function(self)
        local from={0,0,0}
        local to={0,0,0}
        local center={0,0,0}
        local radius=self.radius
        -- local os=self.currentOrient
        -- local oe=self.targetOrient
        local os=self.currentM
        local oe=self.targetM

        local red={0,0,0,0,0,0,0,0,0,1,0,0}
        local green={0,0,0,0,0,0,0,0,0,0,1,0}
        local blue={0,0,0,0,0,0,0,0,0,0,0,1}

        -- scale units:
        -- for i=1,3 do
        --     center[i]=(self.currentPos[i]+self.center[i])*self.unitMultiplier
        --     from[i]=self.currentPos[i]*self.unitMultiplier
        --     to[i]=self.targetPos[i]*self.unitMultiplier
        -- end
        for i=1,3 do
            center[i]=(self.currentM[i*4]+self.center[i])*self.unitMultiplier
            from[i]=self.currentM[i*4]*self.unitMultiplier
            to[i]=self.targetM[i*4]*self.unitMultiplier
        end
        
        radius=radius*self.unitMultiplier

        local d=math.hypotn(from,to)

        local createDummyContainer=function()
            local status,dh=pcall(function() return sim.getObjectHandle('Path') end)
            if not status then
                dh=sim.createDummy(0)
                sim.setObjectName(dh,'Path')
            end
            sim.writeCustomDataBlock(dh,'count',self.pathNumber)
            return dh
        end
        if self.motion==1 or self.motion==2 or self.motion==3 then
            local direction=2*self.motion-5
            local p={}
            local len=-1
            local tstr='?'
            local pstr='    [path '..self.pathNumber..'] '
            if self.motion==1 then
                tstr='line'
                p,len=self:createLinearPath(from,to)
            else
                tstr='arc'
                p,len=self:createCircularPath(from,to,direction,(self.useCenter and center or radius))
            end
            log(LOG.TRACE,'generated %d path points',#p)
            self.pathNumber=self.pathNumber+1
            
            if self.visualizationMethod==2 then
                local dh=createDummyContainer()
                local h=sim.createPath(sim.pathproperty_show_line,{1,sim.distcalcmethod_dl,0},{0.001,1,1},(self.rapid and red or green))
                sim.writeCustomDataBlock(h,'duration',sim.packFloatTable({len/self.speed}))
                sim.setObjectName(h,string.format('Path_%06d',self.pathNumber))
                sim.setObjectParent(h,dh,true)
                local data={}
                for i=1,#p do
                    local tau=(i-1)/(#p-1)
                    for j=1,3 do table.insert(data,p[i][j]) end
                    -- for j=1,3 do table.insert(data,self.currentOrient[j]*(1-tau)+self.targetOrient[j]*tau) end
                    local m=sim.interpolateMatrices(self.currentM,self.targetM,tau)
                    local euler=sim.getEulerAnglesFromMatrix(m)
                    for j=1,3 do table.insert(data,euler[j]) end
                    for j=1,5 do table.insert(data,0) end
                end
                sim.insertPathCtrlPoints(h,0,0,#p,data)
            end
            if self.visualizationMethod==1 then
                for i=1,#p-1 do
                    local data={p[i][1],p[i][2],p[i][3],p[i+1][1],p[i+1][2],p[i+1][3]}
                    if self.rapid then
                        sim.addDrawingObjectItem(self.redLineContainer,data)
                    else
                        sim.addDrawingObjectItem(self.greenLineContainer,data)
                    end
                end
            end
            local pathPoints={}
            for i=1,#p do
                local tau=(i-1)/(#p-1)
                for j=1,3 do table.insert(pathPoints,p[i][j]) end
                -- for j=1,3 do table.insert(pathPoints,self.currentOrient[j]*(1-tau)+self.targetOrient[j]*tau) end
                local m=sim.interpolateMatrices(self.currentM,self.targetM,tau)
                local euler=sim.getEulerAnglesFromMatrix(m)
                for j=1,3 do table.insert(pathPoints,euler[j]) end
            end
            self.pathItems[#self.pathItems+1]={len/self.speed,pathPoints}
        elseif self.motion==4 then
            -- pause
            local seconds=0.001*self.param
            self.pathNumber=self.pathNumber+1
            if self.visualizationMethod==2 then
                local dh=createDummyContainer()
                local h=sim.createDummy(0)
                sim.setObjectName(h,string.format('Path_%06d',self.pathNumber))
                sim.setObjectParent(h,dh,true)
                sim.writeCustomDataBlock(h,'duration',sim.packFloatTable({seconds}))
            end
            if self.visualizationMethod==1 then
                -- sim.addDrawingObjectItem(self.bluePointContainer,{self.currentPos[1]*self.unitMultiplier,self.currentPos[2]*self.unitMultiplier,self.currentPos[3]*self.unitMultiplier})
                sim.addDrawingObjectItem(self.bluePointContainer,{self.currentM[4]*self.unitMultiplier,self.currentM[8]*self.unitMultiplier,self.currentM[12]*self.unitMultiplier})
            end
            self.pathItems[#self.pathItems+1]={seconds,{}}
        end

        -- for i=1,3 do
        --     self.currentPos[i]=self.targetPos[i]
        --     self.currentOrient[i]=self.targetOrient[i]
        -- end
        for i=1,12 do
            self.currentM[i]=self.targetM[i]
        end
        
    end,

    A=function(self,value)
        -- A: Absolute or incremental position of A axis (rotational axis around X axis)
        log(LOG.TRACE,'A%s  A-axis position',value)
        -- self.targetOrient[1]=(self.absolute and 0 or self.targetOrient[1])+value
        if self.absolute then
            local euler=sim.getEulerAnglesFromMatrix(self.targetM)
            self.targetM=sim.buildMatrix({self.targetM[4],self.targetM[8],self.targetM[12]},{value,euler[2],euler[3]})
        else
            self.targetM=sim.rotateAroundAxis(self.targetM,{self.targetM[1],self.targetM[5],self.targetM[9]},{self.targetM[4],self.targetM[8],self.targetM[12]},value)
        end
    end,

    B=function(self,value)
        -- B: Absolute or incremental position of B axis (rotational axis around Y axis)	
        log(LOG.TRACE,'B%s  B-axis position',value)
        -- self.targetOrient[2]=(self.absolute and 0 or self.targetOrient[2])+value
        if self.absolute then
            local euler=sim.getEulerAnglesFromMatrix(self.targetM)
            self.targetM=sim.buildMatrix({self.targetM[4],self.targetM[8],self.targetM[12]},{euler[1],value,euler[3]})
        else
            self.targetM=sim.rotateAroundAxis(self.targetM,{self.targetM[2],self.targetM[6],self.targetM[10]},{self.targetM[4],self.targetM[8],self.targetM[12]},value)
        end
    end,

    C=function(self,value)
        -- C: Absolute or incremental position of C axis (rotational axis around Z axis)	
        log(LOG.TRACE,'C%s  C-axis position',value)
        -- self.targetOrient[3]=(self.absolute and 0 or self.targetOrient[3])+value
        if self.absolute then
            local euler=sim.getEulerAnglesFromMatrix(self.targetM)
            self.targetM=sim.buildMatrix({self.targetM[4],self.targetM[8],self.targetM[12]},{euler[1],euler[2],value})
        else
            self.targetM=sim.rotateAroundAxis(self.targetM,{self.targetM[3],self.targetM[7],self.targetM[11]},{self.targetM[4],self.targetM[8],self.targetM[12]},value)
        end
    end,

    F=function(self,value)
        -- F: Defines feed rate.
        --    Common units are distance per time for mills (inches per minute, IPM, or
        --    millimeters per minute, mm/min) and distance per revolution for lathes
        --    (inches per revolution, IPR, or millimeters per revolution, mm/rev)
        log(LOG.TRACE,'F%s  Feedrate',value)
        speed=value
    end,

    G0=function(self)
        log(LOG.TRACE,'G00  Rapid positioning')
        self.rapid=true
        self.motion=1
    end,

    G1=function(self)
        log(LOG.TRACE,'G01  Linear interpolation')
        self.rapid=false
        self.motion=1
    end,

    G2=function(self)
        log(LOG.TRACE,'G02  Circular interpolation, clockwise')
        --      Center given with I,J,K commands (or radius with R)
        self.rapid=false
        self.motion=2
    end,

    G3=function(self)
        log(LOG.TRACE,'G03  Circular interpolation, counterclockwise')
        --      Center given with I,J,K commands (or radius with R)
        self.rapid=false
        self.motion=3
    end,

    G4=function(self)
        log(LOG.TRACE,'G03  Dwell (pause)')
        self.motion=4
    end,

    G20=function(self)
        log(LOG.TRACE,'G20  Programming in inches')
        self.unitMultiplier=25.4*0.001
    end,

    G21=function(self)
        log(LOG.TRACE,'G21  Programming in millimeters (mm)')
        self.unitMultiplier=0.001
    end,

    G28=function(self)
        log(LOG.TRACE,'G28  Return to home position')
        -- self.targetPos={0,0,0}
        self.targetM[4]=0
        self.targetM[8]=0
        self.targetM[12]=0
    end,

    G90=function(self)
        log(LOG.TRACE,'G90  Absolute programming')
        self.absolute=true
    end,

    G91=function(self)
        log(LOG.TRACE,'G91  Incremental programming')
        self.absolute=false
    end,

    I=function(self,value)
        -- I: Defines arc center in X axis for G02 or G03 arc commands.
        --    Also used as a parameter within some fixed cycles.
        log(LOG.TRACE,'I%s  Arc center in X axis',value)
        self.center[1]=value
        self.useCenter=true
    end,

    J=function(self,value)
        -- J: Defines arc center in Y axis for G02 or G03 arc commands.
        --    Also used as a parameter within some fixed cycles.	
        log(LOG.TRACE,'J%s  Arc center in Y axis',value)
        self.center[2]=value
        self.useCenter=true
    end,

    K=function(self,value)
        -- K: Defines arc center in Z axis for G02 or G03 arc commands.
        --    Also used as a parameter within some fixed cycles, equal to L address.	
        log(LOG.TRACE,'K%s  Arc center in Z axis',value)
        self.center[3]=value
        self.useCenter=true
    end,

    P=function(self,value)
        self.param=value
    end,

    R=function(self,value)
        -- R: Defines size of arc radius, or defines retract height in milling canned cycles
        --    For radii, not all controls support the R address for G02 and G03, in which
        --    case IJK vectors are used. For retract height, the "R level", as it's called,
        --    is returned to if G99 is programmed.
        log(LOG.TRACE,'R%s  Size of arc radius',value)
        self.radius=value
        self.useCenter=false
    end,

    U=function(self,value)
        log(LOG.TRACE,'U%s  Incremental position of X axis',value)
        -- self.targetPos[1]=self.targetPos[1]+value
        self.targetM[4]=self.targetM[4]+value
        if self.motion==0 then self.motion=self.lastMotion end
    end,

    V=function(self,value)
        log(LOG.TRACE,'V%s  Incremental position of X axis',value)
        -- self.targetPos[2]=self.targetPos[2]+value
        self.targetM[8]=self.targetM[8]+value
        if self.motion==0 then self.motion=self.lastMotion end
    end,

    W=function(self,value)
        log(LOG.TRACE,'W%s  Incremental position of X axis',value)
        -- self.targetPos[3]=self.targetPos[3]+value
        self.targetM[12]=self.targetM[12]+value
        if self.motion==0 then self.motion=self.lastMotion end
    end,

    X=function(self,value)
        -- X: Absolute or incremental position of X axis.
        --    Also defines dwell time on some machines (instead of "P" or "U").
        log(LOG.TRACE,'X%s  Absolute/incremental position of X axis',value)
        -- self.targetPos[1]=(self.absolute and 0 or self.targetPos[1])+value
        self.targetM[4]=(self.absolute and 0 or self.targetM[4])+value
        if self.motion==0 then self.motion=self.lastMotion end
    end,

    Y=function(self,value)
        -- Y: Absolute or incremental position of Y axis	
        log(LOG.TRACE,'Y%s  Absolute/incremental position of Y axis',value)
        -- self.targetPos[2]=(self.absolute and 0 or self.targetPos[2])+value
        self.targetM[8]=(self.absolute and 0 or self.targetM[8])+value
        if self.motion==0 then self.motion=self.lastMotion end
    end,

    Z=function(self,value)
        -- Z: Absolute or incremental position of Z axis
        --    The main spindle's axis of rotation often determines which axis of a
        --    machine tool is labeled as Z.
        log(LOG.TRACE,'Z%s  Absolute/incremental position of Z axis',value)
        -- self.targetPos[3]=(self.absolute and 0 or self.targetPos[3])+value
        self.targetM[12]=(self.absolute and 0 or self.targetM[12])+value
        if self.motion==0 then self.motion=self.lastMotion end
    end
}