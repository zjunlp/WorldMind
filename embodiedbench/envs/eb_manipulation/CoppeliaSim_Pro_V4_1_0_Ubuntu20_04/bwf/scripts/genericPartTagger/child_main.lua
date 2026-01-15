function model.getDistributionValue(distribution)
    if (type(distribution[1])=='table') and (#distribution[1]==2) then
        local cnt=0
        for i=1,#distribution,1 do
           cnt=cnt+distribution[i][1] 
        end
        local p=sim.getFloatParameter(sim.floatparam_rand)*cnt
        cnt=0
        for i=1,#distribution,1 do
            if cnt+distribution[i][1]>=p then
                return distribution[i][2]
            end
            cnt=cnt+distribution[i][1] 
        end
    else
        local cnt=#distribution
        local p=1+math.floor(sim.getFloatParameter(sim.floatparam_rand)*cnt-0.0001)
        return distribution[p]
    end
end

function model.isPartDetected(partHandle)
    local p=sim.getObjectPosition(partHandle,model.handle)
    return math.abs(p[1])<model.width*0.5 and math.abs(p[2])<model.length*0.5 and math.abs(p[3])<model.height*0.5
end

function sysCall_init()
    model.codeVersion=1
    
    local data=model.readInfo()
    if sim.boolAnd32(data['bitCoded'],8)>0 then
        model.console=sim.auxiliaryConsoleOpen('Tagged Parts',1000,4,nil,{600,300},nil,{0.9,0.9,1})
    end
    model.changeColor=(sim.boolAnd32(data['bitCoded'],16)>0)
    model.width=data['size'][1]
    model.length=data['size'][2]
    model.height=data['size'][3]
    model.alreadyDetectedPartsList={}
    model.counter=0
end

function sysCall_sensing()
    local t=sim.getSimulationTime()
    if model.changeColor then
        local line=''
        local p=simBWF.getAllInstanciatedParts()
        local newColorDistribution
        if #p>0 then
            local data=sim.readCustomDataBlock(model.handle,model.tagName)
            data=sim.unpackTable(data)
            newColorDistribution='{'..data['partColorDistribution']..'}'
            f=loadstring("return "..newColorDistribution)
            newColorDistribution=f()
        end
        for i=1,#p,1 do
            if model.isPartDetected(p[i]) then
                if not model.alreadyDetectedPartsList[p[i]] then
                    model.alreadyDetectedPartsList[p[i]]=t
                    local colorChanged=false
                    local data=sim.readCustomDataBlock(p[i],simBWF.modelTags.PART)
                    data=sim.unpackTable(data)
                    local lline="Object name '"..simBWF.getObjectAltName(p[i]).."':\n"
                    if model.changeColor then
                        local newColor=model.getDistributionValue(newColorDistribution)
                        if newColor and newColor~='<DEFAULT>' then
                            colorChanged=true
                            local l=sim.getObjectsInTree(p[i],sim.object_shape_type)
                            for i=1,#l,1 do
                                sim.setShapeColor(l[i],nil,sim.colorcomponent_ambient_diffuse,newColor)
                            end
                        end
                    end
                    if colorChanged then
                        lline=lline.."    --> color was changed\n"
                    end
                    lline=lline.."\n"
                    if colorChanged then
                        line=line..lline
                        sim.writeCustomDataBlock(p[i],simBWF.modelTags.PART,sim.packTable(data))
                    end
                    model.counter=model.counter+1
                end
            else
                model.alreadyDetectedPartsList[p[i]]=nil
            end
        end
        if model.console and line~='' then
            sim.auxiliaryConsolePrint(model.console,line)
        end
    end
end

