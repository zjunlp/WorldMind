local simB0={}

--@fun spin Call nodeSpinOnce() continuously
--@arg string handle the node handle
function simB0.nodeSpin(handle)
    while sim.getSimulationState()~=sim.simulation_advancing_abouttostop do
        simB0.nodeSpinOnce(handle)
        sim.switchThread()
    end
end

--@fun pingResolver Check if resolver node is reachable
function simB0.pingResolver()
    local dummyNode=simB0.nodeCreate('dummyNode')
    simB0.nodeSetAnnounceTimeout(dummyNode, 2000) -- 2 seconds timeout
    local running=pcall(function() simB0.nodeInit(dummyNode) end)
    if running then simB0.nodeCleanup(dummyNode) end
    simB0.nodeDestroy(dummyNode)
    return running
end

-- Since Dec 2018, functions have been renamed for consistency:
function simB0.renamedFunction(oldName, newName)
    return function()
        msg = string.format('Function simB0.%s has been renamed to simB0.%s.', oldName, newName)
        error(msg)
    end
end
simB0.create = simB0.renamedFunction('create', 'nodeCreate')
simB0.setAnnounceTimeout = simB0.renamedFunction('setAnnounceTimeout', 'nodeSetAnnounceTimeout')
simB0.init = simB0.renamedFunction('init', 'nodeInit')
simB0.spinOnce = simB0.renamedFunction('spinOnce', 'nodeSpinOnce')
simB0.cleanup = simB0.renamedFunction('cleanup', 'nodeCleanup')
simB0.destroy = simB0.renamedFunction('destroy', 'nodeDestroy')
simB0.initSocket = simB0.renamedFunction('initSocket', 'socketInit')
simB0.spinOnceSocket = simB0.renamedFunction('spinOnceSocket', 'socketSpinOnce')
simB0.pollSocket = simB0.renamedFunction('pollSocket', 'socketPoll')
simB0.readSocket = simB0.renamedFunction('readSocket', 'socketRead')
simB0.writeSocket = simB0.renamedFunction('writeSocket', 'socketWrite')
simB0.cleanupSocket = simB0.renamedFunction('cleanupSocket', 'socketCleanup')
simB0.createPublisher = simB0.renamedFunction('createPublisher', 'publisherCreate')
simB0.publish = simB0.renamedFunction('publish', 'publisherPublish')
simB0.destroyPublisher = simB0.renamedFunction('destroyPublisher', 'publisherDestroy')
simB0.createSubscriber = simB0.renamedFunction('createSubscriber', 'subscriberCreate')
simB0.destroySubscriber = simB0.renamedFunction('destroySubscriber', 'subscriberDestroy')
simB0.createServiceClient = simB0.renamedFunction('createServiceClient', 'serviceClientCreate')
simB0.call = simB0.renamedFunction('call', 'serviceClientCall')
simB0.destroyServiceClient = simB0.renamedFunction('destroyServiceClient', 'serviceClientDestroy')
simB0.createServiceServer = simB0.renamedFunction('createServiceServer', 'serviceServerCreate')
simB0.destroyServiceServer = simB0.renamedFunction('destroyServiceServer', 'serviceServerDestroy')
simB0.setCompression = simB0.renamedFunction('setCompression', 'socketSetCompression')
simB0.setSocketOption = simB0.renamedFunction('setSocketOption', 'socketSetOption')

return simB0
