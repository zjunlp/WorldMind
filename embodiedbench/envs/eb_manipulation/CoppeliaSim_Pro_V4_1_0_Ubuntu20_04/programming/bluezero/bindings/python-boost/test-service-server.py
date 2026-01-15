import pyb0 as b0
def callback(req):
    print('Received request "%s"' % req)
    rep = 'hi'
    print('Sending reply "%s"...' % rep)
    return rep
node = b0.Node('python-service-server')
srv = b0.ServiceServer(node, 'control', callback)
node.init()
node.spin()
node.cleanup()
