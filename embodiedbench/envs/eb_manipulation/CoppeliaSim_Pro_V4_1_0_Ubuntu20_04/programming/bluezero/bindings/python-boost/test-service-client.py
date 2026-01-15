import pyb0 as b0
node = b0.Node('python-service-client')
cli = b0.ServiceClient(node, 'control')
node.init()
req = 'hello'
print('Sending "%s"...' % req)
rep = cli.call(req)
print('Received "%s"' % rep)
node.cleanup()
