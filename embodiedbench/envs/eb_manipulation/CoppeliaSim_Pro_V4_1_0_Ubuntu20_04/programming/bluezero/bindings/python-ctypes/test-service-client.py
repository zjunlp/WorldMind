# -*- coding: utf-8 -*-
import b0

if not b0.is_initialized():
    b0.init()
node = b0.Node('python-service-client')
cli = b0.ServiceClient(node, 'control')
node.init()
print('Using service "%s"...' % cli.get_service_name())
req_str = u'hellò'
print('Sending "%s"...' % req_str)
req = req_str.encode('utf-8')
rep = cli.call(req)
rep_str = rep.decode('utf-8')
print('Received "%s"' % rep_str)
node.cleanup()
