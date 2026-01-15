# -*- coding: utf-8 -*-
import b0

if not b0.is_initialized():
    b0.init()
def callback(req):
    req_str = req.decode('utf-8')
    print('Received request "%s"...' % req_str)
    rep_str = u'hì'
    print('Sending reply "%s"...' % rep_str)
    rep = rep_str.encode('utf-8')
    return rep
node = b0.Node('python-service-server')
srv = b0.ServiceServer(node, 'control', callback)
node.init()
print('Offering service "%s"...' % srv.get_service_name())
node.spin()
node.cleanup()
