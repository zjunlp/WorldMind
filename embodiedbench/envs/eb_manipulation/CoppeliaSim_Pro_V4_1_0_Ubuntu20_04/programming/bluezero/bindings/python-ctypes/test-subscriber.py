# -*- coding: utf-8 -*-
import b0

if not b0.is_initialized():
    b0.init()
def callback(msg):
    msg_str = msg.decode('utf-8')
    print('Received message "%s"' % msg_str)
node = b0.Node('python-subscriber')
sub = b0.Subscriber(node, 'A', callback)
node.init()
print('Subscribed to topic "%s"...' % sub.get_topic_name())
node.spin()
node.cleanup()
