# -*- coding: utf-8 -*-
import b0
import time

if not b0.is_initialized():
    b0.init()
node = b0.Node('python-publisher')
pub = b0.Publisher(node, 'A')
node.init()
print('Publishing to topic "%s"...' % pub.get_topic_name())
i = 0
while not node.shutdown_requested():
    msg_str = u'µsg-%d' % i
    i += 1
    print('Sending message "%s"...' % msg_str)
    msg = msg_str.encode('utf-8')
    pub.publish(msg)
    node.spin_once()
    time.sleep(1)
node.cleanup()
