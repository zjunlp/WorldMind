import pyb0 as b0
from time import sleep
node = b0.Node('python-publisher')
pub = b0.Publisher(node, 'A')
node.init()
for i in range(1000000):
    msg = 'msg-%d' % i
    print('Sending message "%s"...' % msg)
    pub.publish(msg)
    sleep(1)
node.cleanup()
