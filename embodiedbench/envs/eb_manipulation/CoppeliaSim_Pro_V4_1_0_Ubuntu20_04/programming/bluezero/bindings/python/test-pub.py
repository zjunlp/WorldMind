import bluezero as b0
import time
node = b0.Node('python-swig-publisher-node')
pub = b0.Publisher(node, 'mytopic')
node.init()
for i in range(10000):
    print('.')
    pub.publish('msg-%d' % i)
    time.sleep(1000)
node.cleanup()
