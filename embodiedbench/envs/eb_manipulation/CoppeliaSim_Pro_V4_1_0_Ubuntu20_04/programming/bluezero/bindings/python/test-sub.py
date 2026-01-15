import bluezero as b0
import time
node = b0.Node('python-swig-subscriber-node')
class Cb(b0.SubscriberCallback):
    def onMessage(self, m):
        print('received message: %s' % m)
sub = b0.Subscriber(node, 'mytopic', Cb())
node.init()
node.spin()
node.cleanup()
