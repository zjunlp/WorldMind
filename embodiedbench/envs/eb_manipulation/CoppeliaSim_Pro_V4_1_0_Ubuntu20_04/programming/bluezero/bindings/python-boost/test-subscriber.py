import pyb0 as b0
def callback(msg):
    print('Received message: "%s"' % msg)
node = b0.Node('python-subscriber')
sub = b0.Subscriber(node, 'A', callback)
node.init()
node.spin()
node.cleanup()
