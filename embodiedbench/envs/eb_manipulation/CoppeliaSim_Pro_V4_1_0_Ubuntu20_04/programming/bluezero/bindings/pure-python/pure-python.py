import json
import time
import os
import zmq

class MessagePart:
    __slots__ = 'content_type', 'compression_algorithm', 'compression_level', 'payload'
    def __init__(self):
        for f in ('content_type', 'compression_algorithm', 'payload'): setattr(self, f, '')
        self.compression_level = 0
    def parse(self, i, last_start, headers, payload):
        content_length = int(headers.pop('Content-length-%d' % i, 0))
        self.content_type = headers.pop('Content-type-%d' % i, '')
        self.compression_algorithm = headers.pop('Compression-algorithm-%d' % i, '')
        self.compression_level = int(headers.pop('Compression-level-%d' % i, 0))
        uncompressed_content_length = int(headers.pop('Uncompressed-content-length-%d' % i, 0))
        if self.compression_algorithm: raise RuntimeError('compression not supported')
        self.payload = payload[last_start:last_start+content_length]
        return last_start + content_length
    def serialize_headers(self, i):
        h = [('Content-length-%d', len(self.payload))]
        if self.content_type: h.append(('Content-type-%d', self.content_type))
        if self.compression_algorithm: raise RuntimeError('compression not supported')
        #    h.append(('Compression-algorithm-%d', self.compression_algorithm))
        #    if self.compression_level: h.append(('Compression-level-%d', self.compression_level))
        #    uncompressed_content_length = 0
        #    h.append(('Uncompressed-content-length-%d', uncompressed_content_length))
        return ''.join('%s: %s\n' % ((k % i), v) for k, v in h)

class MessageEnvelope:
    __slots__ = 'header0', 'parts', 'headers'
    def __init__(self):
        self.header0, self.parts, self.headers = '', [], {}
    def pop_header(key, value_type=str, default_value=''):
        return value_type(self.headers.pop(key, default_value))
    def parse(self, s):
        headers_txt, payload = s.split('\n\n', 1)
        self.header0, *header_lines = headers_txt.splitlines()
        self.headers = {k: v for line in header_lines for (k, v) in [line.split(': ', 1)]}
        part_count, self.parts, last_start = int(self.headers.pop('Part-count')), [], 0
        for i in range(part_count):
            part = MessagePart()
            last_start = part.parse(i, last_start, self.headers, payload)
            self.parts.append(part)
    def serialize(self):
        s = self.header0 + '\n'
        s += 'Content-length: %d\n' % sum(len(part.payload) for part in self.parts)
        s += 'Part-count: %d\n' % len(self.parts)
        s += ''.join(part.serialize_headers(i) for i, part in enumerate(self.parts))
        s += ''.join('%s: %s\n' % (key, value) for key, value in self.headers.items())
        s += '\n'
        for part in self.parts: s += part.payload
        return s

class Socket:
    def __init__(self, node, sock_type, name, managed=True, notify_graph=False):
        self.node, self.name, self.notify_graph, self.remote_addr = node, name, notify_graph, ''
        if managed: self.node.add_socket(self)
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(sock_type)
    def get_free_addr(self):
        import socket
        host_id = os.environ.get('B0_HOST_ID', 'localhost')
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        s.close()
        return host_id, free_port
    def spin_once(self):
        pass
    def poll(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        socks = dict(poller.poll())
        return socks[self.socket] == zmq.POLLIN
    def read_envelope(self):
        data = self.socket.recv_string()
        env = MessageEnvelope()
        env.parse(data)
        if env.header0 != self.name:
            raise TypeError('bad header: got %s, expected %s' % (env.header0, self.name))
        return env
    def read_msg(self):
        env = self.read_envelope()
        if not env.parts:
            raise RuntimeError('received message with no parts')
        try:
            msg = env.parts[0].payload
            msg = json.loads(msg)
        except: pass
        return msg, env.parts[0].content_type
    def write_envelope(self, env):
        data = env.serialize()
        return self.socket.send_string(data)
    def write_msg(self, msg, msgtype):
        env = MessageEnvelope()
        env.header0 = self.name
        env.parts = [MessagePart()]
        env.parts[0].payload = json.dumps(msg) if isinstance(msg, dict) else msg
        env.parts[0].content_type = msgtype
        self.write_envelope(env)
    def notify_topic(self, reverse, active):
        self.node.resolv_call({'node_topic': {'node_name': self.node.name, 'topic_name': self.name,
            'reverse': reverse, 'active': active}})
    def notify_service(self, reverse, active):
        self.node.resolv_call({'node_service': {'node_name': self.node.name, 'service_name': self.name,
            'reverse': reverse, 'active': active}})

class Publisher(Socket):
    def __init__(self, node, name, managed=True, notify_graph=True):
        super().__init__(node, zmq.PUB, name, managed, notify_graph)
    def init(self):
        self.socket.connect(self.remote_addr or self.node.xsub_addr)
        if self.notify_graph: self.notify_topic(False, True)
    def publish(self, msg, msgtype):
        self.write_msg(msg, msgtype)
    def cleanup(self):
        if self.notify_graph: self.notify_topic(False, False)
        self.socket.disconnect()

class Subscriber(Socket):
    def __init__(self, node, name, callback, managed=True, notify_graph=True):
        super().__init__(node, zmq.SUB, name, managed, notify_graph)
        if not callable(callback): raise TypeError('callback is not callable')
        self.callback = callback
    def init(self):
        self.socket.connect(self.remote_addr or self.node.xpub_addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.name)
        if self.notify_graph: self.notify_topic(True, True)
    def spin_once(self):
        while self.poll():
            msg, msgtype = self.read_msg()
            self.callback(msg, msgtype)
    def cleanup(self):
        if self.notify_graph: self.notify_topic(True, False)
        self.socket.disconnect()

class ServiceClient(Socket):
    def __init__(self, node, name, managed=True, notify_graph=True):
        super().__init__(node, zmq.REQ, name, managed, notify_graph)
    def init(self):
        self.socket.connect(self.remote_addr or self.resolve_service(self.name))
        if self.notify_graph: self.notify_service(True, True)
    def resolve_service(self, name):
        rep, reptype = self.node.resolv_call({'resolve_service': {'service_name': name}})
        if not rep['ok']: raise RuntimeError('failed resolve service')
        return rep['sock_addr']
    def call(self, req, reqtype):
        self.write_msg(req, reqtype)
        rep, reptype = self.read_msg()
        return rep, reptype
    def cleanup(self):
        if self.notify_graph: self.notify_service(True, False)

class ServiceServer(Socket):
    def __init__(self, node, name, callback, managed=True, notify_graph=True):
        super().__init__(node, zmq.REP, name, managed, notify_graph)
        if not callable(callback): raise TypeError('callback is not callable')
        self.callback = callback
    def init(self):
        hostname, free_port = self.get_free_addr()
        self.socket.bind('tcp://*:%d' % free_port)
        self.announce_service('tcp://%s:%d' % (hostname, free_port))
        if self.notify_graph: self.notify_service(False, True)
    def announce_service(self, addr):
        rep, reptype = self.node.resolv_call({'announce_service': {'node_name': self.node.name,
            'service_name': self.name, 'sock_addr': addr}})
        if not rep['ok']: raise TypeError('announce service failed')
    def spin_once(self):
        while self.poll():
            req, reqtype = self.read_msg()
            rep, reptype = self.callback(req, reqtype)
            self.write_msg(rep, reptype)
    def cleanup(self):
        if self.notify_graph: self.notify_service(False, False)

class Node:
    def __init__(self, name):
        self.name = name
        self.sockets = []
        self.shutdown_flag = False
        self.state = 'created'
        self.resolv_cli = ServiceClient(self, 'resolv', False, False)
    def add_socket(self, socket):
        if self.state != 'created':
            raise RuntimeError('cannot add socket to an already initialized node')
        self.sockets.append(socket)
    def resolv_call(self, req):
        rep, reptype = self.resolv_cli.call(req, 'ResolvRequest')
        return rep[list(req.keys())[0]], reptype
    def init(self):
        if self.state != 'created': raise RuntimeError('invalid state')
        self.resolv_cli.remote_addr = os.environ.get('B0_RESOLVER', 'tcp://127.0.0.1:22000')
        self.resolv_cli.init() # this socket is not managed
        self.announce_node()
        for socket in self.sockets: socket.init()
        self.state = 'ready'
    def announce_node(self):
        rep, reptype = self.resolv_call({'announce_node': {'node_name': self.name}})
        if not rep['ok']: raise RuntimeError('announce node failed')
        self.xpub_addr = rep['xpub_sock_addr']
        self.xsub_addr = rep['xsub_sock_addr']
        self.name = rep['node_name']
    def send_heartbeat(self):
        rep, reptype = self.resolv_call({'heartbeat': {'node_name': self.name}})
    def spin_once(self):
        if self.state != 'ready': raise RuntimeError('invalid state')
        for socket in self.sockets: socket.spin_once()
    def spin(self, rate = 10):
        while not self.shutdown_flag:
            self.spin_once()
            self.send_heartbeat()
            time.sleep(1. / rate)
    def cleanup(self):
        if self.state != 'ready': raise RuntimeError('invalid state')
        self.state = 'terminated'
        self.notify_shutdown()
        for socket in self.sockets: socket.cleanup()
    def notify_shutdown(self):
        rep, reptype = self.resolv_call({'shutdown_node': {'node_name': self.name}})

def run_pub():
    node = Node('python-publisher-node')
    pub = Publisher(node, 'A')
    node.init()
    while 1:
        pub.publish({'a': 1, 'b': 2}, 'foo')
        time.sleep(0.1)
    node.cleanup()

def run_sub():
    def cb(msg, msgtype):
        print(msg, msgtype)
    node = Node('python-subscriber-node')
    sub = Subscriber(node, 'A', cb)
    node.init()
    node.spin()
    node.cleanup()

def run_cli():
    node = Node('python-service-client')
    cli = ServiceClient(node, 'control')
    node.init()
    print(cli.call({'msg': 'hello', 'number': 23}, 'foo'))
    node.cleanup()

def run_srv():
    def cb(msg, msgtype):
        return {'answer': 42}, 'TheAnswer'
    node = Node('python-service-server')
    srv = ServiceServer(node, 'control', cb)
    node.init()
    node.spin()
    node.cleanup()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: %s <task_name>' % sys.argv[0])
        sys.exit(1)
    globals()['run_%s' % sys.argv[1]](*sys.argv[2:])

