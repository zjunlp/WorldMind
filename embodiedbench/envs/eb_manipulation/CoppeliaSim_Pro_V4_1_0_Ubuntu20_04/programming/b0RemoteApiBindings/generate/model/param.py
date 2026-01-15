import os
class Param(object):
    mapping = {}

    def __init__(self, node):
        if node.tag != 'param':
            raise ValueError('expected <param>, got <%s>' % node.tag)
        self.name = node.attrib['name']
        try:
            descnode = node.find('description')
            self.description = '' if descnode is None else descnode.text
        except AttributeError:
            self.description = ''
        self.dtype = node.attrib['type']
        self.ctype_base = self.dtype
        self.default = node.attrib.get('default', None)
        self.skip = node.attrib.get('skip', 'false').lower() in ('true', 'yes', '1')
        self.nullable = node.attrib.get('nullable', 'false').lower() in ('true', 'yes', '1')
        self.write_in = True
        self.write_out = True
        if self.dtype == 'table' and self.nullable:
            raise ValueError('cannot have nullable table')

    def mandatory(self):
        return self.default is None

    def optional(self):
        return self.default is not None

    def ctype(self):
        if self.nullable: return 'boost::optional<%s>' % self.ctype_base
        else: return self.ctype_base

    def ctype_normalized(self):
        replacements = {
            '::': '__',
            '<': '__',
            '>': '__',
            ' ': '',
        }
        ret = self.ctype()
        for a, b in replacements.items():
            ret = ret.replace(a, b)
        return ret

    def htype(self):
        return self.dtype

    def cdefault(self):
        return self.default

    def hdefault(self):
        return self.default

    def argmod(self):
        return ''

    @staticmethod
    def register_type(dtype, clazz):
        Param.mapping[dtype] = clazz

    @staticmethod
    def factory(node):
        dtype = node.attrib['type']
        if dtype not in Param.mapping:
            print('ERROR: type "{}" not found in mapping; valid types are: {}'.format(dtype, ', '.join('"%s"' % k for k in Param.mapping.keys())))
        return Param.mapping[dtype](node)

    def declaration(self):
        return '{} {}'.format(self.ctype(), self.name)

    def declaration_with_default(self):
        return self.declaration() + (' = {}'.format(self.cdefault()) if self.cdefault() else '')

class ParamInt(Param):
    def __init__(self, node):
        super(ParamInt, self).__init__(node)

    def htype(self):
        return 'number'

class ParamFloat(Param):
    def __init__(self, node):
        super(ParamFloat, self).__init__(node)

    def htype(self):
        return 'number'

class ParamDouble(Param):
    def __init__(self, node):
        super(ParamDouble, self).__init__(node)

    def htype(self):
        return 'number'

class ParamString(Param):
    def __init__(self, node):
        super(ParamString, self).__init__(node)
        self.ctype_base = 'std::string'

    def cdefault(self):
        if self.default is None: return None
        return '"%s"' % self.default.replace('\\','\\\\').replace('"','\\"')

    def hdefault(self):
        if self.default is None: return None
        return "'%s'" % self.default.replace('\\','\\\\').replace('"','\\"')

        
class ParamBool(Param):
    def __init__(self, node):
        super(ParamBool, self).__init__(node)

class ParamAny(Param):
    def __init__(self, node):
        super(ParamAny, self).__init__(node)
        
class ParamTable(Param):
    def __init__(self, node):
        super(ParamTable, self).__init__(node)
        self.itype = node.attrib.get('item-type', None)
        self.minsize = int(node.attrib.get('minsize', 0))
        self.maxsize = int(node.attrib['maxsize']) if 'maxsize' in node.attrib else None
        if 'size' in node.attrib:
            self.minsize = int(node.attrib['size'])
            self.maxsize = int(node.attrib['size'])
        if self.itype is None:
            self.write_in = False
            self.write_out = False

    def item_dummy(self):
        n = type('dummyNode', (object,), dict(tag='param', attrib={'name': 'dummy', 'type': self.itype}))
        return Param.factory(n)

    def ctype(self):
        if self.itype is not None:
            return 'std::vector<%s>' % self.item_dummy().ctype()
        else:
            return 'void *'

    def ctype_normalized(self):
        return self.item_dummy().ctype().replace('::', '__')

    def htype(self):
        return 'table' + ('_%d' % self.minsize if self.minsize else '')

    def cdefault(self):
        if self.default is not None:
            d = self.default
            d = 'boost::assign::list_of{}.convert_to_container<{} >()'.format(''.join(map(lambda x: '(%s)' % x.strip(), d.strip()[1:-1].split(','))), self.ctype())
            return d

class ParamStruct(Param):
    def __init__(self, node, name):
        super(ParamStruct, self).__init__(node)
        self.structname = name
        self.xoptional = False
        if self.default is not None:
            if self.default == '{}':
                self.xoptional = True
            else:
                raise ValueError('default value not supported in <struct>')

    def mandatory(self):
        return not self.xoptional

    def optional(self):
        return self.xoptional

    def cdefault(self):
        return None

    def argmod(self):
        return '&'


        
class Pbool(Param):
    def __init__(self, node):
        super(Pbool, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='lua':
            return 'boolean'
        return 'bool'

class Pint(Param):
    def __init__(self, node):
        super(Pint, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='lua':
            return 'number'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'number'
        if os.getenv('remoteApiDocLang')=='python':
            return 'number'
        return 'int'

class Plong(Param):
    def __init__(self, node):
        super(Plong, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='lua':
            return 'number'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'number'
        if os.getenv('remoteApiDocLang')=='python':
            return 'number'
        return 'long'

class Pfloat(Param):
    def __init__(self, node):
        super(Pfloat, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='lua':
            return 'number'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'number'
        if os.getenv('remoteApiDocLang')=='python':
            return 'number'
        return 'float'

class Pdouble(Param):
    def __init__(self, node):
        super(Pdouble, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='lua':
            return 'number'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'number'
        if os.getenv('remoteApiDocLang')=='python':
            return 'number'
        return 'double'

class Pstring(Param):
    def __init__(self, node):
        super(Pstring, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='cpp':
            return 'const char*'
        if os.getenv('remoteApiDocLang')=='java':
            return 'final String'
        return 'string'
        
class Pint_eval(Param):
    def __init__(self, node):
        super(Pint_eval, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='cpp':
            return 'int/(const char*)'
        if os.getenv('remoteApiDocLang')=='java':
            return 'int/String'
        return 'number/string'
        
class Pmap(Param):
    def __init__(self, node):
        super(Pmap, self).__init__(node)

    def htype(self):
        return 'map'
        
class Pbytea(Param):
    def __init__(self, node):
        super(Pbytea, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final byte[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'string'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'byteArray'
        if os.getenv('remoteApiDocLang')=='python':
            return 'byteArray'
        return 'byte[]'

class Pinta(Param):
    def __init__(self, node):
        super(Pinta, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final int[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'int[]'

class Pinta2(Param):
    def __init__(self, node):
        super(Pinta2, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final int[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'int[2]'

class Pinta3(Param):
    def __init__(self, node):
        super(Pinta3, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final int[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'int[3]'

class Pfloata(Param):
    def __init__(self, node):
        super(Pfloata, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[]'

class Pfloata2(Param):
    def __init__(self, node):
        super(Pfloata2, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[2]'
        
class Pfloata3(Param):
    def __init__(self, node):
        super(Pfloata3, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[3]'
        
class Pfloata4(Param):
    def __init__(self, node):
        super(Pfloata4, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[4]'
        
class Pfloata7(Param):
    def __init__(self, node):
        super(Pfloata7, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[7]'

class Pfloata12(Param):
    def __init__(self, node):
        super(Pfloata12, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final float[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'float[12]'

class Pdoublea(Param):
    def __init__(self, node):
        super(Pdoublea, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[]'

class Pdoublea2(Param):
    def __init__(self, node):
        super(Pdoublea2, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[2]'
        
class Pdoublea3(Param):
    def __init__(self, node):
        super(Pdoublea3, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[3]'
        
class Pdoublea4(Param):
    def __init__(self, node):
        super(Pdoublea4, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[4]'
        
class Pdoublea7(Param):
    def __init__(self, node):
        super(Pdoublea7, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[7]'

class Pdoublea12(Param):
    def __init__(self, node):
        super(Pdoublea12, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final double[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'table'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'array'
        if os.getenv('remoteApiDocLang')=='python':
            return 'list'
        return 'double[12]'

class Pcallback(Param):
    def __init__(self, node):
        super(Pcallback, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='cpp':
            return 'CB_FUNC'
        if os.getenv('remoteApiDocLang')=='java':
            return 'final Consumer<MessageUnpacker>'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'callback'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'callback'
        if os.getenv('remoteApiDocLang')=='python':
            return 'callback'
        return 'callback'
        
class Ppacked_unpacked(Param):
    def __init__(self, node):
        super(Ppacked_unpacked, self).__init__(node)

    def htype(self):
        if os.getenv('remoteApiDocLang')=='java':
            return 'final byte[]'
        if os.getenv('remoteApiDocLang')=='lua':
            return 'anyType'
        if os.getenv('remoteApiDocLang')=='matlab':
            return 'anyType'
        if os.getenv('remoteApiDocLang')=='python':
            return 'anyType'
        return 'packed_unpacked'
        
        
class Pany(Param):
    def __init__(self, node):
        super(Pany, self).__init__(node)

    def htype(self):
        return '?'









        
#Param.register_type('anything', Param)
#Param.register_type('int', ParamInt)
#Param.register_type('float', ParamFloat)
#Param.register_type('double', ParamDouble)
#Param.register_type('string', ParamString)
#Param.register_type('bool', ParamBool)
Param.register_type('table', ParamTable)


Param.register_type('bool', Pbool)
Param.register_type('int', Pint)
Param.register_type('long', Plong)
Param.register_type('float', Pfloat)
Param.register_type('double', Pdouble)
Param.register_type('string', Pstring)
Param.register_type('int_eval', Pint_eval)
Param.register_type('byte[]', Pbytea)
Param.register_type('int[]', Pinta)

Param.register_type('float[]', Pfloata)
Param.register_type('double[]', Pdoublea)
Param.register_type('int[2]', Pinta2)
Param.register_type('float[2]', Pfloata2)
Param.register_type('double[2]', Pdoublea2)
Param.register_type('int[3]', Pinta3)
Param.register_type('float[3]', Pfloata3)
Param.register_type('double[3]', Pdoublea3)
Param.register_type('float[4]', Pfloata4)
Param.register_type('double[4]', Pdoublea4)
Param.register_type('float[7]', Pfloata7)
Param.register_type('double[7]', Pdoublea7)
Param.register_type('float[12]', Pfloata12)
Param.register_type('double[12]', Pdoublea12)
Param.register_type('map', ParamAny)
Param.register_type('callback', Pcallback)
Param.register_type('?', Pany)
Param.register_type('packed_unpacked', Ppacked_unpacked)


