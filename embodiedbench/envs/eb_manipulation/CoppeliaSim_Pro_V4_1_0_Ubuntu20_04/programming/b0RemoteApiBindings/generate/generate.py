import argparse
import os
import os.path
import errno
import re
import sys
import subprocess

from parse import parse
import model

parser = argparse.ArgumentParser(description='Generate various things for CoppeliaSim plugin.')
parser.add_argument('output_dir', type=str, default=None, help='the output directory')
parser.add_argument('--xml-file', type=str, default='callbacks.xml', help='the XML file with the callback definitions')
parser.add_argument("--verbose", help='print commands being executed', action='store_true')

parser.add_argument("--gen-simx-stubs", help='generate stubs for the BlueZero based remote API (C++, Java, Lua, Matlab, Python)', action='store_true')
parser.add_argument("--gen-simx-docs", help='generate docs for the BlueZero based remote API (C++, Java, Lua, Matlab, Python and API list)', action='store_true')
parser.add_argument("--gen-simx-all", help='generate everything related to the BlueZero based remote API', action='store_true')
args = parser.parse_args()

if args is False:
    SystemExit

args.verbose = True

if args.verbose:
    print(' '.join(['"%s"' % arg if ' ' in arg else arg for arg in sys.argv]))

self_dir = os.path.dirname(os.path.realpath(__file__))

def output(filename):
    return os.path.join(args.output_dir, filename)

def rel(filename):
    return os.path.join(self_dir, filename)

def runsubprocess(what, cmdargs):
    if args.verbose:
        print(' '.join(['"%s"' % arg if ' ' in arg else arg for arg in cmdargs]))
    child = subprocess.Popen(cmdargs)
    child.communicate()
    if child.returncode != 0:
        print('failed to run %s' % what)
        sys.exit(1)

def runtool(what, *cmdargs):
    runsubprocess(what, ['python', rel(what + '.py')] + list(cmdargs))

def runprogram(what, *cmdargs):
    runsubprocess(what, [what] + list(cmdargs))

# check dependencies & inputs:
input_xml = args.xml_file

# create output dir if needed:
try:
    os.makedirs(args.output_dir)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(args.output_dir):
        pass

plugin = parse(args.xml_file)

if args.gen_simx_stubs or args.gen_simx_all:
    os.environ['remoteApiDocLang'] = 'cpp'
    for ext in ['cpp', 'h']:
        runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/cpp/stubs.' + ext), '-o', output('b0RemoteApi.' + ext), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'java'
    for ext in ['java']:
        runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/java/stub.' + ext), '-o', output('b0RemoteApi.' + ext), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'lua'
    for ext in ['lua']:
        runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/lua/stub.' + ext), '-o', output('b0RemoteApi.' + ext), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'matlab'
    for ext in ['m']:
        runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/matlab/stub.' + ext), '-o', output('b0RemoteApi.' + ext), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'python'
    for ext in ['py']:
        runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/python/stub.' + ext), '-o', output('b0RemoteApi.' + ext), '-P', self_dir)
        
if args.gen_simx_docs or args.gen_simx_all:
    os.environ['remoteApiDocLang'] = 'cpp'
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc.htm'), '-o', output('b0RemoteApi-cpp.htm'), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'java'
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc.htm'), '-o', output('b0RemoteApi-java.htm'), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'lua'
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc.htm'), '-o', output('b0RemoteApi-lua.htm'), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'matlab'
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc.htm'), '-o', output('b0RemoteApi-matlab.htm'), '-P', self_dir)

    os.environ['remoteApiDocLang'] = 'python'
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc.htm'), '-o', output('b0RemoteApi-python.htm'), '-P', self_dir)
        
    runtool('pycpp', '-p', 'xml_file=' + args.xml_file, '-i', rel('simxStubs/doc-list.htm'), '-o', output('b0RemoteApi-functionList.htm'), '-P', self_dir)
