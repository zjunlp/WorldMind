To automatically generate the bindings and html doc from the 'simxFunctions.xml' file, use for instance:

python generate.py --gen-simx-all --xml-file simxFunctions.xml ./generated

If you added your own functions to 'simxFunctions.xml', then do not forget to add the
server counterpart of them at the beginning of file 'lua/b0RemoteApiServer.lua'
