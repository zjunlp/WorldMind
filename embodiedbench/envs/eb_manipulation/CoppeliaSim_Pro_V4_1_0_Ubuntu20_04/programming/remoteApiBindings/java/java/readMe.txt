Make sure you have following files in your directory, in order to run the various examples:

1. folder "coppelia"
3. the appropriate remote API library: "remoteApiJava.dll" (Windows), "libremoteApi.dylib" (Mac) or "libremoteApi.so" (Linux)
4. simpleTest.java (or any other example program)

You might also have to add the folder to the system path. In Linux for instance, you could call:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`

before executing simpleTest.

For more examples, have a look at the python folder: language is different but principles are the same
