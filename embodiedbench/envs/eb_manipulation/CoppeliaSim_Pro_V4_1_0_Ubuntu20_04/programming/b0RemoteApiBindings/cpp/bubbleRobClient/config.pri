# location of boost headers
#    BOOST_INCLUDEPATH = "d:/coppeliaRobotics/programming/vcpkg/installed/x64-windows/include" # (e.g. Windows)
    
# Boost libraries to link:
#    BOOST_LIB_PATH = "D:/coppeliaRobotics/programming/vcpkg/installed/x64-windows/lib" # (e.g. Windows)

# location of lua headers
#    LUA_INCLUDEPATH = "d:/lua-5.1.5/src"

# lua libraries to link
#    LUA_LIBS = "d:/lua-5.1.5/src/lua51.lib" # (e.g. Windows)

# jdk location
#    JDK_DIR = "C:/Program Files/Java/jdk1.7.0_40" # (e.g. Windows)

# jdk header path
#    JDK_INCLUDEPATH = "$${JDK_DIR}/include" "$${JDK_DIR}/include/win32" # (e.g. Windows)

# location of B0 headers:
#    B0_INCLUDEPATH = "d:/coppeliaRobotics/programming/blueZero/include" # (e.g. Windows)

# B0 libraries to link:
#    B0_LIB = "d:/coppeliaRobotics/programming/blueZero/build/Release/b0.lib" # (e.g. Windows)
#    B0_LIB_STATIC = "d:/coppeliaRobotics/programming/blueZero/build/Release/b0-static.lib" # (e.g. Windows)

#ZMQ:
#    ZMQ_LIB = "D:\coppeliaRobotics\programming\vcpkg\installed\x64-windows\lib\libzmq-mt-4_3_1.lib" # (e.g. Windows)   
    
# ZLIB:    
#    ZLIB_LIB = "D:\coppeliaRobotics\programming\vcpkg\installed\x64-windows\lib\zlib.lib" # (e.g. Windows)

# qscintilla location:
#    QSCINTILLA_DIR = "d:/QScintilla_commercial-2.10.8" # (e.g. Windows)

# qscintilla headers:
#    QSCINTILLA_INCLUDEPATH = "$${QSCINTILLA_DIR}/include" "$${QSCINTILLA_DIR}/Qt4Qt5" # (e.g. Windows)

# qscintilla libraries to link:
#    QSCINTILLA_LIBS = "$${QSCINTILLA_DIR}/build-qscintilla-Desktop_Qt_5_9_0_MSVC2017_64bit2-Release/release/qscintilla2_qt5.lib" # (e.g. Windows)
    
# Make sure if a config.pri is found one level above, that it will be used instead of this one:
    exists(../config.pri) { include(../config.pri) }

