include(config.pri)

QT -= core
QT -= gui

TARGET = b0Lua
TEMPLATE = lib

DEFINES -= UNICODE
DEFINES += QT_COMPIL
CONFIG += shared

INCLUDEPATH += $$BOOST_INCLUDEPATH
INCLUDEPATH += $$B0_INCLUDEPATH/b0/bindings
INCLUDEPATH += ../../include
INCLUDEPATH += $$LUA_INCLUDEPATH
LIBS += $$LUA_LIBS
LIBS += $$B0_LIB
LIBS += -L$$BOOST_LIB_PATH

*-msvc* {
    QMAKE_CXXFLAGS += -O2
    QMAKE_CXXFLAGS += -W3
}
*-g++* {
    QMAKE_CXXFLAGS += -O3
    QMAKE_CXXFLAGS += -Wall
    QMAKE_CXXFLAGS += -Wno-unused-parameter
    QMAKE_CXXFLAGS += -Wno-strict-aliasing
    QMAKE_CXXFLAGS += -Wno-empty-body
    QMAKE_CXXFLAGS += -Wno-write-strings

    QMAKE_CXXFLAGS += -Wno-unused-but-set-variable
    QMAKE_CXXFLAGS += -Wno-unused-local-typedefs
    QMAKE_CXXFLAGS += -Wno-narrowing

    QMAKE_CFLAGS += -O3
    QMAKE_CFLAGS += -Wall
    QMAKE_CFLAGS += -Wno-strict-aliasing
    QMAKE_CFLAGS += -Wno-unused-parameter
    QMAKE_CFLAGS += -Wno-unused-but-set-variable
    QMAKE_CFLAGS += -Wno-unused-local-typedefs
}

win32 {
    LIBS += -lwinmm
    LIBS += -lWs2_32
    LIBS += -lKernel32
}

macx {
    LIBS += -lboost_system
}

unix:!macx {
    LIBS += -lboost_system
    LIBS += -lrt
}

SOURCES += \
    b0RemoteApiLua.cpp \


unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}
