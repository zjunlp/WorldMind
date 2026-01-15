include(config.pri)

QT -= core
QT -= gui

TARGET = bubbleRobClient_b0RemoteApi
TEMPLATE = app

DEFINES -= UNICODE
CONFIG   += console
CONFIG   -= app_bundle

INCLUDEPATH += $$BOOST_INCLUDEPATH
INCLUDEPATH += $$B0_INCLUDEPATH/b0/bindings
INCLUDEPATH += ..
INCLUDEPATH += ../msgpack-c/include

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
}

win32 {
    LIBS += $$B0_LIB
    LIBS += -L$$BOOST_LIB_PATH
}

macx {
    LIBS += $$B0_LIB
}

unix:!macx {
    LIBS += $$B0_LIB
    LIBS += -lboost_system
}

SOURCES += \
    bubbleRobClient.cpp \
    ../b0RemoteApi.cpp \

HEADERS +=\
    ../b0RemoteApi.h \

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}


















