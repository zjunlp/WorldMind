QT -= core
QT -= gui

TARGET = remoteApi
TEMPLATE = lib

DEFINES -= UNICODE
DEFINES += QT_COMPIL
DEFINES += NON_MATLAB_PARSING
DEFINES += MAX_EXT_API_CONNECTIONS=255

CONFIG += shared
INCLUDEPATH += "../../include"
INCLUDEPATH += "../../remoteApi"

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
#    LIBS += -lrt
}

unix:!macx {
    LIBS += -lrt
}

SOURCES += \
    ../../remoteApi/extApi.c \
    ../../remoteApi/extApiPlatform.c \
    ../../common/shared_memory.c \

HEADERS +=\
    ../../remoteApi/extApi.h \
    ../../remoteApi/extApiPlatform.h \
    ../../remoteApi/extApiInternal.h \
    ../../include/shared_memory.h \

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

