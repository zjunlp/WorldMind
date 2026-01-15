@ECHO OFF

set VCVARSALL_STRING="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set COMPILER_FLAGS=/W3 /O1 /EHsc /c
set COMPILER_DEFINES=/D "WIN32" /D "_WINDOWS" /D "NDEBUG"
set COMPILER_INCLUDES=/I"..\..\..\blueZero\include\b0\bindings" /I".." /I"..\msgpack-c\include" /I"C:\local\boost_1_62_0"
set COMPILERSTRING=%COMPILER_INCLUDES% %COMPILER_DEFINES% %COMPILER_FLAGS%  

REM  b0.lib is normally located in the same folder as coppeliaSim.exe
set LINKER_LIBS="..\..\..\bluezero\build\release\b0.lib"
set LINKERSTRING=/OUT:"testing.exe" %LINKER_LIBS%

if not defined DevEnvDir (
    call %VCVARSALL_STRING%
)

cl testing.cpp %COMPILERSTRING%
cl ..\b0RemoteApi.cpp %COMPILERSTRING%
link testing.obj b0RemoteApi.obj %LINKERSTRING%
