% COMPILATION ON WINDOWS.

disp("The compiler expects to have all source files in one directory.")
disp("So copy and paste following files:")
disp("")
disp("- remote API source files (programming/remoteApi/*)")
disp("- include files (programming/include/*)")
disp("")
disp("Then do something like:")
disp("")
disp('"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat x64"')
disp("mkoctfile -DMAX_EXT_API_CONNECTIONS=255 -DNON_MATLAB_PARSING -DDO_NOT_USE_SHARED_MEMORY -lwinmm -lWs2_32 remApi.cc extApi.c extApiPlatform.c")
disp("")
disp("(you might have to adjust this file manually)")
disp("")
disp("Press any key to start")
disp("")
pause()

"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat x64"
mkoctfile -DMAX_EXT_API_CONNECTIONS=255 -DNON_MATLAB_PARSING -DDO_NOT_USE_SHARED_MEMORY -lwinmm -lWs2_32 remApi.cc extApi.c extApiPlatform.c


