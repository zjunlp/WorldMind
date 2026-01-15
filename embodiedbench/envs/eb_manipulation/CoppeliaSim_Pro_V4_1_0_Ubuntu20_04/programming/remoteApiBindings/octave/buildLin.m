% COMPILATION ON LINUX. HERE SOME USEFUL INFOS:

disp("Download and install Octave:")
disp("")
disp("$ sudo apt-add-repository ppa:octave/stable")
disp("$ sudo apt-get update")
disp("$ sudo apt-get install octave")
disp("")
disp("and optionally, in order to compile the oct file yourself:")
disp("")
disp("$ sudo apt-get install octave-pkg-dev")
disp("")
disp("The compiler expects to have all source files in this directory.")
disp("So copy and paste following files:")
disp("- remote API source files (programming/remoteApi/*)")
disp("- include files (programming/include/*)")
disp("")
disp('Then, in this directory, from the octave console, type "buildLin"')
disp("")
disp("Press any key to start")
disp("")
pause()

mkoctfile -DMAX_EXT_API_CONNECTIONS=255 -DNON_MATLAB_PARSING -DDO_NOT_USE_SHARED_MEMORY remApi.cc extApi.c extApiPlatform.c 
