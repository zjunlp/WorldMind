% COMPILATION ON MAC. HERE SOME USEFUL INFOS:

disp("Install via Fink or homebrew.")
disp("")
disp("Read more here if needed:")
disp("")
disp("http://www.gnu.org/software/octave/download.html")
disp("http://wiki.octave.org/Octave_for_MacOS_X") 
disp("http://ntraft.com/getting-octave-to-work-on-mountain-lion/")
disp("http://stackoverflow.com/questions/16445098/how-can-i-install-gnu-octave-on-mac-with-fink")
disp("http://octave.1599824.n4.nabble.com/missing-mkoctfile-3-6-3-on-OS-X-td4647526.html")
disp("http://www.weescribble.com/technology-menu/138-fixing-fontconfig-warning-for-octave-on-osx")
disp("")
disp("The compiler expects to have all source files in this directory. So copy and paste following files:")
disp("- remote API source files (programming/remoteApi/*)")
disp("- include files (programming/include/*)")
disp("")
disp('Then, in this directory, from the octave console, type "buildMac"')
disp("")
disp("Press any key to start")
disp("")
pause()

mkoctfile -DMAX_EXT_API_CONNECTIONS=255 -DNON_MATLAB_PARSING -DDO_NOT_USE_SHARED_MEMORY remApi.cc extApi.c extApiPlatform.c 
