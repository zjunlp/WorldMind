This is the Ubuntu release V4.1.0, rev. 1, 64bit

****************************
****************************
FROM THE COMMAND LINE, run

$ ./coppeliaSim.sh 

to launch CoppeliaSim
****************************
****************************



**********************************
Various issues you might run into:
**********************************

1. When trying to start CoppeliaSim, following message displays: "Error: could not find or correctly load the CoppeliaSim library"
	a) Make sure you started CoppeliaSim with "./coppeliaSim.sh" FROM THE COMMAND LINE
	b) check what dependency is missing by using the file "libLoadErrorCheck.sh"

2. You are using a dongle license key, but CoppeliaSim displays 'No dongle was found' at launch time.
	a) See below



***************
Using a dongle:
***************

a) $ lsusb
b) Make sure that the dongle is correctly plugged and recognized (VID:1bc0, PID:8100)
c) $ sudo cp 92-SLKey-HID.rules /etc/udev/rules.d/
d) Restart the computer
e) $ ./coppeliaSim.sh

