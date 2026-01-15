-- this add-on simply opens a file dialog, then outputs the name of the selected text file to the status bar and console.
-- This is not very useful, but illustrates how easily CoppeliaSim can be customized using add-ons.
-- Add-on functions can easily be used to write importers/exporters

selectedFile=sim.fileDialog(0,"Please select a text file","","","Text file","txt") -- open the file dialog in "open file" mode

if selectedFile then
	msg="You selected the file: "..selectedFile
else
	msg="You didn't select any file"
end

sim.addStatusbarMessage(msg) -- print to the statusbar
print(msg) -- print to the console
