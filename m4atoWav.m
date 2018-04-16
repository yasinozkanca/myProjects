% Put the m4a file directory to m4aDirectory and the output folder 
% name to wavFolder and just run!
m4aDirectory = 'C:\Users\yasin'
wavFolder = 'parkinsonWavs\'
directory = dir(m4aDirectory);
for i = 3:length(m4aDirectory)
   try 
   [y,Fs] = audioread(strcat(m4aDirectory,'\',directory(i).name));
    audiowrite(strcat(m4aDirectory,'\',wavFolder,directory(i).name,'.wav'),y,Fs)

   catch 
       disp('File %f cannot be changed to wav, something is not quite right',i-2)
       continue
   end
end