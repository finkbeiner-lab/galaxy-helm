% CREATEOUTPUTFOLDERS
% [skel_dir, csv_dir, timeStamp] = createoutputfolders(folderName)
%
function [skel_dir, csv_dir, timeStamp] = createoutputfolders(folderName)

timeStamp = datestr(now, 30);

skel_dir = fullfile(fileparts(folderName), 'results', timeStamp, 'skeleton-images');
csv_dir = fullfile(fileparts(folderName), 'results', timeStamp, 'csv');

mymakedir(skel_dir, timeStamp);
mymakedir(csv_dir, timeStamp);

function mymakedir(dirname, altname)
[success, message] = mkdir(dirname);
if ~success
    fprintf(2, 'Unable to create output folder : %s\n', dirname);
    fprintf(2, 'Cause of error \n: %s\n', message);
    fprintf(2, 'Attempting to create output folder again\n');
    [success, message] = mkdir(altname);
    if ~success
        fprintf(2, 'Unable to create output direcory\nCause : %s\n\nExiting\n', message);
        return;
    end
end

