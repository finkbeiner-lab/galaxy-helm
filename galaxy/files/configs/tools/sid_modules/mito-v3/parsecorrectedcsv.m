% PARSECORRECTEDCSV Parses the CSV file that contains corrected tracks
% This function parses the corrected survival CSV file and returns the
% object labels found in the csv file per well. The output is a structure
% with fields corresponding to the wells. 
% For example:
% objectLabels = 
%     A1: [1 2 3 4 5 6 8 9 10 11 12 13 15]
%    A10: [4 5 7 9 11 13 14 15]
%    A11: [1 2 3 5 6 8 9 10 11 12 14 15 16 17 20]
%    A12: [1 4 7 8 12]
%
% Author: Siddharth Samsi
% 06/05/2015
function objectLabels = parsecorrectedcsv(filename)

objectLabels = struct();
fp = fopen(filename, 'rt');
if fp<0
    return;
end

% read entire file into one cell array
data = textscan(fp, '%s');
fclose(fp);

data = data{1};
data = data(2:end);

y = cellfun(@(IN) textscan(IN, '%s', 3, 'Delimiter', ','), data, 'UniformOutput', false);
y = cellfun(@(A) A{1}(2:3), y, 'UniformOutput', false);

for k = 1:length(y)
    
    if isfield(objectLabels, y{k}{1})
        objectLabels.(y{k}{1}) = cat( 2, objectLabels.(y{k}{1}), str2double(y{k}{2}) );
    else
        objectLabels.(y{k}{1}) = str2double(y{k}{2}) ;
    end
end

% remove duplicates
fields = fieldnames(objectLabels);
for k = 1:length(fields)
    objectLabels.(fields{k}) = unique( objectLabels.(fields{k}) );
end

