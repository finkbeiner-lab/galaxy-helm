%
% Normal usage : 
% appendToCSV(csvName, currentWell, timeStep, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count, varargin)
%
% To save individual neuron data :
% appendToCSV(csvName, currentWell, timeStep, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count, objectID)
%
function appendToCSV(csvName, currentWell, timeStep, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count, varargin)

if isempty(varargin{1})
    headers = {'Well', 'Time', 'Circularity', 'SkeletonLength', 'Eccentricity', 'AspectRatio', 'Solidity', 'BlobCount'};
else
    headers = {'Well', 'ObjectLabelsFound', 'Time', 'Circularity', 'SkeletonLength', 'Eccentricity', 'AspectRatio', 'Solidity', 'BlobCount'};
end

if ~exist(csvName, 'file')
    mode = 'wt'; % create for writing
    fp = fopen(csvName, mode);
    if fp<0
        fprintf(2, 'Unable to open csv file for writing : %s\n', csvName);
        return;
    end
    % write headers
    fprintf(fp, '%s', headers{1});
    fprintf(fp, ',%s', headers{2:end});
    fprintf(fp, '\n');
    fclose(fp);
end
mode = 'at'; % always append

fp = fopen(csvName, mode);
if fp<0
    fprintf(2, 'Unable to open csv file for writing : %s\n', csvName);
    return;
end

% now write the rest of the data
% note: currentWell and timeStep are single values
n = numel(circularity);
for k = 1:n
    fprintf(fp, '%s, ', currentWell);
    
    % "varargin" will contain individual neuron metrics
    if ~isempty(varargin{1})
        fprintf(fp, '%d, ', varargin{1}(k)); 
        % because when we write per neuron data, per time step, we only have one entry in this cell that has data
        % varargin{2} will contain the correct index
        fprintf(fp, '%s, ', timeStep{varargin{2}}); 
    else  
        fprintf(fp, '%s, ', timeStep{varargin{2}});
    end
    
    fprintf(fp, '%f, ', circularity(k));
    fprintf(fp, '%f, ', skel_len(k));
    fprintf(fp, '%f, ', eccentricity(k));
    fprintf(fp, '%f, ', aspect_ratio(k));
    fprintf(fp, '%f, ', solidity(k));
    fprintf(fp, '%f ', blob_count(k));
    fprintf(fp, '\n');
end

fclose(fp);
