%
%
function appendGlrlToCSV(csvName, currentWell, timeStep, features)

headers = {'Well', 'Time', 'SRE', 'LRE', 'GLN', 'RLN', 'RP', 'LGRE', 'HGRE', 'SRLGE', 'SRHGE', 'LRLGE', 'LRHGE'};

if ~exist(csvName, 'file')
    fp = fopen(csvName, 'wt'); % create for writing
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

fp = fopen(csvName, 'at');% always append
if fp<0
    fprintf(2, 'Unable to open csv file for writing : %s\n', csvName);
    return;
end

% now write the rest of the data
% note: currentWell and timeStep are single values
n = size(features, 1);
for k = 1:n
    fprintf(fp, '%s, ', currentWell);
    fprintf(fp, '%s, ', timeStep{k});
    
    % texture features
    fprintf(fp, '%f, ', features(k,:)); % this will add an extra comma at the end
    fprintf(fp, '\b\n'); % remove the extra comma and add newline character    
end

fclose(fp);
