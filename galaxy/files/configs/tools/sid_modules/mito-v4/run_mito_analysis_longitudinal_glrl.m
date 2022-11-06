function run_mito_analysis_longitudinal_glrl(folderName, maskFolder)

addpath(fullfile(pwd, 'GLRL'));
[montage_image, names] = getfilelist(folderName, 'tif');
N = length(montage_image);

[~, csv_dir, timeStamp] = createoutputfolders(folderName);

csvName = fullfile(csv_dir, 'all_wells_data_glrl.csv');

[~, completed] = getfilelist(csv_dir, 'csv');
completed = cellfun(@(A) strtok(A, '_'), completed, 'UniformOutput', false);

for k = 1:N % per well at this point
    
    % check if processed already
    [~, tmp] = strtok(names{k}, '_');
    for iii = 1:4
        [a, tmp] = strtok(tmp, '_');
    end
    if sum(ismember(completed, {a}))>0
        fprintf('Skipping : %s\n', names{k});
    else
        fprintf('Image : %s\n', names{k});
        fprintf('processing %d of %d\n', k, N);
        
        % image : PID20150430_AliciaRobo4TestPlate9_T0_0_A1_MONTAGE_488Empty525
        % mask  : PID20150430_AliciaRobo4TestPlate9_STACK_ALIGNED_A1_MONTAGE_MASK.tif
        
        id = strfind(names{k}, '_');
        current_well = names{k}(id(4)+1:id(5)-1);
        mask_name = fullfile(maskFolder, [names{k}(1:id(2)) 'STACK_ALIGNED' names{k}(id(4):id(5)) 'MONTAGE_MASK.tif']);
        
        % we can have image stacks here due to longitudinal imaging data
        numTimeSteps = length( imfinfo(montage_image{k}) );
        
        if numTimeSteps ~= length( imfinfo(mask_name) )
            numTimeSteps = 1; % somehow we don't have masks and montages for all time steps (or they don't match up...)
        end
        
        feats = zeros(numTimeSteps, 11);
        timeSteps = cell(numTimeSteps, 1);
        for timeStep = 1:numTimeSteps
            timeSteps{timeStep} = sprintf('T%d', timeStep);
            im = imread(montage_image{k}, timeStep);
            labels = imread(mask_name, timeStep);
            numObjects = numel(unique(labels(labels>0)));
            fprintf('Found %d ROIs at time %d\n', numObjects, timeStep);
            BW_LABEL = labels>0;
            
            tmp_feats = zeros(numObjects, 11);
            
            % now read each ROI
            cell_num = cell(numObjects, 1);
            for objNum = 1:numObjects
                fprintf('\tanalyzing roi #%d\n', objNum);
                cell_num{objNum} = [current_well '_' num2str(objNum)];
                [bw_mask, neuron] = extractcomponent(BW_LABEL, objNum, 'FilledImage', 'Image', im, 'Padding', 16);
                imn = mat2gray(neuron);
                
                Y = fft2(imn, 512, 512);
                [GLRLMS1, ~] = grayrlmatrix(abs(fftshift(Y)), 'NumLevels', 256, 'G', []);
                
                stats = grayrlprops(GLRLMS1); % "stats" is a 4x11 matrix
                tmp_feats( objNum, : ) = mean(stats,1); % calculate the average of the four direction texture metric
                
            end % end for objNum loop
            
            % single time point average for current well:
            tmp_feats = mean(tmp_feats,1);
            feats(timeStep,:) = tmp_feats;
        end % end timeStep loop
        
        tmp_name = fullfile(csv_dir, [current_well '_glrl.csv']);
        
        % write data from all time steps from one well
        appendGlrlToCSV(tmp_name, current_well, timeSteps, feats);
        
        % write data from this well to the global csv file that has all the plate data
        appendGlrlToCSV(csvName, current_well, timeSteps, feats);
        
    end % end if sum(ismsmber())
end % end k=1:N loop

fp = fopen(fullfile(csv_dir, 'README'), 'wt');
if fp>0
    fprintf(fp, 'Results from FFT and Texture based analysis using Gray Level Run Length Matrix\n');
    fclose(fp);
end


