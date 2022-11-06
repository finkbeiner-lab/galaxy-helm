%
% run_mito_analysis_longitudinal(montageFolder, maskFolder, correctedTrackFile, writeSkeletonImages)
%
% Author: Siddharth Samsi
% 06/05/2015
function [csv_dir, timeStamp] = run_mito_analysis_longitudinal_parallel(montageFolder, maskFolder, varargin)

opts = {[],0};
opts(1:length(varargin)) = varargin(:);
[correctedTrackFile, writeSkeletonImages] = opts{:};

if writeSkeletonImages
    fprintf('Skeleton images will be saved\n');
else
    fprintf('Skeleton images will not be saved\n');
end

addpath(fullfile(pwd, 'frangi_filter_version2a'));
[montage_image, names] = getfilelist(montageFolder, 'tif');
% pull out the list of wells in this dataset
wellStr = listwells(names); % NOTE: wellStr contains wells in this format : _A1_

[maskFileNames, ~] = getfilelist(maskFolder, 'tif');

numberOfImages = length(montage_image);


% vesselness, multi-threshold and morphology based approach
numSteps = 10;

% bookkeeping...
[skel_dir, csv_dir, timeStamp] = createoutputfolders(montageFolder);
csvTmpName = fullfile(csv_dir, 'all_wells.csv');
perNeuronCSVfile = fullfile(csv_dir, 'all_neurons.csv');

% chek to see if we have the CSV file with corrected tracks
correctedData = false;
if ~isempty(correctedTrackFile)
    % objectLabels is a struct with well names as the fields:
    objectLabels = parsecorrectedcsv(correctedTrackFile);    
    if ~isempty( fieldnames(objectLabels) )
        correctedData = true;
    end
end

fprintf(2, 'Parallel version still experimental\n'));
parfor wellNumber = 1:numberOfImages % process per well
    numTimeSteps = length(imfinfo(montage_image{wellNumber}));
    
    well_skel_len = zeros(numberOfImages, numTimeSteps);
    well_circularity = zeros(numberOfImages, numTimeSteps);
    well_solidity = zeros(numberOfImages, numTimeSteps);
    well_aspect_ratio = zeros(numberOfImages, numTimeSteps);
    well_eccentricity = zeros(numberOfImages, numTimeSteps);
    well_blob_count = zeros(numberOfImages, numTimeSteps);

    fprintf('Image : %s\n', names{wellNumber});
    fprintf('processing %d of %d\n', wellNumber, numberOfImages);
    
    % image : PID20150430_AliciaRobo4TestPlate9_T0_0_A1_MONTAGE_488Empty525
    % mask  : PID20150430_AliciaRobo4TestPlate9_STACK_ALIGNED_A1_MONTAGE_MASK.tif
    
    current_well = wellStr{wellNumber}(2:end-1);
    skelNamePrefix = fullfile(skel_dir, current_well);
    
    mask_name = getmaskname(maskFileNames, wellStr{wellNumber});    
    % we can have image stacks here when analyzing longitudinal imaging data
    numTimeSteps = length( imfinfo(montage_image{wellNumber}) );
    
    if numTimeSteps ~= length( imfinfo(mask_name) )
        % somehow we don't have masks and montages for all time steps (or they don't match up...)
        % in that case just analyze the first image in the multi-page tiff
        numTimeSteps = 1; 
    end
    
    timeStampForCSV = cell(numTimeSteps, 1);
    for timeStep = 1:numTimeSteps
        timeStampForCSV{timeStep} = sprintf('T%d', timeStep);
        im = imread(montage_image{wellNumber}, timeStep);
        labels = imread(mask_name, timeStep);
        
        % see if we have corrected data. if yes, only keep the neurons that
        % are in the corrected data set
        if correctedData && isfield( objectLabels, wellStr{wellNumber}(2:end-1) )
            % this will retain only the corrected neurons:
            objectIDs = objectLabels.(wellStr{wellNumber}(2:end-1));                        
        else
            % this will give all the neurons:
            objectIDs = unique(labels(labels>0));
        end
        
        objectIDs = objectIDs(:);
        
        % objects may disappear over time. so we need to further filter
        % down the list of object IDs 
        objectIDsInImage = unique(labels(labels>0));
        objectIDs = intersect(objectIDs, objectIDsInImage);
        
        numObjects = numel(objectIDs);
        fprintf('Found %d ROIs at time %d\n', numObjects, timeStep);
        
        % following arrays will contain average data per well at a given
        % time point. these are overwritten at each time step
        skel_len = zeros(numObjects,1);
        circularity = zeros(numObjects,1);
        solidity = zeros(numObjects,1);
        aspect_ratio = zeros(numObjects,1);
        eccentricity = zeros(numObjects,1);
        blob_count = zeros(numObjects,1);
        %%%
        
        % now read each ROI
        cell_num = cell(numObjects, 1);
        for objNum = 1:numObjects
            fprintf('\tanalyzing roi #%d\n', objectIDs(objNum));
            
            cell_num{objNum} = [current_well '_' num2str( objectIDs(objNum) )];
            
            BW_TMP = ismember(labels, objectIDs(objNum)); % this will only have one object we are interested in 
            [bw_mask, neuron] = extractcomponent(BW_TMP, 1, 'FilledImage', 'Image', im, 'Padding', 16);
            
            imn = mat2gray(neuron);
            [blobs_final, vesselness, ~, ~, ~, ~] = mito_analysis_blobs_lines(imn, bw_mask, false);
            v = mat2gray(vesselness);
            
            mean_v = mean(v(:));
            std_v = std(v(:));
            thresholds = linspace(0, mean_v+std_v, numSteps+1);
            thresholds = thresholds(2:end);
            nn = length(thresholds);
            
            % follosing arrays will contain morphological features for each
            % object over 10 different thrsholds. these are overwritten for
            % each new object being processed
            aspectRatio_obj = zeros(nn,1);
            eccentricity_obj = aspectRatio_obj; 
            circularity_obj = aspectRatio_obj; 
            solidity_obj = aspectRatio_obj; 
            perimeter_obj = aspectRatio_obj;
            %%%
            
            [numRows, numCols, ~] = size(imn);
            skelMontage = false(2*numRows, numCols*nn);
            for ii = 1:nn
                BW = v > thresholds(ii);
                BW = bwareaopen(BW, 3); % remove isolated pixels ?
                BW(~bw_mask) = 0; % set region outside the mask to 0
                
                cc = bwconncomp(BW);
                props = regionprops(cc, 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', ...
                    'EquivDiameter', 'Area', 'Perimeter', 'Solidity');
                
                aspectRatio_obj(ii) = mean([props.MajorAxisLength]./[props.MinorAxisLength]); % aspect ratio
                eccentricity_obj(ii) = mean([props.Eccentricity]); % eccentricity
                circularity_obj(ii) = mean([props.Perimeter].^2./(4*pi*[props.Area])); % circularity
                solidity_obj(ii) = mean([props.Solidity]); % solidity
                
                sk = bwmorph(BW, 'skel', Inf);
                skelMontage( 1:numRows, 1+(ii-1)*numCols:ii*numCols) = BW;
                skelMontage( numRows+1:end, 1+(ii-1)*numCols:ii*numCols) = sk;
                props1 = regionprops(sk, 'Perimeter');
                perimeter_obj(ii) = mean([props1.Perimeter]); % could help distinguish between long and short segments
            end
            
            if writeSkeletonImages
                skelName = [skelNamePrefix '_' num2str(objectIDs(objNum)) '_' num2str(ii) '_T' num2str(timeStep) '.tif'];
                writeskeleton(imn, skelMontage, skelName);
            end
            
            % calculate average stats for each neuron ******* AT THIS TIME POINT ********
            blob_count(objNum) = size(blobs_final,1); % this is based on grayscale not thresholded image, so only calculated once
            aspect_ratio(objNum) = mean(aspectRatio_obj);
            circularity(objNum) = mean(circularity_obj);
            solidity(objNum) = mean(solidity_obj);
            eccentricity(objNum) = mean(eccentricity_obj);
            skel_len(objNum) = mean(perimeter_obj);
            
        end % end for objNum loop        
        
        % we have data for all objects in given well at current time point
        % save it ?
        appendToCSV(perNeuronCSVfile, current_well, timeStampForCSV, circularity, skel_len, ...
            eccentricity, aspect_ratio, solidity, blob_count, objectIDs, timeStep);

        % calculate and recoed the average stats per well ****** AT THIS TIME POINT ******
        well_skel_len(wellNumber, timeStep) = mean(skel_len);
        well_circularity(wellNumber, timeStep) = mean(circularity);
        well_solidity(wellNumber, timeStep) = mean(solidity);
        well_aspect_ratio(wellNumber, timeStep) = mean(aspect_ratio);
        well_eccentricity(wellNumber, timeStep) = mean(eccentricity);
        well_blob_count(wellNumber, timeStep) = mean(blob_count);
        
    end % end timeStep loop
    tmp_name = fullfile(csv_dir, [current_well '.csv']);    
    
    % append data to csv file that will have all wells
    % this will be problematic in parallel. commented out for now    
    %{
    appendToCSV(csvTmpName, current_well, timeStampForCSV, well_circularity(wellNumber,:), well_skel_len(wellNumber,:), ...
        well_eccentricity(wellNumber,:), well_aspect_ratio(wellNumber,:), ...
        well_solidity(wellNumber,:), well_blob_count(wellNumber,:));
    %}
    
    % write well data to it's own csv file    
    appendToCSV(tmp_name, current_well, timeStampForCSV, well_circularity(wellNumber,:), well_skel_len(wellNumber,:), ...
        well_eccentricity(wellNumber,:), well_aspect_ratio(wellNumber,:), ...
        well_solidity(wellNumber,:), well_blob_count(wellNumber,:));
    
    fprintf('Written data for well %s\n', current_well);
    fprintf('Data written to : %s\n', tmp_name);
end % end k=1:N loop


fp = fopen(fullfile(csv_dir, 'README.txt'), 'wt');
if fp>0
    fprintf(fp, 'Results from vesselness filter and multiple thresholds\n');
    fclose(fp);
end

disp('done');

