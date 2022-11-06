function run_mito_analysis(folderName, maskFolder)

addpath(fullfile(pwd, 'frangi_filter_version2a'));
[montage_image, names] = getfilelist(folderName, 'tif');
N = length(montage_image);

% morphology and multi-threshold based approach
well_skel_len = zeros(N,1);
well_circularity = zeros(N,1);
well_solidity = zeros(N,1);
well_aspect_ratio = zeros(N,1);
well_eccentricity = zeros(N,1);
well_blob_count = zeros(N,1);
numSteps = 10;
timeStamp = datestr(now, 30);

skel_dir = fullfile(fileparts(folderName), 'results', timeStamp, 'skeleton-images');
csv_dir = fullfile(fileparts(folderName), 'results', timeStamp, 'csv');

[success, message] = mkdir(skel_dir);
if ~success
    fprintf(2, 'Unable to create output folder : %s\n', skel_dir);
    fprintf(2, 'Cause of error \n: %s\n', message);
    fprintf(2, 'Attempting to create output folder again\n');
    skel_dir = timeStamp; 
    [success, message] = mkdir(skel_dir);
    if ~success
        fprintf(2, 'Unable to create output direcory\nCause : %s\n\nExiting\n', message);
        return;
    end
end

[success, message] = mkdir(csv_dir);
if ~success
    fprintf(2, 'Unable to create output folder : %s\n', csv_dir);
    fprintf(2, 'Cause of error \n: %s\n', message);
    fprintf(2, 'Attempting to create output folder again\n');
    skel_dir = timeStamp; 
    [success, message] = mkdir(csv_dir);
    if ~success
        fprintf(2, 'Unable to create output direcory\nCause : %s\n\nExiting\n', message);
        return;
    end
end

for k = 1:N % this is processing per montaged image (so this is per well)
    fprintf('Image : %s\n', names{k});
    fprintf('processing %d of %d\n', k, N);
    % image : PID20150430_AliciaRobo4TestPlate9_T0_0_A1_MONTAGE_488Empty525
    % mask  : PID20150430_AliciaRobo4TestPlate9_STACK_ALIGNED_A1_MONTAGE_MASK.tif
    im = imread(montage_image{k});
    id = strfind(names{k}, '_');
    current_well = names{k}(id(4)+1:id(5)-1);
    skelNamePrefix = fullfile(skel_dir, names{k}(id(2)+1:id(5)-1));
    mask_name = fullfile(maskFolder, [names{k}(1:id(2)) 'STACK_ALIGNED' names{k}(id(4):id(5)) 'MONTAGE_MASK.tif']);
    labels = imread(mask_name);
    numObjects = numel(unique(labels(labels>0)));
    fprintf('Found %d ROIs\n', numObjects);
    BW_LABEL = labels>0;
        
    skel_len = zeros(numObjects,1);
    circularity = skel_len;
    solidity = skel_len;
    aspect_ratio = skel_len;
    eccentricity = skel_len;
    blob_count = skel_len;
    
    % now read each ROI
    cell_num = cell(numObjects, 1);
    for objNum = 1:numObjects
        fprintf('\tanalyzing roi #%d\n', objNum);
        cell_num{objNum} = [current_well '_' num2str(objNum)];
        [bw_mask, neuron] = extractcomponent(BW_LABEL, objNum, 'FilledImage', 'Image', im, 'Padding', 16);
        imn = mat2gray(neuron);
        [blobs_final, vesselness, ~, ~, ~, ~] = mito_analysis_blobs_lines(imn, bw_mask, false);
        v = mat2gray(vesselness);
                
        mean_v = mean(v(:));
        std_v = std(v(:));
        thresholds = linspace(0, mean_v+std_v, numSteps+1);
        thresholds = thresholds(2:end);
        nn = length(thresholds);
        
        a = zeros(nn,1);
        b = a; c = a; d = a; e = a;
        [numRows, numCols, ~] = size(imn);
        skelMontage = false(numRows, numCols*nn);
        for ii = 1:nn            
            BW = v > thresholds(ii);
            BW = bwareaopen(BW, 3); % remove isolated pixels ?
            
            cc = bwconncomp(BW);
            props = regionprops(cc, 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', ...
                'EquivDiameter', 'Area', 'Perimeter', 'Solidity');
            
            a(ii) = mean([props.MajorAxisLength]./[props.MinorAxisLength]); % aspect ratio
            b(ii) = mean([props.Eccentricity]); % eccentricity
            c(ii) = mean([props.Perimeter].^2./(4*pi*[props.Area])); % circularity
            d(ii) = mean([props.Solidity]); % solidity
            
            sk = bwmorph(BW, 'skel', Inf);
            skelMontage( :, 1+(ii-1)*numCols:ii*numCols) = sk;
            %skelName = [skelNamePrefix '_' num2str(objNum) '_' num2str(ii) '.tif'];
            %writeskeleton(imn, sk, skelName);
            props1 = regionprops(sk, 'Perimeter');
            e(ii) = mean([props1.Perimeter]); % could help distinguish between long and short segments
        end
        skelName = [skelNamePrefix '_' num2str(objNum) '.tif'];
        writeskeleton(imn, skelMontage, skelName);
        % calculate average stats for each neuron:
        blob_count(objNum) = size(blobs_final,1);
        aspect_ratio(objNum) = mean(a);
        circularity(objNum) = mean(c);
        solidity(objNum) = mean(d);
        eccentricity(objNum) = mean(b);
        skel_len(objNum) = mean(e);
        
    end % end for objNum loop

    
    resultFile = fullfile(csv_dir, sprintf('results-%s-%s.csv', current_well, timeStamp));
    writecsv(resultFile, cell_num, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count);
    fprintf('Written data for well %s\n', current_well);
    
    well_skel_len(k) = mean(skel_len);
    well_circularity(k) = mean(circularity);
    well_solidity(k) = mean(solidity);
    well_aspect_ratio(k) = mean(aspect_ratio);
    well_eccentricity(k) = mean(eccentricity);
    well_blob_count(k) = mean(blob_count);

end % end for k ... loop 


for k = 1:N
    id = strfind(names{k}, '_');
    names{k} = names{k}(id(4)+1:id(5)-1);
end
names = reshape(names, [numel(names) 1]);
resultFile = fullfile(csv_dir, sprintf('results-%s.csv', timeStamp));
fprintf('Writing to : %s\n', resultFile);
writecsv(resultFile, names, well_circularity, well_skel_len, well_blob_count, well_aspect_ratio, well_solidity, well_eccentricity);


