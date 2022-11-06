%
% Path: /Users/siddharth.samsi/work/data/gaia/2014-12-22/mito/BioP1asynR2/Masks/PID20141202_BioP1asynR2_STACK_ALIGNED_A1_MONTAGE_MASK.tif
% f = '/Users/siddharth.samsi/work/data/gaia/2014-12-22/mito/BioP1asynR2/Montaged/FITC-DFTrCy5/PID20141202_BioP1asynR2_T0_0_A1_MONTAGE_FITC-DFTrCy5.tif';
% mask = '/Users/siddharth.samsi/work/data/gaia/2014-12-22/mito/BioP1asynR2/Masks/PID20141202_BioP1asynR2_STACK_ALIGNED_A1_MONTAGE_MASK.tif';

function [blobs_final, vesselness, l1, l2, dirn, scale1] = mito_analysis_blobs_lines(im_orig, mask, show)

if nargin == 2
    show = true;
end

im = mat2gray(im2double(im_orig));
[nr, nc, ~] = size(im);
radius = [2 5];%[2 5]; % minimum and maximum radius
sigma_min = radius(1)/sqrt(2);%0.5;
sigma_max = radius(2)/sqrt(2); % this may be too large since radius = sigma*sqrt(2), we will find max diameter of 10 pixels. thats huge!
sigma_step = 0.1;
if nargin==1
    [~, ~, mask] = getmser(im);
end
%% vesselness filter
%
sigma_min_v = 1; % 3
sigma_max_v = 2; % 5
opts = struct('FrangiScaleRange', [sigma_min_v sigma_max_v], 'FrangiScaleRatio', sigma_step, 'BlackWhite', false, 'verbose', false);
[vesselness, scale1, dirn, all_angles, all_filt, l1, l2, ~, ~] = FrangiFilter2D(im, opts); %#ok<ASGLU>

%% LoG scale space
%
sigma_range = sigma_min:sigma_step:sigma_max;
scale_space = generate_scale_space(im, sigma_range);
scale_space = scale_space.^2;
ns = size(scale_space, 3);

%%
% for each pixel, find the maximum and which scale it occurs at
mx = false(nr, nc, ns);
for k = 1:ns
    tmp = imregionalmax(scale_space(:,:,k));
    tmp(~mask) = 0; % restrict maxima to neuron mask
    mx(:, :, k) = tmp;
end
mx_sum = sum(mx, 3);
id = mx_sum>round(.1*ns); % keep the maxima that are maxima in more than 10% the scales. not sure if this is correct
%%

indices = find(id);
%hfig = figure;
%imagesc(im);
%hold('all');
blobs = zeros(length(indices), 4);
for k = 1:length(indices)
    [row, col] = ind2sub([nr nc], indices(k));
    data = squeeze(scale_space(row, col, :));
    max_loc = find(data==max(data), 1, 'first');
    radius_estimate = sigma_range(max_loc)/sqrt(2);
    %plot(data, '-*');
%    plot(col, row, 'w+');
%    h = show_blobs(hfig, [row col radius_estimate]);
    %text(col, row, sprintf('%0.5f', radius_estimate), 'Color', 'k');
    blobs(k,:) = [row col radius_estimate im(row,col)];
end

%close(hfig)
all_blob_center_ids = sub2ind([nr nc], blobs(:,1), blobs(:,2));
v = vesselness(all_blob_center_ids);
remove = v < 0.000001; % need some small value here basically
blobs = blobs(~remove, :);

blobs = padarray(blobs, [0 1], 'post');
for k = 1:size(blobs,1)
    blobs(k,end) = scale1(blobs(k,1), blobs(k,2));
end

%% examine grayscale levels in each blob
s = getblobstats(double(im), num2cell(blobs,2));
id = kmeans(s, 2, 'EmptyAction', 'singleton'); % cluster into two classes to split into blobs that might be outside the neuron but inside the mask
% determine which ones are inside the neuron
% these will be brigher on average
A = mean(s(id==1,1));
B = mean(s(id==2,1));
if A>B
    % we are interested in A
    blobs_final = blobs(id==1,:);
    discarded = blobs(id==2,:);
else
    blobs_final = blobs(id==2,:);
    discarded = blobs(id==2,:);
end


%% prune blobs
maxoverlap = .3;
blobs_final = my_prune_blobs(blobs_final, maxoverlap);

if show
    figure;imagesc(im);show_blobs(gcf, blobs_final);
    h = show_blobs(gcf, discarded);set(h, 'Color', 'w');
end


