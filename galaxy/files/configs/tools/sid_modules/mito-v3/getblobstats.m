%
% stats = getblobstats(im, blobs);
% blobStats = [center_value mean_gray std_gray min_gray max_gray];
%
function blobStats = getblobstats(im, blobs)
% assuming blobs : [xc yc r]

nblobs = size(blobs,1);
gray_values = cell(nblobs, 1);
sz = size(im);
all_blob_center_ids = zeros(nblobs,1);
for k = 1:nblobs
    [~, idx] = bwcircle( blobs{k}(2:-1:1), blobs{k}(3), sz);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % which one is correct : row vs. col ????
    %all_blob_center_ids(k) = sub2ind(sz, blobs{k}(2), blobs{k}(1));
    all_blob_center_ids(k) = sub2ind(sz, blobs{k}(1), blobs{k}(2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gray_values{k} = im(idx);
end

center_value = im(all_blob_center_ids);
mean_gray = cellfun(@(A) mean(A), gray_values);
std_gray = cellfun(@(A) std(A), gray_values);
min_gray = cellfun(@(A) min(A), gray_values);
max_gray = cellfun(@(A) max(A), gray_values);

blobStats = [center_value mean_gray std_gray min_gray max_gray];

