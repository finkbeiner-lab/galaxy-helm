% EXTRACTCOMPONENT
%   This function is used to extract the specified object from a binary image
%
%   Usage :
%   bw = extractcomponent(bwimg, idx, imgtype)
%   [bw, rgbimg, props] = extractcomponent(bwimg, idx, imgtype, color_img)
%
%   where, imgtype can be 'FilledImage', 'Image' or 'ConvexImage'
%          idx is the index number of the object of interese
%
%   Example:
%   [bw, x, props] = extractcomponent(bwfinal, 8, 'FilledImage', 'Image', readimages(4, 3), 'Padding', 10);
%
%   See also displaycomponents

function [bw, im, props] = extractcomponent(bwimg, idx, imgtype, varargin)

[colorimg, padding, stats] = parseinputargs(varargin);

if isempty(stats)
    stats = regionprops(bwconncomp(bwimg), imgtype, 'Centroid', 'BoundingBox', ...
        'Orientation', 'SubarrayIdx');
end

if isempty(padding)
    %bw = stats(idx).(imgtype); % commented 2015/03/30
    if ~isempty(colorimg)
        im = colorimg(stats(idx).SubarrayIdx{:},:);
    end
    % Extract image with extra rows and columns as specified by user
    [startrow, endrow, startcol, endcol] = getboxindices(0, size(bwimg), ...
        floor(stats(idx).BoundingBox)); % this used to be ceil()
    [r, c] = meshgrid(startrow:endrow, startcol:endcol);
    idxList = sub2ind(size(bwimg), r(:), c(:));
    bw = bwimg(startrow:endrow, startcol:endcol);
else
    % Extract image with extra rows and columns as specified by user
    [startrow, endrow, startcol, endcol] = getboxindices(padding, size(bwimg), ...
        floor(stats(idx).BoundingBox)); % this used to be ceil()
    [r, c] = meshgrid(startrow:endrow, startcol:endcol);
    idxList = sub2ind(size(bwimg), r(:), c(:));
    bw = bwimg(startrow:endrow, startcol:endcol);
    if ~isempty(colorimg)
        im = colorimg(startrow:endrow, startcol:endcol, :);
    end
end

% Translate centroid co-ordinates to match extracted feature
props.centroid = floor(stats(idx).Centroid - stats(idx).BoundingBox(1:2));
props.theta = stats(idx).Orientation;
tmp = false(size(bwimg));
tmp(idxList) = true;
bb = regionprops(tmp, 'BoundingBox');
clear tmp;
%idx = stats(idx).SubarrayIdx{:};
%props.bb = stats(idx).BoundingBox;
props.bb = bb.BoundingBox;
props.idx = idxList;
%if nargin==4
%    img = varargin{1};
%    varargout{1} = img(stats(idx).SubarrayIdx{:},:);
%end


function [colorimg, padding, stats] = parseinputargs(inputs)
colorimg = [];
padding = [];
stats = [];

% inputs is a cell array and must be even
while ~isempty(inputs)
    switch(inputs{1})
        case 'Image'
            colorimg = inputs{2};
        case 'Padding'
            padding = inputs{2};
        case 'RegionProps'
            stats = inputs{2};
    end
    inputs = inputs(3:end);
end


function [startrow, endrow, startcol, endcol] = getboxindices(padfactor, image_dims, boundingbox)
% Assume 0 < padfactor < 100
p0 = round(boundingbox);
%padfactor = padfactor/100;
if padfactor<1
    rowpad = ceil(p0(4)*padfactor);
    colpad = ceil(p0(3)*padfactor);
else
    rowpad = padfactor;
    colpad = padfactor;
end

if p0(1) - colpad <= 0
    startcol = 1;
else
    startcol = p0(1) - colpad;
end
if p0(1)+p0(3)+colpad > image_dims(2)
    endcol = p0(1) + p0(3);
else
    endcol = p0(1) + p0(3) + colpad;
end

if p0(2) - rowpad <= 0
    startrow = 1;
else
    startrow = p0(2) - rowpad;
end
if p0(2)+p0(4)+rowpad > image_dims(1)
    endrow = p0(2) + p0(4) - 1;
else
    endrow = p0(2) + p0(4) -1 + rowpad;
end



