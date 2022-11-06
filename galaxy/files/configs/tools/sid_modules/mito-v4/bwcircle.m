% BWCIRCLE Returns binary image with a circle of specified center and radius
%
%   Usage:
%   bwimg = bwcircle(center, radius, image_size)
%   [bwimg, idx] = bwcircle(center, radius, image_size)
%
%   Function also returns the subscript indices of the matrix locations
%   inside the circle
function [bwimg, idx] = bwcircle(center, radius, image_size)

[x, y] = meshgrid(1:image_size(2), 1:image_size(1));

bwimg = (x-center(1)).^2 + (y-center(2)).^2 <= radius^2 ;

idx = find(bwimg);
