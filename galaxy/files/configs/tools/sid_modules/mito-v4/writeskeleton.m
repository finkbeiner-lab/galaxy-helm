%
function writeskeleton(im, sk, name)
% assuming the following:
% im : ROI
% sk : Binary skeleton (need to convert to correct format for writing to file)
% name : file name with full path (including directory)

im = mat2gray(im2double(im));
sk = mat2gray(im2double(sk));
imwrite([cat(1, im, im) sk], name);

