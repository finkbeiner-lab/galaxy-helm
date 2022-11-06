%
% f = getfilelist(pathname, ext)
%
function [a, b] = getfilelist(pathname, ext)

a = dir([pathname filesep '*.' ext]);
a = struct2cell(a);
a = a(1,:);
b = a;
a = cellfun(@(x) fullfile(pathname, x), a, 'UniformOutput', false);
a = a';

