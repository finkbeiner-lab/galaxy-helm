%
function wells = listwells(filenames)
try
N = length(filenames);
wells = cell(N,1);

possibleWells = cell(96,1);
count = 1;
for k = 65:72
    for ii = 1:12
        possibleWells{count} = sprintf('_%s%d_',char(k), ii);
        count = count + 1;
    end
end

count = 0;
for k = 1:N
     ii = 0;
     found = false;
     while ii<=96 && ~found
         ii = ii + 1;
         if ~isempty( strfind( filenames{k}, possibleWells{ii}) )
             found = true;
             count = count + 1;
             wells{count} = possibleWells{ii};             
         end
     end
end

catch every
this_file = fullfile(pwd, 'error-data.mat');
disp(this_file);
save(this_file);
end
