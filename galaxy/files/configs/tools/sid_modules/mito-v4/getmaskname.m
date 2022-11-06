%
function filename = getmaskname(all_files, well)

id = cellfun(@(IN) ~isempty( strfind(IN, well) ), all_files);
filename = all_files{find(id, 1, 'first')};


