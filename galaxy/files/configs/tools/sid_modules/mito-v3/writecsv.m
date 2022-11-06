function writecsv(filename, name, time, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count)

fp = fopen(filename, 'wt');
if fp<0
    errordlg('Unable to write results file');
    tmp = sprintf('results-%s.mat', strrep(strrep(datestr(now), ':', '-'), ' ', '-'));
    save(tmp, 'filename', 'name', 'circularity', 'skel_len', 'eccentricity', 'aspect_ratio', 'solidity', 'blob_count');
else
    fprintf(fp, 'name, time, circularity, skel_len, eccentricity, aspect_ratio, solidity, blob_count\n');
    for k = 1:length(circularity)
        fprintf(fp, '%s,', name{k});
        fprintf(fp, '%s,', time{k});
        fprintf(fp, '%f,', circularity(k));
        fprintf(fp, '%f,', skel_len(k));
        fprintf(fp, '%f,', eccentricity(k));
        fprintf(fp, '%f,', aspect_ratio(k));
        fprintf(fp, '%f,', solidity(k));
        fprintf(fp, '%f\n', blob_count(k));
    end
    fclose(fp);
end
