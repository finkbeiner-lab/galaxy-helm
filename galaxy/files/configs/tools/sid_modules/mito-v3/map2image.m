function map = map2image(wells, values)

% assume that "wells" is a cell array of string
% assume that "values" is a numeric array
map = zeros(12, 96);

for k = 1:length(wells)
    thisWell = wells{k};
    % split into row and column
    row = thisWell(1);
    col = thisWell(2:end);
    map( getRowNumber(row), str2double(col) ) = values(k);
end

function val = getRowNumber(well)
switch (well)
  case 'A'
    val = 1;
  case 'B'
    val = 2;
  case 'C'
    val = 3;
  case 'D'
    val = 4;
  case 'E'
    val = 5;
  case 'F'
    val = 6;
  case 'G'
    val = 7;
  case 'H'
    val = 8;
  otherwise
    val = 1;
end
