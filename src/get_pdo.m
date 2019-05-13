function [pdo_index, pdo_dates] = get_pdo(start_date, end_date)
% Get the Pacific Decadal Oscillation indices for a given date
% range.
% 
% Modified from get_soi.m by Chad Greene:
% https://www.mathworks.com/matlabcentral/fileexchange/38629-get-el-nino-southern-oscillation-index-values
%
% Data:
% http://research.jisao.washington.edu/pdo/
    
    pdo_ascii = fopen('../data/PDO.txt');
    pdo_struct = textscan(pdo_ascii,'%n %n %n %n %n %n %n %n %n %n %n %n %n', ...
                          'HeaderLines', 0); 
    pdo_mat = cell2mat(pdo_struct); % converts cell structure to matrix
    disp(size(pdo_mat))
    % Regrid data from calendar format to time series:
    pdo_dates = NaN(size(pdo_mat, 1) * 12, 1);
    pdo_index = NaN(size(pdo_mat, 1) * 12, 1);
    counter = 1;
    for yid = 1:size(pdo_mat, 1);
        for month = 1:12
            pdo_dates(counter) = datenum(pdo_mat(yid, 1), month, 15);
            pdo_index(counter) = pdo_mat(yid, month + 1);
            counter = counter + 1;
        end
    end


    % If input arguments (start and/or end dates) are used, rewrite time series
    % data to include only dates of interest: 
    if exist('start_date','var') == 1 
        if start_date < datenum(1900, 1, 1)
            warning('Historical data record begins in January 1950.')
        end
        if end_date > datenum(2017, 12, 31)
            warning(['Historical data record ends in December ' ...
                     '2017.']);
        end
        pdo_index = pdo_index(pdo_dates >= start_date & pdo_dates <= end_date);
        pdo_dates = pdo_dates(pdo_dates >= start_date & pdo_dates <= end_date);
    end
    
    fclose(pdo_ascii);