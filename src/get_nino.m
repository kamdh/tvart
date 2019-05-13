function [soi_ind, soi_dates] = get_nino(start_date, end_date)
% Get the El Niño/Southern Oscillation indices for a given date
% range.
% 
% Modified from get_soi.m by Chad Greene:
% https://www.mathworks.com/matlabcentral/fileexchange/38629-get-el-nino-southern-oscillation-index-values
%
% Data:
% https://www.cpc.ncep.noaa.gov/data/indices/ersst4.nino.mth.81-10.ascii

soi_ascii = fopen('../data/ersst4.nino.mth.81-10.ascii');
soi_struct = textscan(soi_ascii,'%n %n %n %n %n %n %n %n %n %n', ...
                      'HeaderLines', 1); 
soi_mat = cell2mat(soi_struct); % converts cell structure to matrix

% Regrid data from calendar format to time series:
soi_dates = NaN(size(soi_mat,1), 1); % preallocate variable
soi_ind = NaN(size(soi_dates,1), size(soi_mat,2)-2);        % preallocate variable
for m = 1:length(soi_mat); 
    soi_dates(m) = datenum(soi_mat(m,1), soi_mat(m,2),15);
    soi_ind(m, :) = soi_mat(m, 3:end);
end


% If input arguments (start and/or end dates) are used, rewrite time series
% data to include only dates of interest: 
if exist('start_date','var')==1 
    if start_date < datenum(1950,1,1)
        warning('Historical data record begins in January 1950.')
    end
    if exist('end_date','var')~=1
    end_date = datenum(date); % uses today as end_date if not specified
    end
    soi_ind = soi_ind(soi_dates>=start_date&soi_dates<=end_date, :);
    soi_dates = soi_dates(soi_dates>=start_date&soi_dates<=end_date);
end

fclose(soi_ascii);