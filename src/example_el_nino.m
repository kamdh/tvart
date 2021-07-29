clear all
close all
addpath('~/work/MATLAB')

seed=1;
rng(seed);

figdir = ['../figures/test_rng_', num2str(seed), '/'];
mkdir(figdir);

%% base parameters
M = 9;   % 4.5 weeks/month * 2 months 
R = 6;
downsample = 6;
%% tensor DMD algorithm
max_iter = 300;             % iterations
eta = 2e-3;                % Tikhonov
beta = 1e4;                 % regularizatio
center = 0;
                            %eta2 = 200/beta;            % closeness to relaxation variable
%% ALS
nu = 0.0;    % ALS noise level
RALS = 1;
%% prox grad
step_size = 1e-5;
save_file = 'test_el_nino_12_uncentered.mat';

sst = ncread('../data/sst.wkmean.1990-present.nc', 'sst');
ls_mask = ncread('../data/lsmask.nc', 'mask');
time = ncread('../data/sst.wkmean.1990-present.nc', 'time');
time = time + datenum('1800-1-1 00:00:00');
time = datetime(time, 'ConvertFrom', 'datenum');

%% Simple downsampling
%sst = sst(1:downsample:end, 1:downsample:end, :);
ls_mask = ls_mask(1:downsample:end, 1:downsample:end);
%% Try imresize instead (in loop below)

tau = size(sst, 3);
N = sum(ls_mask(:) == 1);
X = zeros(N, tau);

for i = 1:tau
    sst_t = sst(:, :, i);
    sst_t = imresize(sst_t, 1 / downsample);
    sst_t = sst_t(ls_mask == 1);
    X(:, i) = sst_t;
end
%X = standardize(X);

% [lambda, A, B, C, cost, Xten, Yten, rmse] = ...
%     tensor_DMD_alt_min_smooth(X, M, R, ...
%                               'center', center, ...
%                               'eta', eta, ...
%                               'beta', beta, ...
%                               'max_iter', max_iter,...
%                               'verbosity', 2);

[lambda, A, B, C, cost, Xten, Yten, rmse] = ...
    TVART_alt_min(X, M, R, ...
                  'center', center, ...
                  'eta', eta, ...
                  'beta', beta, ...
                  'max_iter', 10, ...
                  'regularization', 'spline', ...
                  'verbosity', 2);
% Set B = A and restart!
if ~center
    s = struct('A', A, 'B', A, 'C', C);
else
    s = struct('A', A, 'B', [A; B(end,:)], 'C', C);
end
[lambda, A, B, C, cost, Xten, Yten, rmse] = ...
     TVART_alt_min(X, M, R, ...
                   'center', center, ...
                   'eta', eta, ...
                   'beta', beta, ...
                   'max_iter', max_iter, ...
                   'regularization', 'spline', ...
                   'verbosity', 2,...
                   'init', s);



%plot((1:size(C,1))*M/52., C);
%xlabel('years')
%[lambda_r, A_r, B_r, C_r] = rebalance(A*diag(lambda), B, C, 1);
A_r = A; B_r = B; C_r = C; %W_r = W; 
lambda_r = lambda;
                           %W = C;
%[lambda_r, A_r, B_r, C_r] = reorder_components(lambda_r, A_r, B_r, C_r);
%[lambda_r, A_r, B_r, C_r, W_r] = rebalance_2(A, B, C, W, 1);

                           
[lambda_r, A_r, B_r, C_r] = rebalance(A, B, C, 1);
[lambda_r, A_r, B_r, C_r] = reorder_components(lambda_r, A_r, B_r, C_r);



plot_times = time(floor(M/2):M:end-M+1);
%[soi, soi_t] = get_soi(datenum(plot_times(1)), datenum(plot_times(end)));
[nino, nino_t] = get_nino(datenum(plot_times(1)), datenum(plot_times(end)));
[pdo, pdo_t] = get_pdo(datenum(plot_times(1)), datenum(plot_times(end)));

save(save_file);

%% Plotting
figure
semilogy(cost)


figure(2);
for mode = 1:R
    ax1 = subplot(4,1,1);
    plot(plot_times, C_r(:,mode), 'k-');
    %hold on
    %plot(plot_times, W_r(:,mode), 'ko');
    title(sprintf('Mode %d, \\lambda = %1.2f\nTemporal mode', mode, ...
                  lambda_r(mode)));
    ylim('auto')
    ax2 = subplot(4,1,2);
    plot(datetime(nino_t, 'ConvertFrom', 'datenum'), ...
         nino(:, 2:2:end), 'color', [0 0 0] + 0.4)
    %title(['El Ni' char(241) 'o indices'])
    title('ENSO indices');
    xlabel('Year')
    ax3 = subplot(4,1,3);
    regrid_a = nan(size(ls_mask));
    regrid_a(ls_mask == 1) = A_r(:, mode);
    regrid_b = nan(size(ls_mask));
    if center
        regrid_b(ls_mask == 1) = B_r(1:end-1, mode);
    else
        regrid_b(ls_mask == 1) = B_r(1:end, mode);
    end
    mypcolor = @(x) pcolor([x nan(size(x,1),1); nan(1, size(x,2)+ ...
                                                    1)]);
    imAlpha=ones(size(regrid_a'));
    imAlpha(isnan(regrid_a'))=0;
    imagesc(regrid_a', 'alphadata', imAlpha);
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title('Left spatial mode')
    colormap(flipud(brewermap([], 'PuOr')))
    c = max(abs(caxis));
    caxis([-c, c])
    colorbar
    ax4 = subplot(4,1,4);
    imAlpha=ones(size(regrid_b'));
    imAlpha(isnan(regrid_b'))=0;
    imagesc(regrid_b', 'alphadata', imAlpha);
    %imagesc(regrid_b');
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title('Right spatial mode')
    colormap(flipud(brewermap([], 'PuOr')))
    c = max(abs(caxis));
    caxis([-c, c])
    colorbar
    if center
        xlabel(sprintf('affine weight = %1.2g, %1.2g%%', ...
                       B_r(end, mode), ...
                       abs(B_r(end, mode)) / sum(abs(B_r(end,:))) * 100 ),...
               'fontsize', 14);
    end
    drawnow
    set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
    set(gcf, 'PaperUnits', 'inches', ...
             'PaperPosition', [0 0 6.5 7.5], 'PaperPositionMode', 'manual');
    print('-depsc2', '-loose', '-r200', ...
          [figdir 'example_el_nino_mode_' num2str(mode) '.eps']);
    % pause
    clf
end



figure(3);
for mode = 1:R
    ax1 = subplot(4,1,1);
    plot(plot_times, C_r(:,mode), 'k-');
    %hold on
    %plot(plot_times, W_r(:,mode), 'ko');
    title(sprintf('Mode %d, \\lambda = %1.2f\nTemporal mode', mode, ...
                  lambda_r(mode)));
    ylim('auto')
    ax2 = subplot(4,1,2);
    plot(datetime(pdo_t, 'ConvertFrom', 'datenum'), pdo, 'k-');
    %title(['El Ni' char(241) 'o indices'])
    title('PDO index');
    xlabel('Year')
    ax3 = subplot(4,1,3);
    regrid_a = nan(size(ls_mask));
    regrid_a(ls_mask == 1) = A_r(:, mode);
    regrid_b = nan(size(ls_mask));
    if center
        regrid_b(ls_mask == 1) = B_r(1:end-1, mode);
    else
        regrid_b(ls_mask == 1) = B_r(1:end, mode);
    end
    mypcolor = @(x) pcolor([x nan(size(x,1),1); nan(1, size(x,2)+ ...
                                                    1)]);
    imAlpha=ones(size(regrid_a'));
    imAlpha(isnan(regrid_a'))=0;
    imagesc(regrid_a', 'alphadata', imAlpha);
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title('Left spatial mode')
    colormap(flipud(brewermap([], 'PuOr')))
    c = max(abs(caxis));
    caxis([-c, c])
    colorbar
    ax4 = subplot(4,1,4);
    imAlpha=ones(size(regrid_b'));
    imAlpha(isnan(regrid_b'))=0;
    imagesc(regrid_b', 'alphadata', imAlpha);
    %imagesc(regrid_b');
    set(gca,'YTickLabel',[]);
    set(gca,'XTickLabel',[]);
    title('Right spatial mode')
    colormap(flipud(brewermap([], 'PuOr')))
    c = max(abs(caxis));
    caxis([-c, c])
    colorbar
    if center
        xlabel(sprintf('affine weight = %1.2g, %1.2g%%', ...
                       B_r(end, mode), ...
                       abs(B_r(end, mode)) / sum(abs(B_r(end,:))) * 100 ),...
               'fontsize', 14);
    end
    drawnow
    set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
    set(gcf, 'PaperUnits', 'inches', ...
             'PaperPosition', [0 0 6.5 7.5], 'PaperPositionMode', 'manual');
    print('-depsc2', '-loose', '-r200', ...
          [figdir 'example_pdo_mode_' num2str(mode) '.eps']);
    % pause
    clf
end

