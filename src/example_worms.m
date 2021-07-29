clear all
close all

tic

addpath('~/work/MATLAB/unlocbox')
init_unlocbox();

worm_id = 1;
worm_data = h5read(['~/src/local-linear-segmentation/sample_data/' ...
                    'worm_tseries.h5'], ['/' num2str(worm_id) ...
                    '/tseries']);
%worm_data= standardize(worm_data);
figdir = '../figures/';

rng(1);
% %% Tensor decomp params
R = 6;
M = 6;
%M = 10;
eta = 0.05;
eta2 = 0.1;
beta = 6.0;
center = 1;
regularization = 'TV';
max_iter = 1000;
numclust = 3;

%% Good for TV-l0
% R = 12;
% M = 4;
% eta = 10.0;
% eta2 = 0.1;
% beta = 0.05;
% center = 0;
% max_iter = 1000;
% regularization = 'TVL0';


% [lambda, A, B, C, cost, Xten, Yten] = ...
%     tensor_DMD_ALS_aux(worm_data, M, R, ...
%                        'center', center, ...
%                        'eta', eta, ...
%                        'eta2', eta2, ...
%                        'beta', beta, ...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-4, ...
%                        'max_iter', max_iter,...
%                        'verbosity', 1);

[lambda, A, B, C, cost, Xten, Yten] = ...
    TVART_alt_min(worm_data, M, R, ...
                  'center', center, ...
                  'eta', eta, ...
                  'beta', beta, ...
                  'regularization', regularization, ...
                  'verbosity', 2, ...
                  'max_iter', max_iter);


% [lambda, A, B, C] = rebalance(A, B, C, 1);
% [lambda, A, B, C] = reorder_components(lambda, A, B, C);




%% Clustering
%D = compute_pdist(A, B, C*diag(lambda));
D = pdist(C);
Z = linkage(squareform(D), 'complete');
T = cluster(Z, 'maxclust', numclust);

phase = atan2(worm_data(2,:), worm_data(1,:));


%% Load Costa model results for comparison
Costa_data = load(['~/src/local-linear-segmentation/sample_data/' ...
                   'worm_1_results.mat']);
C_segmentation = (Costa_data.segmentation(:, 1) + ...
    Costa_data.segmentation(:, 2)) / 2; % midpoints of windows
C_labels = mod(3 - Costa_data.cluster_labels, 3) + 1 % reorder


figure
ax1 = subplot(4, 1, 1);
plot(1:length(worm_data), worm_data, 'color', [0,0,0]+0.4);
%plot(1:length(worm_data), worm_data);
%legend({'a_1','a_2','a_3','a_4'})
title('Observations')
ax2 = subplot(4, 1, 2);
plot(1:(length(worm_data)), phase,...
     'color', [0,0,0]+0.4)
ylim([-pi, pi])
title('Phase')
ax3 = subplot(4, 1, 3);
plot(M/2 + (1:M:M*length(C)), C);
title('Temporal modes')
%ylim([0.06, 0.077])
ax4 = subplot(4, 1, 4);
hold on
plot(M/2 + (1:M:M*length(C)), T, 'ks-');
plot(C_segmentation, C_labels, 'ko--');
text(10, 1.5, 'forward', 'Color', 'red')
text(65, 1.5, 'turn', 'Color', 'red')
text(140, 1.5, 'backward', 'Color', 'red')
legend({'TVART', 'Costa et al.'})
ylim([0.8, numclust+.2])
yticks([1,2,3])
box on
xlabel('Time step')
title('Clusters')
linkaxes([ax1 ax2 ax3 ax4], 'x')
set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
set(gcf, 'PaperUnits', 'inches', ...
         'PaperPosition', [0 0 6.5 7.5], 'PaperPositionMode', ...
         'manual');
print('-depsc2', '-loose', '-r200', [figdir 'example_worms.eps']);

elapsedtime = toc();

fprintf('%f seconds elapsed\n', elapsedtime);
