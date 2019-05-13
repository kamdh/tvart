clear all
close all

addpath('~/work/MATLAB/')
addpath('~/work/MATLAB/tensor_toolbox')
addpath('~/work/MATLAB/export_fig')
addpath('~/work/MATLAB/unlocbox')
init_unlocbox();
figdir = '../figures/';

%% Parameters
%% test system
%rng('shuffle');
%rng(1337)
rng(1)
N = 10;
M = 20;
r = 8;
num_steps = M * 10 + 1;
num_trans = 200;
noise_std = 0.5;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
%% tensor DMD algorithm
algorithm = 'alt_min';   % ALS or prox_grad
regularization = 'TV';
max_iter = 300;            % iterations
offset = 0;
center = 0;
eta = 1. / N;                % Tikhonov
beta =  5;               % regularization strength
proximal = 0;
save_plots = 1;

% %% testing params
% N_vec = [6,10,14,18,20,30,50,80,100,200,400,1000,2000,4000];
% %eta_vec = linspace(1, 0.01, length(N_vec));
% eta_vec = logspace(0, 1, length(N_vec));
% eta = 0.1 %eta_vec(end-2);
% N = N_vec(end-2);
% beta = 120 / N;


% algorithm = 'ALS_aux';   % ALS or prox_grad
% regularization = 'TV';
% max_iter = 300;            % iterations
% eta1 = 0.80;                % Tikhonov
% beta = 0.4;               % regularization strength
% eta2 = 1;               % closeness to relaxation variable
%                        
% regularization = 'TVL0';
% max_iter = 300;            % iterations
% offset = 0;
% center = 0;
% eta = 1;                % Tikhonov
% beta = 0.25;  
% regularization strength
%
% regularization = 'spline';
% max_iter = 300;            % iterations
% offset = 0;
% center = 0;
% eta = 1;                % Tikhonov
% beta = 10;               % regularization strength
%
% algorithm = 'ALS_aux';   % ALS or prox_grad
% regularization = 'TV';
% max_iter = 300;            % iterations
% eta1 = 1.;                % Tikhonova
% beta = 0.4;               % regularization strength
% eta2 = 1;               % closeness to relaxation variable
%
% algorithm = 'ALS_prox_mix';
% regularization = 'TV';
% max_iter = 400;            % iterations
% eta1 = 1e2;                % Tikhonov
% beta = 0.1;   
% regularization strength
%
% algorithm = 'ALS_aux';   ALS or prox_grad
% regularization = 'TV';
% max_iter = 60;            iterations
% eta1 = 100.0;                Tikhonov
% eta2 = 0.1;               closeness to relaxation variable
% beta = 1.0;               regularization strength
%
% algorithm = 'ALS_smooth';   % ALS or prox_grad
% max_iter = 60;
% eta1 = 50;
% eta2 = 0.1;
% beta = 1.0;   % regularization strength

%% Setup the problem

[X, A1, A2] = switching_linear_setup(N, num_steps, noise_std, offset);
%load(sprintf('test_data_N_%d_M_%d_sigma_%f.mat', N, num_steps, ...
%                 noise_std));

%% Preprocessing:
[U,S,V] = svd(X, 0);

%% reinitialize RNG
%rng('shuffle')

%% run tensor DMD
if strcmp(algorithm, 'alt_min')
    [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec] = ...
        TVART_alt_min(X, M, r, ...
                      'center', center, ...
                      'eta', eta, ...
                      'beta', beta, ...
                      'max_iter', max_iter, ...
                      'verbosity', 2, ...
                      'proximal', proximal, ...
                      'regularization', regularization);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    xlim([0, length(rmse_vec)])
elseif strcmp(algorithm, 'ALS')
    [lambda, A, B, C, Xten, Yten] = ...
        tensor_DMD_ALS(X, M, r, ...
                       'center', offset, ...
                       'eta', eta, ...
                       'nu', nu, ...
                       'beta', beta, ...
                       'max_iter', max_iter,...
                       'regularization', regularization);
elseif strcmp(algorithm, 'alt_min_l0')
    [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec] = ...
        tensor_DMD_alt_min_l0(X, M, r, ...
                           'center', center, ...
                           'eta', eta, ...
                           'beta', beta, ...
                           'max_iter', max_iter, ...
                           'verbosity', 2);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    xlim([0, length(rmse_vec)])
elseif strcmp(algorithm, 'ALS_aux')
    [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec, W] = ...
        tensor_DMD_ALS_aux(X, M, r, ...
                           'center', offset, ...
                           'eta1', eta1, ...
                           'eta2', eta2, ...
                           'beta', beta, ...
                           'max_iter', max_iter, ...
                           'proximal', 0, ...
                           'verbosity', 2, ...
                           'regularization', regularization);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    xlim([0, length(rmse_vec)])
elseif strcmp(algorithm, 'ALS_prox_mix')
    [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec, W] = ...
        tensor_DMD_ALS_prox_mix(X, M, r, ...
                                'center', 0, ...
                                'eta1', eta1, ...
                                'beta', beta, ...
                                'max_iter', max_iter, ...
                                'proximal', 0, ...
                                'regularization', regularization);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    %xlim([0,max_iter+1])
elseif strcmp(algorithm, 'ALS_smooth')
    [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec] = ...
        tensor_DMD_ALS_smooth(X, M, r, ...
                              'center', 1, ...
                              'eta1', eta1, ...
                              'eta2', eta2, ...
                              'beta', beta, ...
                              'max_iter', max_iter,...
                              'verbosity', 2);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    %xlim([0,max_iter+1])
elseif strcmp(algorithm, 'prox_grad')
    [lambda, A, B, C, Xten, Yten, cost_vec, rmse_vec] = ...
        tensor_DMD_prox_grad(X, M, r, ...
                             'center', 0, ...
                             'eta', eta, ...
                             'step_size', step_size, ...
                             'beta', beta, ...
                             'iter_disp', 1, ...
                             'max_iter', max_iter);
    figure(1)
    semilogy(cost_vec)
    hold on
    semilogy(rmse_vec, 'r-')
    legend({'cost', 'RMSE'})
    ylabel('cost')
    xlabel('iteration')
    %xlim([0,max_iter+1])
else
    error('algorithm not implemented');
end

%% noise compensation
if noise_compensate
    C = C / (1-noise_std);
end


% [lambda_r, A_r, B_r, C_r, W_r] = rebalance_2(A, B, C, W, 1, 1e-6);
% [lambda_r, A_r, B_r, C_r, W_r] = reorder_components(lambda_r, A_r, B_r, ...
%                                                   C_r, W_r);

% lambda = lambda_r;
% A = A_r;
% B = B_r;
% C = C_r;
% W = W_r;


%% Figure 2
figure(2)
subplot(2,1,1)
plot(X', 'color', [0,0,0] + 0.4, 'linewidth', 0.5)
axis tight
vline(M*[1:(T-1)], 'k--')
xlim([1, num_steps])
xlabel('Time step')
%ylabel('State variable')
title('Observations')
subplot(2,1,2)
plot(C, '-');
%if strcmp(algorithm, 'ALS_aux')
%    hold on
%    plot(W, 'ko');
%end
set(gca, 'xtick', 1:T)
axis tight
xlim([0.5, T+.5])
arr = ylim();
ylim([arr(1) - 0.05, arr(2) + 0.05])
xlabel('Window')
title('Temporal modes')
figure(2)
set(gcf, 'Color', 'w', 'Position', [100 500 800 600]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 6.5 4], 'PaperPositionMode', 'manual');
set(gcf,'renderer','painters')
%export_fig('-depsc2', '-r200', [figdir 'switching_summary.eps']);
if save_plots
    print('-depsc2', '-loose', '-r200', [figdir ...
                        'switching_summary.eps']);
end
%print('-dpng', '-r200', [figdir 'switching_summary.png']);

% A = Aten.U{1};
% B = Aten.U{2};
% C = Aten.U{3};

 
%% test closeness
figure(3);
set(gcf, 'position', [246          66         431        1231]);
for k = 1:T
    Dk = diag(C(k,:));       
    Ak = A * diag(lambda) * Dk * B';
    %Ak = Aten(:, :, k)
    ten_MSE = norm(Yten(:,:,k) - Ak * Xten(:,:,k), 'fro')^2 / (N*M);
    if k <= T/2
        % system 1
        A_true = A1;
        system = 1;
    else
        % system 2
        A_true = A2;
        system = 2;
    end
    
    [D] = eig(Ak(:,1:N));
    [Dtrue] = eig(A_true);
    
    A_err = max(max(abs(Ak(:, 1:N) - A_true)));
    A_err_2 = norm(Ak(:, 1:N) - A_true, 2);
    true_MSE = norm(Yten(:,:,k) - A_true * Xten(1:N,:,k), 'fro')^2 / ...
        (N*M);
    fprintf('\nWindow %d, system %d\n', k, system)
    fprintf('prediction MSE using A_tensor: %1.3g (SNR: %1.2g)\n', ten_MSE, ...
            ten_MSE / noise_std^2)
    fprintf('prediction MSE using A_truth: %1.3g (SNR: %1.2g)\n', true_MSE,...
            true_MSE / noise_std^2)
    fprintf('tensor MSE relative to truth: %1.3g%%\n', ...
            ten_MSE / true_MSE * 100)
    fprintf('A infty error: %1.3g\n', A_err)
    fprintf('A 2-error: %1.3g\n', A_err_2)
    fprintf('radius(A_tensor) = %1.2g, radius(A_true) = %1.2g\n', ...
            max(abs(D)), max(abs(Dtrue)));
    
    set(0, 'CurrentFigure', 3);
    subplot(3,1,1);
    imagesc(Ak)
    colorbar
    colormap(brewermap([], 'PuOr'))
    caxis([-1,1]/sqrt(N))
    %caxis auto
    title('TVAR system matrix', 'fontweight', 'normal')
    axis square
    
    subplot(3,1,2);
    imagesc(A_true);
    colorbar
    colormap(brewermap([], 'PuOr'))
    caxis([-1,1]/sqrt(N))
    %caxis auto
    title(['True matrix (system ' num2str(system) ')'], 'fontweight', 'normal')
    axis square

    subplot(3,1,3);
    imagesc(Ak(:,1:N) - A_true);
    colorbar
    colormap(brewermap([], 'PuOr'))
    caxis([-1, 1]/sqrt(N))
    %caxis auto
    title('Difference', 'fontweight', 'normal')
    axis square
    
    if k == 1
        set(gcf, 'Color', 'w', 'Position', [0 0 600 1000]);
        set(gcf, 'PaperUnits', 'inches', ...
                 'PaperPosition', [0 0 3 7.2], ...
                 'PaperPositionMode', 'manual');
        set(gcf,'renderer','painters')
        %print('-dpng', '-painters', '-r200',  [figdir
        %'switching_matrices.png'])
        if save_plots
            print('-depsc2', '-loose', '-r200', [figdir ...
                                'switching_matrices.eps']);
        end
        %export_fig('-depsc2', '-r200', [figdir 'switching_matrices.eps']);
        %        print('-dpdf', '-painters', [figdir 'switching_matrices.pdf']);
    end
    
    pause
    if k ~= T
        clf(3)
    end
end



%% Figure 4
figure(4)

set(gcf, 'position', [693          63         560        1240])
subplot(3,1,1);
imagesc(A);
title('Left spatial modes', 'fontweight', 'normal')
colormap(brewermap([], 'PuOr'))
c = caxis();
c = max(abs(c));
caxis([-c, c])
colorbar
axis square

subplot(3,1,2);
imagesc(B(1:N,:));
title('Right spatial modes', 'fontweight', 'normal')
colormap(brewermap([], 'PuOr'))
%caxis([-1, 1])
c = caxis();
c = max(abs(c));
caxis([-c, c])
colorbar
axis square

subplot(3,1,3);
imagesc(C);
title('Temporal modes', 'fontweight', 'normal')
colormap(brewermap([], 'PuOr'))
%caxis([-1, 1])
c = caxis();
c = max(abs(c));
caxis([-c, c])
colorbar
axis square

set(gcf, 'Color', 'w', 'Position', [0 0 600 1000]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 3 7.2], ...
         'PaperPositionMode', 'manual');
set(gcf,'renderer','painters')
if save_plots
    print('-depsc2', '-loose', '-r200', [figdir ...
                        'switching_components.eps']);
end
%export_fig('-depsc2', '-r200', [figdir 'switching_components.eps']);
%print('-dpng', '-r200', [figdir 'switching_components.png']);



%% Figure 5
figure(5);
scatter(real(Dtrue), imag(Dtrue), 'ob');
hold on
scatter(real(D), imag(D), '+k');
plot(cos(linspace(0,1,100)*2*pi), sin(linspace(0,1,100)*2*pi), 'k--')
title('eigenvalues', 'fontweight', 'normal')
xlim([-1.2, 1.2])
ylim([-1.2, 1.2])
axis square;
set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 4 4.5], ...
         'PaperPositionMode', 'manual');
set(gcf,'renderer','painters')
%export_fig('-depsc2', '-r200', [figdir 'switching_eigs.eps']);
if save_plots
    print('-depsc2', '-loose', '-r200', [figdir ...
                        'switching_eigs.eps']);
end
%print('-dpng', '-r200', [figdir 'switching_eigs.png']);



%% Clustering
%D = compute_pdist(A, B, C*diag(lambda));
D = pdist(C);
Z = linkage(squareform(D), 'complete');
cluster_ids = cluster(Z, 'maxclust', 3);
