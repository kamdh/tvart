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
rng(2)

N = 10;
M = 1;
r = 4;
num_steps = M * 160 + 1;
noise_std = 0.2;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
%% tensor DMD algorithm
algorithm = 'alt_min';   % ALS or prox_grad
regularization = 'spline';
max_iter = 300;           % iterations
offset = 0;
center = 0;
eta = 6. / N;            % Tikhonov
beta = 600 * log10(N)^2;               % regularization strength
proximal = 0;
save_plots = 1;


%% Setup the problem

[X, thetas, U] = smooth_linear_setup(N, num_steps, noise_std);
%load(sprintf('test_data_N_%d_M_%d_sigma_%f.mat', N, num_steps, ...
%                 noise_std));

%% Preprocessing:
%[U,S,V] = svd(X, 0);

%% reinitialize RNG
%rng('shuffle')

%% run tensor DMD
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

%% Figure 2
figure(2)
subplot(2,1,1)
plot(X', 'color', [0,0,0] + 0.4, 'linewidth', 0.5)
axis tight
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
%set(gca, 'xtick', 1:T)
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
%export_fig('-depsc2', '-r200', [figdir 'smooth_summary.eps']);
if save_plots
    print('-depsc2', '-loose', '-r200', [figdir ...
                        'smooth_summary.eps']);
end
%print('-dpng', '-r200', [figdir 'smooth_summary.png']);

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
    ten_MSE = norm(Yten(:,:,k) - Ak * Xten(:,:,k), 'fro')^2 / (N* ...
                                                      M);
    Ar = [cos(thetas(k+1)) -sin(thetas(k+1));
          sin(thetas(k+1))  cos(thetas(k+1))];
    A_true = U * Ar * U';
    
    [D] = eig(Ak(:,1:N));
    [Dtrue] = eig(A_true);
    
    A_err = max(max(abs(Ak(:, 1:N) - A_true)));
    A_err_2 = norm(Ak(:, 1:N) - A_true, 2);
    true_MSE = norm(Yten(:,:,k) - A_true * Xten(1:N,:,k), 'fro')^2 / ...
        (N*M);

    fprintf('\nWindow %d\n', k)
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
    title('True matrix', 'fontweight', 'normal')
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
        %'smooth_matrices.png'])
        if save_plots
            print('-depsc2', '-loose', '-r200', [figdir ...
                                'smooth_matrices.eps']);
        end
        %export_fig('-depsc2', '-r200', [figdir 'smooth_matrices.eps']);
        %        print('-dpdf', '-painters', [figdir 'smooth_matrices.pdf']);
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
                        'smooth_components.eps']);
end
%export_fig('-depsc2', '-r200', [figdir 'smooth_components.eps']);
%print('-dpng', '-r200', [figdir 'smooth_components.png']);



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
%export_fig('-depsc2', '-r200', [figdir 'smooth_eigs.eps']);
if save_plots
    print('-depsc2', '-loose', '-r200', [figdir ...
                        'smooth_eigs.eps']);
end
%print('-dpng', '-r200', [figdir 'smooth_eigs.png']);



%% Clustering
%D = compute_pdist(A, B, C*diag(lambda));
D = pdist(C);
Z = linkage(squareform(D), 'complete');
cluster_ids = cluster(Z, 'maxclust', 3);
