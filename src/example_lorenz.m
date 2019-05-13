clear all
close all
addpath('~/work/MATLAB/unlocbox')
init_unlocbox();

load('../data/lorenz_0_001.mat', 'state', 'step');

save_file = 'output_lorenz.mat';
figdir = '../figures/';
dt = step;
clear('step');

rng(1)
%% Tensor DMD algorithm

% %% Good params w/o standardization, w/ TV-l0
M = 10;
R = 4;
max_iter = 2000;             % iterations
eta1 = 0.01;               % Tikhonov/proximal
eta2 = 1e-3;               % relaxed vars
beta = 4e2;               % regularization
center = 1;
noise = 1.0;
% %% Good w/ standardization
% M = 3;
% R = 6;
% max_iter = 140;             % iterations
% eta1 = 1e2;                 % Tikhonov/proximal
% eta2 = 1e-4;               % relaxed vars
% beta = 0.001;               % regularization
% center = 0;
% noise = 0.2;
%% Good params w/o standardization or centering
% M = 10;
% R = 6;
% max_iter = 2000;             % iterations
% eta1 = 1e-2;               % Tikhonov/proximal
% eta2 = 1e-1;               % relaxed vars
% beta = 20;               % regularization
% center = 1;
% noise = 1.0;

state = state(:, 1:4200);
%X = state(:, 1:2000);
X = state;
%X = randn(30,3) * state;
% X = standardize(X);
X = X + noise*randn(size(X));
%X = X - repmat(mean(X,2), 1, length(X));
%X = X ./ repmat(std(X,0,2), 1, length(X));

% [lambda, A, B, C, cost, Xten, Yten, rmse] = ...
%     tensor_DMD_ALS_smooth(X, M, R, ...
%                        'center', center, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta, ...
%                        'max_iter', max_iter);

% [lambda, A, B, C, cost, Xten, Yten, rmse, W] = ...
%     tensor_DMD_ALS_aux(X, M, R, ...
%                        'center', center, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta, ...
%                        'proximal', 0, ...
%                        'verbosity', 2,...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-6, ...
%                        'atol', 1e-6, ...
%                        'max_iter', max_iter);

[lambda, A, B, C, cost, Xten, Yten, rmse] = ...
    TVART_alt_min(X, M, R, ...
                  'center', center, ...
                  'eta', eta1, ...
                  'beta', beta, ...
                  'regularization', 'TV', ...
                  'verbosity', 2, ...
                  'max_iter', max_iter);

% [lambda, A, B, C, cost, Xten, Yten, rmse] = ...
%     tensor_DMD_alt_min(X, M, R, ...
%                        'center', center, ...
%                        'eta', eta1, ...
%                        'beta', beta, ...
%                        'regularization', 'TV', ...
%                        'rtol', 1e-4, ...
%                        'atol', 1e-5, ...
%                        'verbosity', 2,...
%                        'max_iter', max_iter);

% [lambda, A, B, C, cost, Xten, Yten, rmse] = ...
%     tensor_DMD_alt_min_l0(X, M, R, ...
%                        'center', center, ...
%                        'eta', eta1, ...
%                        'beta', beta, ...
%                        'regularization', 'TV0', ...
%                        'rtol', 1e-3, ...
%                        'atol', 1e-5, ...
%                        'verbosity', 2,...
%                        'max_iter', max_iter);

% [lambda, A, B, C, cost, Xten, Yten, rmse, W] = ...
%     tensor_DMD_ALS_smooth(X, M, R, ...
%                        'center', center, ...
%                        'eta1', eta1, ...
%                        'eta2', eta2, ...
%                        'beta', beta, ...
%                        'rtol', 1e-4, ...
%                        'atol', 1e-4, ...
%                        'max_iter', max_iter);

A_r = A; B_r = B; C_r = C; lambda_r = lambda;
% %W_r = W; 
% lambda_r = lambda;
W_r = C;
%[lambda_r, A_r, B_r, C_r, W_r] = rebalance_2(A, B, C, W_r, 1, 1e-6);
%[lambda_r, A_r, B_r, C_r, W_r] = reorder_components(lambda_r, A_r, B_r, ...
%                                                     C_r, W_r);


figure(1);
semilogy(cost);
hold on
semilogy(rmse, 'r-')
legend({'cost', 'RMSE'})
axis('auto')

% [~,I] = sort(lambda, 'descend');
% A = A(:,I);
% B = B(:,I);
% C = C(:,I);
% lambda = lambda(I);
% for i=1:size(C,2)
%     if all(C(:,i) < 0)
%         C(:,i) = C(:,i) * -1;
%         W(:,i) = W(:,i) * -1;
%         A(:,i) = A(:,i) * -1;
%     end
% end
%% Clustering
%D = compute_pdist(A_r, B_r, C_r*diag(lambda_r));
D = pdist(C);
Z = linkage(squareform(D), 'complete');
cluster_ids = cluster(Z, 'maxclust', 3);
T = size(C_r, 1);
N = size(A_r, 1);
xA = zeros(T, N*(N+center));
for k = 1:T
    Akaug = A_r * diag(C_r(k,:)) * B_r';
    xA(k, :) = Akaug(:);
end
%[clust_ids, clust_centroids] = kmedoids(xA, 3);
[clust_ids, clust_centroids] = kmedoids(C, 3);


figure(2);
ax1 = subplot(3,1,1);
plot(1:length(X), X', 'linewidth', 0.5, 'color', [0,0,0]+0.4)
title('Observations')
%legend({'x_1', 'x_2', 'x_3'})
% ax1 = subplot(3,1,2);
% plot(1:length(state), state(1,:))
% title('Trajectory')
%ylim([-30, 45])
ax2 = subplot(3,1,2);
plot(M*(1:length(C_r)), C_r, 'linewidth', 0.5);
title('Temporal modes')
%legend({'1', '2', '3', '4'})
%ylim([0.0475, 0.0491])
%hold on
%plot(M*(1:length(C_r)), W_r, 'ko');
% linkaxes([ax0 ax1 ax2], 'x')
ax3 = subplot(3,1,3);
plot(M/2 + (1:M:M*length(C)), cluster_ids, 'ks-');
ylim([0.8, 3.2])
%xlabel('Time')
title('Cluster')
xlabel('Time step')
linkaxes([ax1 ax2 ax3], 'x')
figure(2);
xlim([0, 2000])
%axis tight
%ylim([0.0475, 0.0491])
set(gcf, 'Color', 'w', 'Position', [100 200 600 700]);
set(gcf, 'PaperUnits', 'inches', ...
         'PaperPosition', [0 0 6.5 7.5], 'PaperPositionMode', 'manual');
print('-depsc2', '-loose', [figdir 'example_lorenz.eps']);

disp(lambda_r)
% fprintf('A_r^T A_r = \n')
% disp(A_r'*A_r)
% fprintf('B^T B = \n')
% disp(B'*B)
% fprintf('C_r^T C_r = \n')
% disp(C_r'*C_r)

save(save_file)


%% Compare scaled Jacobians
s1 = 10;  % sigma
s2 = 28;  % rho
s3 = 8/3; % beta
dt = 0.001;
fps = [ 0,                0,               0;
        sqrt(s3*(s2-1)),  sqrt(s3*(s2-1)), s2-1;
       -sqrt(s3*(s2-1)), -sqrt(s3*(s2-1)), s2-1];
Df = @(x) [-s1      s1    0;
           s2-x(3)  -1    -x(1);
           x(2)     x(1)  -s3];

for fp = 1:3
    disp('Jacobian + fwd Euler:')
    disp([eye(N) + dt * Df(fps(fp,:)),  fps(fp,:)'] )
end
for fp = 1:3
    fprintf('\nTVART cluster centroid:\n')
    Aclust = A*diag(clust_centroids(fp,:))*B(1:end-1,:)';
    bclust = A*diag(clust_centroids(fp,:))*B(end,:)';
    c = (eye(N) - Aclust) \ bclust;
    disp([Aclust, c])
    %disp(reshape(clust_centroids(fp,:), [N, N+center]))
    fprintf('\n')
end
