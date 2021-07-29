clear all
close all

addpath('~/work/MATLAB/')
%addpath('~/work/MATLAB/tensor_toolbox')
%addpath('~/work/MATLAB/export_fig')
addpath('~/work/MATLAB/unlocbox')
init_unlocbox();
figdir = '../figures/';

%% Parameters
%% test system
rng(1)

M = 20;
r = 4;
num_rep = 3;
num_steps = M * 100 + 1;
%num_steps = M * 10 + 1;
num_trans = 200;
noise_std = 0.5;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
offset = 0;
%% tensor DMD algorithm
regularization = 'TV';
max_iter = 300;            % iterations
center = 0;
eta = 1.;                % Tikhonov
beta =  1.;               % regularization strength
proximal = 0;
plotting = 0;

N_vec = [10,14,18,20,30,50,80,100,200,400]; %,2000,4000];
%eta_vec = linspace(1, 0.01, length(N_vec));
%eta_vec = logspace(0, -2, length(N_vec));
eta_vec = 1 ./ N_vec;
beta_vec = ones(length(N_vec), 1);
% output_file = '../data/comparison_output_rank6.csv';
% save_file = '../data/comparison_output_rank6.mat';
% output_file = '../data/comparison_output_rank4.csv';
% save_file = '../data/comparison_output_rank4.mat';
output_file = '../data/comparison_output_2000_rank6.csv';
save_file = '../data/comparison_output_2000_rank6.mat';

%% Now do the computation
err_table = zeros(num_rep * length(N_vec), 3 + 5);
row = 0;
N_idx = 0;
elapsed_times = zeros(length(N_vec), num_rep);
for N = N_vec
    N_idx = N_idx + 1;
    for rep = 1:num_rep
        % if N > 40
        %     eta = 0.;
        %     %beta = 4;
        % end
        % if N > 500
        %     eta = 0.05;
        %     %beta = 6;
        % end
        eta = eta_vec(N_idx);
        beta = beta_vec(N_idx);
        %% Setup the problem
        %save(sprintf('test_data_N_%d_M_%d_sigma_%f.mat', N, num_steps, ...
        %             noise_std), 'X', 'A1', 'A2');
        load(sprintf('../data/test_data_N_%d_M_%d_sigma_%f_rep_%d.mat', N, num_steps, ...
                     noise_std, rep));

        %% TVART model
        tic
        [lambda, A, B, C, cost_vec, Xten, Yten, rmse_vec] = ...
            TVART_alt_min(X, M, r, ...
                          'center', center, ...
                          'eta', eta, ...
                          'beta', beta, ...
                          'max_iter', max_iter, ...
                          'verbosity', 1, ...
                          'rtol', 1e-4,...
                          'proximal', proximal, ...
                          'regularization', regularization);
        t_elapsed = toc
        elapsed_times(N_idx, rep) = t_elapsed;
        fprintf('Fitting N = %d took %f s\n', N, t_elapsed);


        %% Independent models
        [Aindep] = indep_models(X, M);

        %% Low-rank independent models
        [U,S,V] = svd(X, 0);
        %Xr = U(:, 1:r) * S(1:r, 1:r) * V(:, 1:r)';
        %[Aindep_r] = indep_models(Xr, M);
        [Aindep_r] = indep_models(S(1:r, 1:r) * V(:, 1:r)' , M);

        for model = 1:3
            row = row + 1;
            err_mse = 0.;
            true_mse = 0.;
            err_inf = 0.;
            err_2 = 0.;
            err_fro = 0.;
            for t = 1:(num_steps - 1)
                k = floor((t - 1) / M) + 1;
                % true system matrix
                if t < num_steps / 2
                    A_true = A1;
                else
                    A_true = A2;
                end
                % predicted system matrix
                if model == 1
                    % indep(n)
                    A_pred = Aindep(:, :, k);
                elseif model == 2
                    % indep(r)
                    %Ak = Aindep_r(:, :, k);
                    A_pred = U(:, 1:r) * Aindep_r(:, :, k) * U(:, 1:r)';
                elseif model == 3
                    % TVART(r)
                    Dk = diag(C(k,:));
                    A_pred = A * diag(lambda) * Dk * B';
                end
                % compute errors
                xpred = A_pred * X(:, t);
                err_mse = err_mse + norm(xpred - X(:, t+1), 2)^2;
                true_mse = true_mse + norm(A_true * X(:, t) - X(:, t+1), 2)^2;
                err_inf = err_inf +  max(abs(A_pred(:) - A_true(:)));
                err_2 = err_2 + norm(A_pred - A_true, 2);
                err_fro = err_fro + norm(A_pred - A_true, 'fro');
            end % for t
            if model == 1
                disp('---full rank model---')
            elseif model == 2
                disp('---low rank model---')
            elseif model == 3
                disp('---TVART model---')
            end
            err_mse = err_mse / (N * (num_steps - 1))
            true_mse = true_mse / (N * (num_steps - 1))
            err_inf = err_inf / (num_steps - 1)
            err_2 = err_2 / (num_steps - 1)
            err_fro = err_fro / (num_steps - 1)
            
            err_table(row, :) = [N, nan, model, err_inf, err_2, ...
                                err_fro, err_mse, true_mse];            
            % for k = 1:T
            %     row = row + 1;
            %     if model == 1
            %         % indep(n)
            %         Ak = Aindep(:, :, k);
            %     elseif model == 2
            %         % indep(r)
            %         %Ak = Aindep_r(:, :, k);
            %         Ak = U(:, 1:r) * Aindep_r(:, :, k) * U(:, 1:r)';
            %     elseif model == 3
            %         % TVART(r)
            %         Dk = diag(C(k,:));
            %         Ak = A * diag(lambda) * Dk * B';
            %     end
            %     if k < T/2
            %         % system 1
            %         A_true = A1;
            %         system = 1;
            %     elseif k == T/2 && ~mod(T,2)
            %         A_true = (A1 + A2)/2;
            %         system = 1.5;
            %     else
            %         % system 2
            %         A_true = A2;
            %         system = 2;
            %     end
                
            %     [D] = eig(Ak(:, 1:N));
            %     [Dtrue] = eig(A_true);
            %     % compute errors:
            %     model_MSE = norm(Yten(:,:,k) - Ak * Xten(:,:,k), 'fro')^2 / (N*M);
            %     true_MSE = norm(Yten(:,:,k) - A_true * Xten(1:N,:,k), 'fro')^2 / ...
            %         (N*M);
            %     A_err_inf = max(max(abs(Ak(:, 1:N) - A_true)));
            %     A_err_2 = norm(Ak(:, 1:N) - A_true, 2);
            %     A_err_fro = norm(Ak(:, 1:N) - A_true, 'fro');
                
            %     err_table(row, :) = [N, k, model, A_err_inf, A_err_2, ...
            %                         A_err_fro, model_MSE, true_MSE];
                
            %     fprintf('\n\tWindow %d, system %d\n', k, system)
            %     fprintf('\tprediction MSE using A_tensor: %1.3g (SNR: %1.2g)\n', model_MSE, ...
            %             model_MSE / noise_std^2)
            %     fprintf('\tprediction MSE using A_truth: %1.3g (SNR: %1.2g)\n', true_MSE,...
            %             true_MSE / noise_std^2)
            %     fprintf('\tmodel MSE relative to truth: %1.3g%%\n', ...
            %             model_MSE / true_MSE * 100)
            %     fprintf('\tA infty error: %1.3g\n', A_err_inf)
            %     fprintf('\tA 2-error: %1.3g\n', A_err_2)
            %     fprintf('\tradius(A_tensor) = %1.2g, radius(A_true) = %1.2g\n', ...
            %             max(abs(D)), max(abs(Dtrue)));
                
            %     if plotting
            %         figure(1)
            %         set(gcf, 'position', [246          66         431        1231]);
            %         subplot(3,1,1);
            %         imagesc(Ak)
            %         colorbar
            %         colormap(brewermap([], 'PuOr'))
            %         caxis([-1,1])
            %         title(sprintf('Inferred system matrix, model %d', model), 'fontweight', 'normal')
            %         axis square
            %         subplot(3,1,2);
            %         imagesc(A_true);
            %         colorbar
            %         colormap(brewermap([], 'PuOr'))
            %         caxis([-1,1])
            %         title(['True matrix (system ' num2str(system) ')'], 'fontweight', 'normal')
            %         axis square
            %         subplot(3,1,3);
            %         imagesc(Ak(:,1:N) - A_true);
            %         colorbar
            %         colormap(brewermap([], 'PuOr'))
            %         caxis([-1, 1])
            %         title('Difference', 'fontweight', 'normal')
            %         axis square
            %         [~] = input('Press any key to continue');
            %         if k ~= T
            %             close
            %         end
            %     end
        end % model

        % for model = 1:3
        %     idx = err_table(:, 3) == model & err_table(:, 1) == N;
        %     errs = mean(err_table(idx, 4:end), 1);
        %     A_err_inf = errs(1); 
        %     A_err_2 = errs(2); 
        %     A_err_fro = errs(3);
        %     model_MSE = errs(4);
        %     true_MSE = errs(5);
        %     fprintf(['Model %d :\n\tA infinity error = %1.2g\n\tA 2-error = %1.2g\n\t' ...
        %              'model MSE = %1.2g\n\ttrue MSE = %1.2g\n'], model, A_err_inf, A_err_2, ...
        %             model_MSE, true_MSE);
        % end
    end % rep loop
end % N loop

% %% summary of errors for plotting
% error_summary = zeros(3, length(N_vec), 5);
% for model = 1:3
%     for nidx = 1:length(N_vec)
%         N = N_vec(nidx);
%         idx = err_table(:, 3) == model & err_table(:, 1) == N;
%         errs = mean(err_table(idx, 4:end, 1), 1);
%         error_summary(model, nidx, :) = errs;
%     end
% end

error_table = array2table(err_table, ...
                          'VariableNames', {'N', 'window' 'model', ...
                    'err_inf', 'err_2', 'err_fro', 'model_MSE', ...
                    'true_MSE'});

model_vec = err_table(:,3);

writetable(error_table, output_file);

% figure;
% loglog(N_vec, error_summary(:,:,1))
% legend({'independent', 'rank r indep', 'TVART'})
% xlabel('N')
% ylabel('||A - A_{true} ||_\infty')
% print('-depsc', [figdir, 'compare_err_inf.eps'])

% figure;
% loglog(N_vec, error_summary(:,:,2))
% legend({'independent', 'rank r indep', 'TVART'})
% xlabel('N')
% ylabel('||A - A_{true} ||_2')
% print('-depsc', [figdir, 'compare_err_2.eps'])

% figure;
% loglog(N_vec, error_summary(:,:,3))
% legend({'independent', 'rank r indep', 'TVART'})
% xlabel('N')
% ylabel('||A - A_{true} ||_{Fro}')
% print('-depsc', [figdir, 'compare_err_fro.eps'])

save(save_file)