function [lambda, A, B, C, cost_vec, varargout] = ...
        TVART_alt_min(data, window_size, r, varargin)
    % TVART_alt_min 
    %   Solve TVART problem using alternating minimization.
    %
    % Parameters:
    %  data : N by time matrix of data
    %  window_size : number of time points in window
    %  r : rank
    % 
    % Output:
    %  lambda : vector of scalings for components (all ones)
    %  A : left spatial modes
    %  B : right spatial modes
    %  C : temporal modes
    %  cost_vec : cost per iterate
    %  X (optional) : data reshaped into tensor
    %  Y (optional) : target data reshaped into tensor
    %  rmse_vec : RMSE per iterate
    %  
    % Optional parameters: 
    %  max_iter : maximum number of iterations (default: 20)
    %  eta : Tikhonov regularization parameter (default: 0.1)
    %  beta : temporal regularization parameter (default 0)
    %  regularization : temporal regularization (default: 'TV',
    %    options: 'TV', 'Spline', 'TV0')
    %  center : fit an affine/centered model (default: 0)
    %  iter_disp : display cost after this many iterations
    %    (default: 1)
    %  init : struct containing initialization (default: false)
    %  atol : absolute tolerance (default: 1e-6)
    %  rtol : relative tolerance (default: 1e-4)
    %
    
    % Input parsing, defaults
    max_iter = 20;
    iter_disp = 1;
    center = 0;
    eta = 0.1;   % Tikhonov
    beta = 0.0;   % total variation parameter
    regularization = 'TV';
    rtol = 1e-4;
    atol = 1e-6;
    verbosity = 1;
    
    p = inputParser;
    addParameter(p, 'max_iter', max_iter);
    addParameter(p, 'iter_disp', iter_disp);
    addParameter(p, 'center', center);
    addParameter(p, 'eta', eta);
    addParameter(p, 'proximal', 0);
    addParameter(p, 'beta', beta);
    addParameter(p, 'regularization', regularization);
    addParameter(p, 'scale_params', false);
    addParameter(p, 'init', false);
    addParameter(p, 'atol', atol);
    addParameter(p, 'rtol', rtol);
    addParameter(p, 'verbosity', verbosity);
    parse(p, varargin{:});
    max_iter = p.Results.max_iter;
    iter_disp = p.Results.iter_disp;
    center = p.Results.center;
    eta = p.Results.eta;
    proximal = p.Results.proximal;
    beta = p.Results.beta;
    regularization = p.Results.regularization;
    init = p.Results.init;
    atol = p.Results.atol;
    rtol = p.Results.rtol;
    verbosity = p.Results.verbosity;

    if verbosity > 0
        fprintf('Running TVART alternating minimization\n');
        fprintf('\tregularization %s\n', regularization);
        fprintf('\tproximal = %d\n', proximal);
    end
    if verbosity > 1
        fprintf('\tbeta = %1.3g\n', beta);
        fprintf('\teta  = %1.3g\n', eta);
    end
    fprintf('\n');
    
    if ~(strcmp(regularization, 'TV') || ...
         strcmp(regularization, 'spline') || ...
         strcmp(regularization, 'TVL0'))
        disp(['Received regularization = ' regularization])
        error(['Unknown regularization type, should be one of ''TV'', ' ...
               '''spline,'' or ''TVL0''']);
    end
    
    N = size(data, 1);
    M = window_size;
    t = size(data, 2);
    
    T = floor((t-1) / window_size);
    X = data(:, 1:end-1);
    X = X(:, 1:M*T);
    Y = data(:, 2:end);
    Y = Y(:, 1:M*T);
   
    % initialize decomposition
    if isstruct(init)
        disp('using initialization given')
        A = init.A;
        B = init.B;
        C = init.C;
        if isfield(init, 'lambda')
            lambda = init.lambda;
            C = C * diag(lambda);
        end
        lambda = ones(r, 1);
    else
        [A, B, C] = init_dmd(X, Y, r, T);  % initalization
        if center
            B = [B; zeros(1, r)];
        end
        lambda = ones(r, 1);
    end

    % reshape the data into tensors
    % throws out some data if not an integer multiple of M
    if center == 0
        X = reshape(X, [N, M, T]);
    else
        % append ones to input data
        X = reshape([X; ones(1, M*T)], [N+1, M, T]);
    end
    Y = reshape(Y, [N, M, T]);
    
    cost_vec = zeros(max_iter, 1);
    rmse_vec = zeros(max_iter, 1);
    % Tcurr = 0;
    c = cost(X, Y, A, B, C, N, M, T, r, eta, ...
             beta, regularization, proximal);
    %cost_vec(1) = c;
    fprintf('initial cost = %1.5g\n', c);
    %fprintf('eta = %1.3g, 1/eta = %1.5g\n', eta, 1/(eta));
    update_iter = 0;
    for iter = 1:max_iter
        %% Update A
        old_A = A;
        old_c = c;
        A = solve_A(X, Y, A, B, C, N, M, T, r, eta, proximal);
        if verbosity >= 2
            [c,rmse] = cost(X, Y, A, B, C, N, M, T, r,...
                            eta, beta, regularization, proximal);
            fprintf('\tpost A solve: cost = %1.5g, RMSE = %1.3g\n', ...
                    c, rmse);
        end
        if old_c < c
            fprintf(['Warning: not updating A because cost went up!\' ...
                     'n']);
            A = old_A;
            c = old_c;
        end

        %% Update B
        old_B = B;
        old_c = c;
        B = solve_B_cg(X, Y, A, B, C, N, M, T, r, eta, proximal);
        if verbosity >= 2
            [c,rmse] = cost(X, Y, A, B, C, N, M, T, r,...
                            eta, beta, regularization, proximal);
            fprintf('\tpost B solve: cost = %1.5g, RMSE = %1.3g\n', ...
                    c, rmse);        
        end
        if old_c < c
            fprintf(['Warning: not updating B because cost went up!\' ...
                     'n']);
            B = old_B;
            c = old_c;
        end

        %% Update C
        old_C = C;
        olc_c = c;
        if strcmp(regularization, 'spline')
            C = solve_C_cg(X, Y, A, B, C, N, M, T, r, eta, beta, proximal);
        else
            C = solve_C_proxg(X, Y, A, B, C, N, M, T, r, eta, beta, ...
                              regularization, proximal);
        end
        [c,rmse] = cost(X, Y, A, B, C, N, M, T, r,...
                        eta, beta, regularization, proximal);
        if verbosity >= 2
            fprintf('\tpost C solve: cost = %1.5g, RMSE = %1.3g\n', ...
                    c, rmse);
        end
        if old_c < c
            fprintf(['Warning: not updating C because cost went up!\' ...
                     'n']);
            C = old_C;
            c = old_c;
        end

        cost_vec(iter) = c;
        rmse_vec(iter) = rmse;
        if mod(iter, iter_disp) == 0
            fprintf('iter %d: cost = %1.5g, RMSE = %1.3g\n', ...
                    iter, c, rmse);
        end
        
        if iter >= 2
            %cost_rel = cost_vec(2);
            cost_rel = cost_vec(iter - 1);
        end
        if iter > 2 && abs(cost_vec(iter) - cost_vec(iter-1)) < ...
                max([rtol * cost_rel, atol])
            fprintf('tolerance reached!\n');
            break
        end
        if isnan(cost_vec(iter))
            fprintf('Error: algorithm diverged\n')
            break
        end
    end
    rmse_vec = rmse_vec(1:iter);
    cost_vec = cost_vec(1:iter);

    var_tot = 0;
    var_int = 0;
    var_aff = 0;
    for k = 1:T
        Dk = diag(C(k, :));
        Ypred_k = A * diag(lambda) * Dk * B' * X(:,:,k);
        var_aff = var_aff + norm(Y(:,:,k) - Ypred_k, 'fro')^2;
        if center
            Ypred_int = A * diag(lambda) * Dk * B(end, :)' * ...
                ones(1, size(Ypred_k, 2));
            var_int = var_int + norm(Y(:,:,k) - Ypred_int, ...
                                     'fro')^2;
        else
            var_int = var_int + norm(Y(:,:,k), 'fro')^2;
        end
        var_tot = var_tot + norm(Y(:,:,k), 'fro')^2;
    end
    var_tot = var_tot / (N*M*T);
    var_int = var_int / (N*M*T);
    var_aff = var_aff / (N*M*T);
    fprintf('total var:\t\t%1.3f\n', var_tot);
    fprintf('intercept resid var:\t%1.3f, %1.3g%%\n', var_int, ...
            var_int / var_tot* 100);
    fprintf('affine model resid var:\t%1.3f, %1.3g%%\n', var_aff, ...
            var_aff / var_tot* 100);
    
    if nargout >= 6
        varargout{1} = X;
    end
    if nargout >= 7
        varargout{2} = Y;
    end
    if nargout >= 8
        varargout{3} = rmse_vec;
    end
end

function Anew = solve_A(X, Y, A, B, C, N, M, T, r, eta, proximal)
    rhs = zeros(N, r);
    sysmat = zeros(r, r);
    for k = 1:T
        Dk = diag(C(k, :));
        rhs = rhs + Y(:,:,k) * X(:,:,k)' * B * Dk;
        sysmat = sysmat + Dk * B' * X(:,:,k) * X(:,:,k)' * B * Dk;
    end
    sysmat = sysmat + (1/eta) * eye(size(sysmat));
    if proximal
        rhs = rhs + (1/eta) * A;
    end
    %fprintf('\tcond(A) = %1.3g\n', cond(sysmat))
    Anew = rhs / sysmat;
    % function y=apply_mat(x)
    %     x = reshape(x, size(A));
    %     y = x*(sysmat + (1/eta)*eye(size(sysmat)));
    %     y = y(:);
    % end
    % maxit = 10;
    % tol = 1e-4;
    % [x, flag, rr, iter] = pcg(@apply_mat, rhs(:), tol, maxit);
    % Anew = reshape(x, size(A));
    % if flag ~= 0
    %     disp('Warning: pcg did not converge in A solve')
    %     fprintf('flag: %d, relres: %f, iter: %d\n', flag, rr, iter)
    % end
end

% function Bnew = solve_B(X, Y, A, B, C, N, M, T, r, eta)
% % solve nasty sylvester equation the stupid way
%     rhs = zeros(size(B));
%     sysmat = zeros(prod(size(B)), prod(size(B)));
%     %fprintf('\t\tsolve_B: 1/eta = %1.5g\n', 1/eta);
%     for k = 1:T
%         Dk = diag(C(k, :));
%         rhs = rhs + X(:,:,k) * Y(:,:,k)' * A * Dk;
%         Rk = Dk * A' * A * Dk;
%         Lk = X(:,:,k) * X(:,:,k)';
%         sysmat = sysmat + kron(Rk', Lk);
%     end
%     % if proximal
%     %     rhs = rhs + (1/eta1) * B;
%     % end
%     rhs = rhs(:);
%     if size(B,1) > N
%         tmp = eye(size(sysmat));
%         for i = 1:r
%             tmp(i*(N+1), i*(N+1)) = 0.0;
%         end
%         bvec = (sysmat + (1/eta) * tmp) \ rhs;
%     else
%         bvec = (sysmat + (1/eta) * eye(size(sysmat))) \ rhs;
%     end
%     Bnew = reshape(bvec, size(B));
% end

function Bnew = solve_B_cg(X, Y, A, B, C, N, M, T, r, eta, proximal)
    maxit = 24;
    tol = 1e-3;
%fprintf('\t\tsolve_B: 1/eta = %1.5g\n', 1/eta);
    function z = apply_sylv(b)
        Bt = reshape(b, size(B));
        Zt = zeros(size(X,1), r);
        for k = 1:T
            Dk = diag(C(k, :));
            Rk = Dk * A' * A * Dk;
            Lk = X(:,:,k) * X(:,:,k)';
            Zt = Zt + Lk * Bt * Rk;
        end
        if size(B,1) > N
            Zt = Zt + (1/eta) * [Bt(1:N,:); zeros(1,r)];
        else
            Zt = Zt + (1/eta) * Bt;
        end
        z = Zt(:);
    end

    rhs = zeros(size(B));
    for k = 1:T
        Dk = diag(C(k, :));
        rhs = rhs + X(:,:,k) * Y(:,:,k)' * A * Dk;
    end
    if proximal
        rhs = rhs + (1/eta) * B;
    end
    rhs = rhs(:);

    % function z = precond(bx)
    %     Bx = reshape(bx, size(B));
    %     Z = zeros(size(X,1), r);
    %     L = zeros(size(X,1), size(X,1));
    %     R = zeros(r, r);
    %     for k = 1:T
    %         Dk = diag(C(k, :));
    %         R = R + Dk * A' * A * Dk;
    %p         L = L + X(:,:,k) * X(:,:,k)';
    %     end
    %     Z = L \ Bx / R;
    %     z = Z(:);
    % end
    
    [x, flag, rr, iter] = pcg(@apply_sylv, rhs, tol, maxit, [], [], ...
                              B(:));
    Bnew = reshape(x, size(B));
    if flag ~= 0
        disp('Warning: pcg did not converge in B solve')
        fprintf('flag: %d, relres: %f, iter: %d\n', flag, rr, iter)
    end
end
    
function Cnew = solve_C_proxg(X, Y, A, B, C, N, M, T, r, eta, beta, ...
                              regularization, proximal)
    max_iter = 40;
    shrinkage = 1/10;
    %step_size = 2e-4 / T;
    
    function c = g(X, Y, A, B, C, N, M, T, r, eta, beta)
        c = 0;
        for k = 1:T
            Dk = diag(C(k, :));
            Ypred_k = A * Dk * B' * X(:,:,k);
            c = c + ...
                0.5 * norm(Y(:,:,k) - Ypred_k, 'fro')^2;
        end
        if ~proximal
            if size(B,1) > N
                c = c + 0.5 / (eta) * ...
                    (norm(A, 'fro')^2 + ...
                     norm(B(1:N,:), 'fro')^2 + ...
                     norm(C, 'fro')^2);
            else
                c = c + 0.5 / (eta) * ...
                    (norm(A, 'fro')^2 + ...
                     norm(B, 'fro')^2 + ...
                     norm(C, 'fro')^2);
            end
        end
    end

    step_size = 1;
    for iter = 1:max_iter
        if iter == 1
            gold = g(X, Y, A, B, C, N, M, T, r, eta, beta);
            grad = grad_C(X, Y, A, B, C, N, M, T, r, eta);
            Cmom = C;
        else
            Cmom = C + (iter - 2)/(iter + 1) * (C - Cold);
            gold = g(X, Y, A, B, Cmom, N, M, T, r, eta, beta);
            grad = grad_C(X, Y, A, B, Cmom, N, M, T, r, eta);
        end
        if proximal
            grad = grad - (1/eta) * C;
        end        %fprintf('\t\tg_old = %1.3g', gold);
        while 1 % line search loop
            Cnew = Cmom - step_size * grad;
            if beta > 0
                if strcmp(regularization, 'TV')
                    for k = 1:r
                        %Cnew(:, k) = l1tv(Cnew(:, k), beta * step_size);
                        param = {};
                        param.verbose = 0;
                        param.use_fast = 1;
                        param.init = Cnew(:, k);
                        Cnew(:, k) = prox_tv1d(Cnew(:, k), ...
                                               beta * step_size,...
                                               param);
                    end
                elseif strcmp(regularization, 'TVL0')
                    for k = 1:r
                        [Cnew(:,k), ~] = prox_tv0(Cnew(:, k), beta * step_size);
                    end
                end
            end
            G = (Cmom(:) - Cnew(:)) / step_size;
            gnew = g(X, Y, A, B, Cnew, N, M, T, r, eta, beta);
            if gnew > (gold - step_size * grad(:)' * G + ...
                       step_size / 2 * norm(G)^2)
                if step_size > 1e-12
                    step_size = step_size * shrinkage;
                    %fprintf('shrinking step size to %f\n', ...
                    %        step_size);
                else
                    %fprintf(' g_new = %1.3g\n', gnew);
                    step_size = step_size * shrinkage;
                    warning('step size very small!')
                    % Cold = C;
                    % C = Cnew;
                    % break
                end
            else
                %fprintf(' g_new = %1.3g\n', gnew);
                Cold = C;
                C = Cnew;
                break
            end
        end
    end
end

function Cnew = solve_C_cg(X, Y, A, B, C, N, M, T, r, eta, beta, proximal)
    maxit = 24;
    tol = 1e-4;
    diff_mat = setup_smoother(T);
    
    function z = apply_mat(c)
        Ct = reshape(c, size(C));
        Zt = zeros(size(C));
        for k = 1:T
            Xk = X(:, :, k);
            Yk = Y(:, :, k);
            % rhs = diag(B' * Xk * Yk' * A);
            L = B'* Xk * Xk' * B;
            R = A' * A;
            sysmat = L .* R;
            sysmat = sysmat + (1/eta) * eye(r);
            Zt(k, :) = sysmat * Ct(k, :)';
        end
        Zt = Zt + beta * diff_mat' * diff_mat * Ct;
        z = Zt(:);
    end
    
    rhs = zeros(size(C));
    for k = 1:T
        Xk = X(:, :, k);
        Yk = Y(:, :, k);
        rhs(k, :) = diag(B' * Xk * Yk' * A);
    end
    if proximal
        rhs = rhs + (1/eta) * C;
    end
    rhs = rhs(:);
    
    [x, flag, rr, iter] = pcg(@apply_mat, rhs, tol, maxit, [], [], ...
                              C(:));
    Cnew = reshape(x, size(C));
    if flag ~= 0
        disp('Warning: pcg did not converge in C solve')
        fprintf('flag: %d, relres: %f, iter: %d\n', flag, rr, iter)
    end
end

function [c,varargout] = cost(X, Y, A, B, C, N, M, T, r, ...
                              eta, beta, regularization, proximal)
    c = 0;
    for k = 1:T
        Dk = diag(C(k, :));
        Ypred_k = A * Dk * B' * X(:,:,k);
        c = c + ...
            0.5 * norm(Y(:,:,k) - Ypred_k, 'fro')^2;
    end
    rmse = sqrt(2 * c / (N * M * T));
    if beta > 0
        if strcmp(regularization, 'TV')
            %% L1 TV
            DC = diff(C);
            c = c + beta * sum(abs(DC(:)));
        % elseif strcmp(regularization, 'groupTV')
        %     %% Group TV
        %     for k = 2:T
        %         c = c + beta * norm(C(k, :) - C(k-1, :), 2);
        %     end
        elseif strcmp(regularization, 'spline')
            diff_mat = setup_smoother(T);
            c = c + 0.5 * beta * norm(diff_mat * C, 'fro')^2;
            %c = c + 0.5 * beta * norm(diff(C), 'fro')^2;
        elseif strcmp(regularization, 'TV0')
            %% L0 TV
            c = c + beta * nnz(diff(C));
        end
    end
    if ~proximal
        if size(B,1) > N
            c = c + 0.5 / (eta) * ...
                (norm(A, 'fro')^2 + ...
                 norm(B(1:N,:), 'fro')^2 + ...
                 norm(C, 'fro')^2);
        else
            c = c + 0.5 / (eta) * ...
                (norm(A, 'fro')^2 + ...
                 norm(B, 'fro')^2 + ...
                 norm(C, 'fro')^2);
        end
    end
    
    if nargout ==2
        varargout{1} = rmse;
    end
end



function g = grad_C(X, Y, A, B, C, N, M, T, r, eta)
    g = zeros(size(C));
    for k = 1:T
        % grad for each row of C separately
        Dk = diag(C(k, :));
        g1 = diag(B' * X(:,:,k) * Y(:,:,k)' * A);
        L = B'* X(:,:,k) * X(:,:,k)' * B;
        R = A' * A;
        sysmat = L .* R;
        g2 = diag(L * Dk * R);
        g(k, :) = - g1 + g2;
    end
    g = g + (1/eta) * C;
end

function D = setup_smoother(T)
    I2  = speye(T-1, T-1);
    O2  = zeros(T-1, 1);
    D   = [I2 O2]+[O2 -1*I2]; % first difference matrix
                              %Dchol = chol( (eta2 * beta) * (D'*D) + eye(T) );
                              %sysinv = pinv(sysmat);
end
