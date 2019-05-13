function [A] = indep_models(data, window_size, varargin)
    % param defaults
    center = 0;
    verbosity = 1;
    % parse params
    p = inputParser;
    addParameter(p, 'center', center);
    addParameter(p, 'verbosity', verbosity);
    parse(p, varargin{:});
    center = p.Results.center;
    verbosity = p.Results.verbosity;

    if verbosity > 0
        fprintf('Running independent modeling\n');
        fprintf('\twindow size: %d\n', window_size);
    end
    fprintf('\n');
    
    N = size(data, 1);
    M = window_size;
    t = size(data, 2);
    
    T = floor((t-1) / window_size);
    X = data(:, 1:end-1);
    X = X(:, 1:M*T);
    Y = data(:, 2:end);
    Y = Y(:, 1:M*T);

    if center == 0
        X = reshape(X, [N, M, T]);
    else
        % append ones to input data
        X = reshape([X; ones(1, M*T)], [N+1, M, T]);
    end
    Y = reshape(Y, [N, M, T]);
    
    if center
        A = zeros(N, N+1, T);
    else
        A = zeros(N, N, T);
    end
    
    for k = 1:T
        A(:, :, k) = Y(:, :, k) * pinv(X(:, :, k));
    end
    
    var_tot = 0;
    var_int = 0;
    var_aff = 0;
    for k = 1:T
        Ypred_k = A(:, :, k) * X(:, :, k);
        var_aff = var_aff + norm(Y(:,:,k) - Ypred_k, 'fro')^2;
        if center
            Ypred_int = A(:, end, k) * ones(1, size(Ypred_k, 2));
            var_int = var_int + norm(Y(:,:,k) - Ypred_int, 'fro')^2;
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
    
    if nargout >= 2
        varargout{1} = X;
    end
    if nargout >= 3
        varargout{2} = Y;
    end
    if nargout >= 4
        varargout{3} = var_aff;
    end
end
