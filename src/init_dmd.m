function [A, B, C] = init_dmd(X, Y, R, T, varargin)
    % Rfull = 0;
    N = size(X, 1);
    Admd = Y / X;
    [U, S, V] = svd(Admd, 'econ');
    S = diag(S);
    %V = U;
    % [U, D] = eig(Admd);
    % U = real(U);
    % V = U;
    % S = diag(ones(N, 1));
    if R > N
        % warning(['Not implemented for rank > N. Using all ones ' ...
        %          'initialization']);
        % A = repmat(U * diag(sqrt(S)), 1, ceil(R/N));
        % A = A(:, 1:R);
        % B = repmat(V * diag(sqrt(S)), 1, ceil(R/N));
        % B = B(:, 1:R);
        %scale = sqrt(min(S));
        scale = 1;
        A = [U,... % * diag(sqrt(S)), ...
             ones(N, R-size(U, 2)) / sqrt(N) * scale];
        B = [V,... % * diag(sqrt(S)), ...
             ones(N, R-size(V, 2)) / sqrt(N) * scale];
        C = ones(T, R);
         % A = ones(size(Y, 1), R);
        % B = ones(size(X, 1), R);
        % warning('Not implemented for rank > N. Using rand initialization');
        % R = R;
        % R = size(X, 1);
        % A = ones(size(Y, 1), R) + rand(size(Y, 1), R);
        % B = ones(size(X, 1), R) + rand(size(X, 1), R);
        % C = ones(T, R) + rand(T, R);
    else
        % [U, S, V] = svd(X, 'econ');
        % S = diag(S);
        % Ur = U(:, 1:R);
        % Sr = S(1:R);
        % Vr = V(:, 1:R);
        % Srinv = diag(1./Sr);
        % Admd = Y * Vr * Srinv * Ur';
        A = U(:, 1:R); % * diag(sqrt(S(1:R)));
        B = V(:, 1:R); % * diag(sqrt(S(1:R)));
        C = ones(T, R);
    end
    % A = ones(N, R) / sqrt(N);
    % B = ones(N, R) / sqrt(N);
    % C = ones(T, R) / sqrt(T);
    % if nargin > 4
    %     if varargin{1} == 1
    A = A + 0.5 * randn(N, R) / sqrt(N);
    B = B + 0.5 * randn(N, R) / sqrt(N);
    C = C + 0.5 * randn(T, R) / sqrt(T);
    % A = (2 * rand(N, R) - 1) / sqrt(N);
    % B = (2 * rand(N, R) - 1) / sqrt(N);
    % C = (2 * rand(T, R) - 1) / sqrt(T);
    %end
    %end
end