function [X, thetas, U] = smooth_linear_setup(N, num_steps, sigma)
    num_trans = 200;
    ts = 1:(num_steps + num_trans);
    tau = length(ts);
    K = zeros(tau, tau);
    for i = 1:tau
        for j = 1:tau
            K(i,j) = exp(-(ts(i) - ts(j))^2 / (30)^2);
            if i == j
                K(i,j) = K(i,j) + 0.001;
            end
        end
    end
    R = chol(K);
    thetas = R * randn(tau, 1);
    
    if N > 2
        U = random_orthogonal(N, 2);
    else
        U = eye(2);
    end

    x = ones(N,1);
    X = zeros(N, num_steps);
    % transient removal
    for i = 1:tau
        Ar = [cos(thetas(i)) -sin(thetas(i));
             sin(thetas(i))  cos(thetas(i))];
        A = U * Ar * U';
        x = A * x;
        if i == 1
            x = x / norm(x) * sqrt(N);
        end
        if i > num_trans
            X(:, i - num_trans) = x;
        end
    end
    %% add noise
    X = X + randn(size(X)) * sigma; % / sqrt(N);
    thetas = thetas(1+num_trans:end);