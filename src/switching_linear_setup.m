function [X, A1, A2] = switching_linear_setup(N, num_steps, sigma, offset)
    num_trans = 200;
    noise_process = 0.0;
    noise_compensate = 0;

    x0 = ones(N,1) / sqrt(N);
    t1 = 0.1 * pi;
    A1 = [cos(t1) -sin(t1);
          sin(t1)  cos(t1)];
    %t2 = rand()*pi;
    t2 = 0.37 * pi;
    A2 = [cos(t2) -sin(t2);
          sin(t2)  cos(t2)];
    if N > 2
        %A1 = randn(N,N)/sqrt(N);
        %A2 = randn(N,N)/sqrt(N);
        %A1 = (2*rand(N,r)-1)*(2*rand(r,N)-1) / sqrt(N);
        %A2 = (2*rand(N,r)-1)*(2*rand(r,N)-1) / sqrt(N);
        % A1 = randn(N,r)*randn(r,N) / sqrt(r*N);
        % A2 = randn(N,r)*randn(r,N) / sqrt(r*N);
        % A1 = A1 / (max(abs(eig(A1))) );
        % A2 = A2 / (max(abs(eig(A2))) );
        U = random_orthogonal(N, 2);
        A1 = U * A1 * U';
        U =  random_orthogonal(N, 2);
        A2 = U * A2 * U';
    end

    %A2 = A1;

    % disp('eigs of A1')
    % eig(A1)
    % disp('eigs of A2')
    % eig(A2)

    if offset
        %% add offsets
        b1 = 4+rand(N,1);
        b2 = -1-rand(N,1);
    end
    % transient removal
    for i = 1:num_trans
        x0 = A1 * x0;
        if i == 1
            x0 = x0 / norm(x0) * sqrt(N);
        end
    end

    %% integrate linear system
    X = zeros(N, num_steps);
    X(:, 1) = x0;
    for i = 2:num_steps
        if i < (num_steps)/2
            X(:, i) = A1 * X(:, i-1) + randn(N,1) * noise_process;
        else
            X(:, i) = A2 * X(:, i-1) + randn(N,1) * noise_process;
            if i == ceil((num_steps)/2)
                X(:, i) = X(:, i) / norm(X(:, i)) * sqrt(N);
            end
        end
    end
    if offset
        for i = 2:num_steps
            if i < (num_steps)/2
                X(:, i) = X(:, i) + b1;
            else
                X(:, i) = X(:, i) + b2;
            end
        end
    end
    %% add noise
    X = X + randn(size(X)) * sigma; % / sqrt(N);
