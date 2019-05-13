function D = compute_pdist(A, B, C)
    T = size(C, 1);
    D = zeros(T*(T-1)/2, 1);
    idx = 1;
    for w_i = 1:T
        for w_j = (w_i + 1):T
            Cdiff = diag(C(w_j, :)) - diag(C(w_i, :));
            if size(B, 1) == size(A, 1) + 1
                % affine mode
                D(idx) = ...
                    norm(A * Cdiff * B(1:end-1, :)', ...
                         'fro') + ...
                    norm(Cdiff * B(end, :)');
            elseif size(B, 1) == size(A, 1)
                D(idx) = ...
                    norm(A * Cdiff * B', 'fro');
            else
                error('Dimension mismatch in A and B');
            end
            idx = idx + 1;
        end
    end
end