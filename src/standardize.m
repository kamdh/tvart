function Y = standardize(X);
% Y = X - repmat(mean(X, 1), size(X,1), 1);
    Y = X - repmat(mean(X, 2), 1, size(X, 2));
    Y = Y ./ repmat(std(Y, 0, 2), 1, size(X, 2));
end
