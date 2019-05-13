function [lambda, A, B, C, varargout] = ...
        reorder_components(lambda, A, B, C, varargin)
    [~,I] = sort(lambda, 'descend');
    A = A(:,I);
    B = B(:,I);
    C = C(:,I);
    if nargin > 4
        W = varargin{1};
        W = W(:,I);
    end
    lambda = lambda(I);
    for i=1:size(C,2)
        if mean(C(:,i)) < 0
            C(:,i) = C(:,i) * -1;
            A(:,i) = A(:,i) * -1;
            if nargin > 4
                W(:,i) = W(:,i) * -1;
            end
        end
    end
    varargout = {};
    if nargin > 4
        varargout{1} = W;
    end
end
