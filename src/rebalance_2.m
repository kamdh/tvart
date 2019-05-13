function [lambda, Anew, Bnew, Cnew, Wnew] = rebalance_2(A, B, C, W, varargin)
    if nargin > 4
        old_style = varargin{1};
    else
        old_style = 0;
    end
    if nargin > 5
        tol = varargin{2};
    else
        tol = 1e-8;
    end
    r = size(A, 2);
    lambda = zeros(r,1);
    for l = 1:r
        anorm = norm(A(:, l));
        Anew(:, l) = A(:, l) / anorm;
        bnorm = norm(B(:, l));
        Bnew(:, l) = B(:, l) / bnorm;
        cnorm = norm(C(:, l));
        Cnew(:, l) = C(:, l) / cnorm;
        Wnew(:, l) = W(:, l) / cnorm;
        if old_style
            lambda(l) = anorm*bnorm*cnorm;
            if lambda(l) < tol
                Anew(:, l) = 0;
                Bnew(:, l) = 0;
                Cnew(:, l) = 0;
                Wnew(:, l) = 0;
                lambda(l) = 0;
            end
        else
            new_norm = (anorm*bnorm*cnorm)^(1/3);
            Anew(:, l) = Anew(:, l) * new_norm;
            Bnew(:, l) = Bnew(:, l) * new_norm;
            Cnew(:, l) = Cnew(:, l) * new_norm;
            Wnew(:, l) = Wnew(:, l) * new_norm;
            lambda(l) = 1;
        end
    end
    if (size(A, 1) + 1) == size(B, 1)
        % affine
        for l = 1:r
            Anew(:, l) = Anew(:, l) * sign(Bnew(end, l));
            Bnew(end, l) = Bnew(end, l) * sign(Bnew(end, l));
        end
    end
end

% function [lambda, Anew, Bnew, Cnew, Wnew] = rebalance_2(A, B, C, W)
%     r = size(A, 2);
%     lambda = ones(r,1);
%     for l = 1:r
%         anorm = norm(A(:, l));
%         Anew(:, l) = A(:, l) / anorm;
%         bnorm = norm(B(:, l));
%         Bnew(:, l) = B(:, l) / bnorm;
%         cnorm = norm(C(:, l));
%         Cnew(:, l) = C(:, l) * anorm * bnorm;
%         Wnew(:, l) = W(:, l) * anorm * bnorm;
%         % if old_style
%         %     lambda(l) = anorm*bnorm*cnorm;
%         % else
%         %     new_norm = (anorm*bnorm*cnorm)^(1/3);
%         %     Anew(:, l) = Anew(:, l) * new_norm;
%         %     Bnew(:, l) = Bnew(:, l) * new_norm;
%         %     Cnew(:, l) = Cnew(:, l) * new_norm;
%         %     lambda(l) = 1;
%         % end
%     end
%     if (size(A, 1) + 1) == size(B, 1)
%         % affine
%         for l = 1:r
%             Anew(:, l) = Anew(:, l) * sign(Bnew(end, l));
%             Bnew(end, l) = Bnew(end, l) * sign(Bnew(end, l));
%         end
%     end

% end

