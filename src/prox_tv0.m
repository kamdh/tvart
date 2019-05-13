function [y, cost] = prox_tv0(x, beta)
    y = zeros(size(x));
    ipts = findchangepts(x, 'statistic', 'mean', 'minthreshold', beta);
    npts = length(ipts);
    if npts > 0
        for i=1:npts+1
            if i == 1
                idxi = 1:ipts(i)-1;
            elseif i == (npts+1)
                idxi = ipts(i-1):length(x);
            else
                idxi = ipts(i-1):ipts(i)-1;
            end
            y(idxi) = mean(x(idxi));
        end
    else
        y(:) = mean(x);
    end
    cost = npts * beta;
end