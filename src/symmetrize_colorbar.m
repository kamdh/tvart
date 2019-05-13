function symmetrize_colorbar()
    ax = gca();
    cax = caxis();
    lim = max(abs(cax));
    caxis([-lim, lim]);
end