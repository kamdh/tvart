import ssm
import numpy as np
from numpy.linalg import norm, svd
from ssm.util import find_permutation
from scipy.optimize import curve_fit, fsolve

def fit_arhmm_and_return_errors(X, A1, A2, Kmax=4, num_restarts=1,
                                    num_iters=100, rank=None):
    '''
    Fit an ARHMM to test data and return errors.
    
    Parameters
    ==========
    
    X : array, T x N
    A1 : array, N x N
    A2 : array, N x N
    '''
    # hardcoded
    true_K = 2
    # params
    N = X.shape[1]
    T = X.shape[0]

    if rank is not None:
        # project data down
        u, s, vt = np.linalg.svd(X)
        Xp = u[:, 0:rank] * s[0:rank] # T x rank matrix
    else:
        Xp = X

    def _fit_once():
        # fit a model
        if rank is not None:
            arhmm = ssm.HMM(Kmax, rank, observations="ar")
        else:
            arhmm = ssm.HMM(Kmax, N, observations="ar")
        lls = arhmm.fit(Xp, num_iters=num_iters)
        return arhmm, lls

    # Fit num_restarts many models
    results = []
    for restart in range(num_restarts):
        print("restart ", restart + 1, " / ", num_restarts)
        results.append(_fit_once())
    arhmms, llss = list(zip(*results))

    # Take the ARHMM that achieved the highest training ELBO
    best = np.argmax([lls[-1] for lls in llss])
    arhmm, lls = arhmms[best], llss[best]

    # xhat = arhmm.smooth(X)
    pred_states = arhmm.most_likely_states(Xp)

    # Align the labels between true and most likely
    true_states = np.array([0 if i < T/2 else 1 for i in range(T)])
    arhmm.permute(find_permutation(true_states, pred_states,
                                      true_K, Kmax))
    print("predicted states:")
    print(pred_states)    
    # extract predicted A1, A2 matrices
    Ahats, bhats = arhmm.observations.As, arhmm.observations.bs
    if rank is not None:
        # project back up
        Ahats = [ vt[0:rank, :].T @ Ahat @ vt[0:rank, :] for Ahat in Ahats ]
        bhats = [ vt[0:rank, :].T @ bhat for bhat in bhats ]
    
    # A_r = slds.dynamics.As
    # b_r = slds.dynamics.bs
    # Cs = slds.emissions.Cs[0]
    # A1_pred = Cs @ A_r[0] @ np.linalg.pinv(Cs)
    # A2_pred = Cs @ A_r[1] @ np.linalg.pinv(Cs)
    # compare inferred and true
    #err_inf = 0.5 * (np.max(np.abs(A1_pred[:] - A1[:])) + \
    #                 np.max(np.abs(A2_pred[:] - A2[:])))
    #err_2 = 0.5 * (norm(A1_pred - A1, 2) + \
    #               norm(A2_pred - A2, 2))
    #err_fro = 0.5 * (norm(A1_pred - A1, 'fro') + \
    #                 norm(A2_pred - A2, 'fro'))
    err_mse, err_inf, err_2, err_fro = errors(Ahats, bhats, pred_states, true_states, A1, A2, X)
    return (err_inf, err_2, err_fro, err_mse, lls)
    
def fit_slds_and_return_errors(X, A1, A2, Kmax=4, r=6, num_iters=200,
                                   num_restarts=1,
                                   laplace_em=True,
                                   single_subspace=True,
                                   use_ds=True):
    '''
    Fit an SLDS to test data and return errors.
    
    Parameters
    ==========
    
    X : array, T x N
    A1 : array, N x N
    A2 : array, N x N
    '''
    # hardcoded
    true_K = 2
    # params
    N = X.shape[1]
    T = X.shape[0]

    def _fit_once():
        # fit a model
        slds = ssm.SLDS(N, Kmax, r, single_subspace=single_subspace,
                            emissions='gaussian')
        #slds.initialize(X)
        #q_mf = SLDSMeanFieldVariationalPosterior(slds, X)
        if laplace_em:
            elbos, posterior = slds.fit(X, num_iters=num_iters, initialize=True,
                                    method="laplace_em",
                                    variational_posterior="structured_meanfield")
            posterior_x = posterior.mean_continuous_states[0]
        else:
            # Use blackbox + meanfield
            elbos, posterior = slds.fit(X, num_iters=num_iters, initialize=True,
                                            method="bbvi",
                                            variational_posterior="mf")
            # predict states
        return slds, elbos, posterior

    # Fit num_restarts many models
    results = []
    for restart in range(num_restarts):
        print("restart ", restart + 1, " / ", num_restarts)
        results.append(_fit_once())
    sldss, elboss, posteriors = list(zip(*results))

    # Take the SLDS that achieved the highest training ELBO
    best = np.argmax([elbos[-1] for elbos in elboss])
    slds, elbos, posterior = sldss[best], elboss[best], posteriors[best]

    if laplace_em:
        posterior_x = posterior.mean_continuous_states[0]
    else:
        posterior_x = posterior.mean[0]

    # Align the labels between true and most likely
    true_states = np.array([0 if i < T/2 else 1 for i in range(T)])
    slds.permute(find_permutation(true_states,
                                      slds.most_likely_states(posterior_x, X),
                                      true_K, Kmax))
    pred_states = slds.most_likely_states(posterior_x, X)
    print("predicted states:")
    print(pred_states)
    # extract predicted A1, A2 matrices
    Ahats, bhats = convert_slds_to_tvart(slds)
    # A_r = slds.dynamics.As
    # b_r = slds.dynamics.bs
    # Cs = slds.emissions.Cs[0]
    # A1_pred = Cs @ A_r[0] @ np.linalg.pinv(Cs)
    # A2_pred = Cs @ A_r[1] @ np.linalg.pinv(Cs)
    # compare inferred and true
    #err_inf = 0.5 * (np.max(np.abs(A1_pred[:] - A1[:])) + \
    #                 np.max(np.abs(A2_pred[:] - A2[:])))
    #err_2 = 0.5 * (norm(A1_pred - A1, 2) + \
    #               norm(A2_pred - A2, 2))
    #err_fro = 0.5 * (norm(A1_pred - A1, 'fro') + \
    #                 norm(A2_pred - A2, 'fro'))
    err_mse, err_inf, err_2, err_fro = errors(Ahats, bhats, pred_states, true_states, A1, A2, X)
    return (err_inf, err_2, err_fro, err_mse, elbos)

def find_final_iterate(data, rtol):
    # Used for estimating convergence in older version
    # Runtime comparisons were performed by hand in final
    def sigmoid (x, A, x0, slope, C):
        return 1 / (1 + np.exp ((x0 - x) / slope)) *  A + C

    x = np.arange(len(data))
    y = data / np.std(data)

    pinit = [np.max(y), np.median(x), 1, np.min(y)]
    popt, pcov = curve_fit(sigmoid, x, y, pinit, maxfev=10000)
    
    fmax = popt[0] + popt[3]
    if fmax < 0:
        thresh = fmax * (1 + rtol)
    else:
        thresh = fmax * (1 - rtol)
        #thresh = popt[3] + 0.999 * popt[0]
    f = lambda x: sigmoid(x, *popt) - thresh
    maxit = int(fsolve(f, len(data)/2)[0])
    return maxit

def errors(Ahats, bhats, pred_states, true_states, A1, A2, X):
    # params
    N = X.shape[1]
    T = X.shape[0]
    assert len(pred_states) == T, "pred_states must be length T"
    assert len(true_states) == T, "true_states must be length T"
    N = X.shape[0]
    err_mse = 0.
    err_inf = 0.
    err_2 = 0.
    err_fro = 0.
    for t in range(T - 1):
        if true_states[t] == 0:
            A_true = A1
        else:
            A_true = A2
        A_pred = Ahats[pred_states[t]]
        b_pred = bhats[pred_states[t]]
        xpred = A_pred @ X[t, :].T + b_pred
        # A_r = slds.dynamics.As[pred_states[t]]
        # A_pred = Cs @ A_r @ np.linalg.pinv(Cs)
        # xpred = A_pred @ X[t, :].T + Cs @ b_r[pred_states[t]]
        err_mse += norm(xpred - X[t+1, :], 2)**2
        err_inf += np.max(np.abs(A_pred[:] - A_true[:]))
        err_2 += norm(A_pred - A_true, 2)
        err_fro += norm(A_pred - A_true, 'fro')
    err_mse /= float(N * (T - 1.))
    err_inf /= float(T - 1.)
    err_2 /= float(T - 1.)
    err_fro /= float(T - 1.)
    return err_mse, err_inf, err_2, err_fro

def convert_slds_to_tvart(slds, use_ds=True):
    # This code modified from that provided by Scott Linderman
    # Compare the true and inferred parameters
    Cs, ds = slds.emissions.Cs, slds.emissions.ds
    As, bs = slds.dynamics.As,  slds.dynamics.bs
    single_subspace = slds.emissions.single_subspace
    
    # Use the pseudoinverse of C to project down to latent space
    Cinvs = np.linalg.pinv(Cs, rcond=1e-8)
    
    if single_subspace:
        Cs = np.repeat(Cs, slds.K, axis=0)
        Cinvs = np.repeat(Cinvs, slds.K, axis=0)
        ds = np.repeat(ds, slds.K, axis=0)
    
    # Compute the effective transition operator on the data
    Aeffs = np.matmul(Cs, np.matmul(As, Cinvs))
    # Compute effective affine/intercept term
    if use_ds:
        beffs = ds[:, :, None] - np.matmul(Aeffs, ds[:, :, None]) \
          + np.matmul(Cs, bs[:, :, None])
    else:
        beffs = np.matmul(Cs, bs[:, :, None])
    return Aeffs, beffs
