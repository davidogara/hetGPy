from __future__ import annotations
import numpy as np
import hetgpy
from hetgpy.covariance_functions import cov_gen
from scipy.stats import norm, t
from scipy.linalg.lapack import dtrtri
def crit_MEE(x, model, thres = 0, preds = None):
    r'''
    Computes MEE infill criterion
    Maximum Empirical Error criterion

    Parameters
    -----------
    x : nd_array
        matrix of new designs, one point per row (size n x d)
    model: `homGP` or `hetGP`
        including inverse matrices
    thres: float 
        for contour finding
    preds: dict 
        optional predictions at `x` to avoid recomputing if already done

    References
    ----------

    Ranjan, P., Bingham, D. & Michailidis, G (2008). 
    Sequential experiment design for contour estimation from complex computer codes, 
    Technometrics, 50, pp. 527-541. \cr \cr

    Bichon, B., Eldred, M., Swiler, L., Mahadevan, S. & McFarland, J. (2008).
    Efficient global  reliability  analysis  for  nonlinear  implicit  performance  functions, 
    AIAA Journal, 46, pp. 2459-2468. \cr \cr

    Lyu, X., Binois, M. & Ludkovski, M. (2018+). Evaluating Gaussian Process Metamodels and Sequential Designs for Noisy Level Set Estimation. arXiv:1807.06712.
    '''
    if len(x.shape)==1: x = x.reshape(-1,1)
    if preds is None: preds = model.predict(x = x)
    ## TP case
    if type(model) == hetgpy.homGP.homTP or type(model)==hetgpy.hetGP.hetTP: 
        return t.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(preds['sd2']), df = model.nu + len(model.Z))
    ## GP case
    return norm.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(preds['sd2']))

def crit_cSUR(x, model, thres = 0, preds = None):
    r'''
    Computes cSUR infill criterion
    
    Contour Stepwise Uncertainty Reduction criterion
    
    Parameters
    -----------
    x : nd_array
        matrix of new designs, one point per row (size n x d)
    model: `homGP` or `hetGP`
        including inverse matrices
    thres: float 
        for contour finding
    preds: dict 
        optional predictions at `x` to avoid recomputing if already done
    
    References
    ----------
    Lyu, X., Binois, M. & Ludkovski, M. (2018+). Evaluating Gaussian Process Metamodels and Sequential Designs for Noisy Level Set Estimation. arXiv:1807.06712.
    '''
    
    if len(x.shape)==1: x = x.reshape(-1,1)
    if preds is None: preds = model.predict(x = x, xprime = x)
    if type(model) == hetgpy.homGP.homTP or type(model)==hetgpy.hetGP.hetTP:
    
        # unscale the predictive variance and covariance (e.g., go back to the GP case)
        # (since psi is updated separately)
        pcov = preds['cov'] * (model.nu + len(model.Z) - 2) / (model.nu + model.psi - 2)
        psd2 = preds['sd2'] * (model.nu + len(model.Z) - 2) / (model.nu + model.psi - 2)

        Ki_new = np.linalg.cholesky(pcov + np.diag(model.eps + preds['nugs'])).T
        Ki_new = dtrtri(Ki_new)[0]
        Ki_new = Ki_new @ Ki_new.T
        sd2_new = psd2 - np.diag(pcov @ (Ki_new @ pcov.T))
        sd2_new[sd2_new<0] = 0

        psi_n1 = model.psi + model.nu/(model.nu - 2)

        sd2_new = (model.nu + psi_n1 - 2) / (model.nu + len(model.Z) - 1) * sd2_new # unscaled variance

        return(t.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(preds['sd2']), df = model.nu + len(model.Z)) - 
                    t.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(sd2_new), df = model.nu + len(model.Z) + 1))
    else:
    
        Ki_new = np.linalg.cholesky(preds['cov'] + np.diag(model.eps + preds['nugs'])).T
        Ki_new = dtrtri(Ki_new)[0]
        Ki_new = Ki_new @ Ki_new.T
        sd2_new = preds['sd2'] - np.diag(preds['cov'] @ (Ki_new @ preds['cov'].T))
        sd2_new[sd2_new<0] = 0

        return(norm.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(preds['sd2'])) - norm.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(sd2_new)))
    
def crit_ICU(x, model, thres = 0, Xref = None, w = None, preds = None, kxprime = None):
    r'''
    Computes ICU infill criterion
    
    Integrated Contour Uncertainty criterion

    Parameters
    ----------
    x : nd_array
        matrix of new designs, one point per row (size n x d)
    model: `homGP` or `hetGP`
        including inverse matrices
    Xref : nd_array
        matrix of input locations to approximate the integral by a sum
    w  : nd_array
        optional weights vector of weights for \code{Xref} locations
    thres: float
        for contour finding
    preds: dict
        optional predictions at `Xref` to avoid recomputing if already done
    kxprime: nd_array 
        optional covariance matrix between `model.X0` and `Xref` to avoid its recomputation
    
    References
    ----------
    Lyu, X., Binois, M. & Ludkovski, M. (2018+). Evaluating Gaussian Process Metamodels and Sequential Designs for Noisy Level Set Estimation. arXiv:1807.06712.
    '''
    if len(x.shape)==1: x = x.reshape(-1,1)
    if preds is None: preds = model.predict(x = Xref)
    if w is None: w = np.repeat(1, Xref.shape[0])

    predx = model.predict(x = x)
    if kxprime is None:
        covnew = model.predict(x = x, xprime = Xref)['cov']
    else:
        if type(model)==hetgpy.homTP or type(model)==hetgpy.hetTP:
            kxprime = kxprime * model.sigma2
            kx = model.sigma2 * cov_gen(X1 = x, X2 = model.X0, theta = model.theta, type = model.covtype)
            covnew = (model.nu + model.psi - 2) / (model.nu + len(model.Z) - 2) * (model.sigma2 * cov_gen(X1 = x, X2 = Xref, theta = model.theta, type = model.covtype) - (kx @ model.Ki) @ kxprime)
        else:
            kx = model.nu_hat * cov_gen(X1 = x, X2 = model.X0, theta = model.theta, type = model.covtype)
            model.Ki = model.Ki / model.nu_hat
            kxprime = kxprime * model.nu_hat
            if model.trendtype == 'SK':
                covnew = model.nu_hat * cov_gen(X1 = x, X2 = Xref, theta = model.theta, type = model.covtype) - (kx @ model.Ki) @ kxprime
            else:
                covnew = model.nu_hat * cov_gen(X1 = x, X2 = Xref, theta = model.theta, type = model.covtype) - (kx @ model.Ki) @ kxprime + (1 - (np.sum(model.Ki,axis=0,keepdims=True)@ kx.T)).T @ (1 - np.sum(model.Ki,axis=0,keepdims=True) @ kxprime)/np.sum(model.Ki)

    if type(model)==hetgpy.homTP or type(model)==hetgpy.hetTP:
      
        # unscale the predictive variance and covariances (e.g., go back to the GP case)
        # (since psi is updated separately)
        predx['sd2'] = (model.nu + len(model.Z) - 2) / (model.nu + model.psi - 2) * predx['sd2']
        covnew = (model.nu + len(model.Z) - 2) / (model.nu + model.psi - 2) * covnew
        preds['sd2'] = (model.nu + len(model.Z) - 2) / (model.nu + model.psi - 2) * preds['sd2']
        
        sd2_new = preds['sd2'] - np.squeeze(covnew**2)/(predx['sd2'] + predx['nugs'] + model.eps)
        sd2_new[sd2_new<0] = 0
        
        # now update psi
        psi_n1 = model.psi + model.nu/(model.nu - 2)
        
        # now rescale with updated psi
        sd2_new = (model.nu + psi_n1 - 2) / (model.nu + len(model.Z) - 1) * sd2_new
        
        return(- np.sum(w * t.pdf(-np.abs(preds['mean'] - thres)/np.sqrt(sd2_new), df = model.nu + len(model.Z) + 1)))
    else:
        sd2_new <- preds['sd2'] - np.squeeze(covnew**2)/(predx['sd2'] + predx['nugs'] + model.eps)
        sd2_new[sd2_new<0] = 0
        return - np.sum(w * norm.pdf(-np.abs(preds['mean'] - thres)/np.sqrt(sd2_new)))

def crit_tMSE(x, model, thres = 0, preds = None, seps = 0.05):
    r'''
    Computes targeted mean squared error infill criterion
    
    t-MSE criterion

    Parameters
    ----------
    x : nd_array
        matrix of new designs, one point per row (size n x d)
    model: `homGP` or `hetGP`
        including inverse matrices
    thres: float
        for contour finding
    preds: dict
        optional predictions at `x` to avoid recomputing if already done (must contain `cov`)
    seps: float
        parameter for the target window
    
    References
    ----------
    
    Picheny, V., Ginsbourger, D., Roustant, O., Haftka, R., Kim, N. (2010).
    Adaptive designs of experiments for accurate approximation of a target region,
    Journal of Mechanical Design (132), p. 071008.

    Lyu, X., Binois, M. & Ludkovski, M. (2018+). 
    Evaluating Gaussian Process Metamodels and Sequential Designs for Noisy Level Set Estimation. arXiv:1807.06712.
    '''
    if len(x.shape)==1: x = x.reshape(-1,1)
    if preds is None or preds.get('cov') is None: preds = model.predict(x = x, xprime = x)

    w = 1/np.sqrt(2 * np.pi * (preds['sd2']+ seps)) * np.exp(-0.5 * (preds['mean'] - thres)**2 / (preds['sd2'] + seps))

    return w * preds['sd2']

def crit_MCU(x, model, thres = 0, gamma = 2, preds = None):
    r'''
    Parameters
    ----------
    x : nd_array
        matrix of new designs, one point per row (size n x d)
    model: `homGP` or `hetGP`
        including inverse matrices
    thres: float
        for contour finding
    gamma: float
        optional weight in -|f(x) - thres| + gamma * s(x). Default to 2
    preds: dict
        optional predictions at `x` to avoid recomputing if already done (must contain `cov`)
    
    References
    ----------
    Srinivas, N., Krause, A., Kakade, S, & Seeger, M. (2012). 
    Information-theoretic regret bounds for Gaussian process optimization 
    in the bandit setting, IEEE Transactions on Information Theory, 58, pp. 3250-3265.

    Bogunovic, J., Scarlett, J., Krause, A. & Cevher, V. (2016). 
    Truncated variance reduction: A unified approach to Bayesian optimization and level-set estimation,
    in Advances in neural information processing systems, pp. 1507-1515.

    Lyu, X., Binois, M. & Ludkovski, M. (2018+). 
    Evaluating Gaussian Process Metamodels and Sequential Designs for Noisy Level Set Estimation. arXiv:1807.06712.
    '''
    if len(x.shape)==1: x = x.reshape(-1,1)
    if preds is None: preds = model.predict(x = x)


    ## TP case
    if type(model)==hetgpy.homTP or type(model)==hetgpy.hetTP:
        return(-np.abs(preds['mean'] - thres) + gamma * np.sqrt(preds['sd2']))

    ## GP case
    return -np.abs(preds['mean'] - thres) + gamma * np.sqrt(preds['sd2'])