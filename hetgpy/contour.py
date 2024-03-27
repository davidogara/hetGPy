import numpy as np
import hetgpy
from scipy.stats import norm, t
from scipy.linalg.lapack import dtrtri
def crit_MEE(x, model, thres = 0, preds = None):
    '''
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
    
        Ki_new = np.linalg.cholesky(pcov + np.diag(model.eps + preds['nugs'])).T
        Ki_new = dtrtri(Ki_new)[0]
        Ki_new = Ki_new @ Ki_new.T
        sd2_new = psd2 - np.diag(pcov @ (Ki_new @ pcov.T))
        sd2_new[sd2_new<0] = 0

        return(norm.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(preds['sd2'])) - norm.cdf(-np.abs(preds['mean'] - thres)/np.sqrt(sd2_new)))