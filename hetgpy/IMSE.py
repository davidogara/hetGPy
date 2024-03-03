
import numpy as np
from hetgpy.src import EMSE
from hetgpy import hetGP, homGP
from hetgpy.covariance_functions import cov_gen
from scipy.linalg.lapack import dtrtri

TYPE = type
def IMSPE(model, theta = None, Lambda = None, mult = None, covtype = None, nu= None, eps = np.sqrt(np.finfo(float).eps)):
    '''
    IMSPE of a given design.

    Integrated Mean Square Prediction Error

    Parameters
    ----------

    X : ``hetgpy.hetGP.hetGP`` or ``hetgpy.homGP.homGP`` model. 
        Alternatively, one can provide a matrix of unique designs considered
    theta : ndarray_like
        lengthscales
    Lambda : ndarray_like
        diagonal matrix for the noise
    mult : ndarray_like
        number of replicates at each design
    covtype : str
        either "Gaussian", "Matern3_2" or "Matern5_2"
    nu : float
        variance parameter
    eps : float
        numerical nugget
    
    Details
    -------
    One can provide directly a model of class ``hetGP`` or ``homGP``, or provide design locations ``X`` and all other arguments
    '''
    if type(model)==hetGP.hetGP or type(model)==homGP.homGP:
        Wij = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)
        if model.trendtype == "OK":
            tmp = np.squeeze(1 - 2 * model.Ki.sum(axis=0) @ mi(mu1 = model.X0, theta = model.theta, type = model.covtype) + model.Ki.sum(axis=0) @ Wij @ model.Ki(axis=1)/model.Ki.sum())
        else:
            tmp = 0
        return model.nu_hat * (1 - np.sum(model.Ki * Wij) + tmp)
    else:
        C = cov_gen(model, theta = theta, type = covtype)
        Ki = np.linalg.cholesky(C + np.diag(Lambda/mult + eps)).T
        Ki = dtrtri(Ki)[0]
        Ki = Ki @ Ki.T
        return nu * (1 - np.sum(Ki * Wij(mu1 = model, theta = theta, type = covtype)))
    
def crit_IMSPE(x, model, id = None, Wijs = None):
    ## Precalculations
  
    if Wijs is None: Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)


    if id is not None:
        if model.Lambda is None:
            tmp = model.g
        else:
            tmp = model.Lambda[id]
        
        return 1 - (np.sum(model.Ki*Wijs) + model.Ki[:,id].T @ Wijs @ model.Ki[:,id] / ((model.mult[id]*(model.mult[id] + 1)) / (1*tmp) - model.Ki[id, id]))
    

    if len(x.shape)==1:
        x = x.reshape(-1,1)
    newWijs = Wij(mu2 = x, mu1 = model.X0, theta = model.theta, type = model.covtype)
    W11 = Wij(mu2 = x, mu1 = x, theta = model.theta, type = model.covtype)

    kn1 = cov_gen(x, model.X0, theta = model.theta, type = model.covtype)
    
    new_lambda = model.predict(x = x, nugs_only = True)['nugs']/model.nu_hat
    
    vn = np.squeeze(1 - kn1 @ model.Ki @ kn1.T) + new_lambda + model.eps
    gn = - 1.0 * (model.Ki @ kn1.T) / vn
    return 1 - (np.sum(model.Ki*Wijs) + gn.T @ (Wijs @ gn)*vn + 2 * newWijs.T @ gn + W11 / vn)

def Wij(mu1, mu2 = None,  theta = None, type = "Gaussian"):
    '''
    Compute double integral of the covariance kernel over a [0,1]^d domain

    Parameters
    ----------
    mu1,mu2 : ndarray)like
        input locations considered
    theta : ndarray_like
        lengthscale hyperparameter of the kernel
    type : str 
        kernel type, one of ``"Gaussian"``, ``"Matern5_2"`` or ``"Matern3_2"``
    
    References
    ----------
    M. Binois, J. Huang, R. B. Gramacy, M. Ludkovski (2019), Replication or exploration? Sequential design for stochastic simulation experiments,
    Technometrics, 61(1), 7-23. Preprint available on arXiv:1710.03206.
    '''
    if theta is None: raise ValueError("theta required")

    if mu1.shape[1] > 1 and len(theta) == 1: theta = np.repeat(theta, mu1.shape[1])

    if type == "Gaussian":
        if mu2 is None:
            return EMSE.Wijs_gauss_sym_cpp(mu1, np.sqrt(theta))
        return EMSE.Wijs_gauss_cpp(mu1, mu2, np.sqrt(theta))
    if(type == "Matern5_2"):
        if mu2 is None:
            return EMSE.Wijs_mat52_sym_cpp(mu1, theta)
        return EMSE.Wijs_mat52_cpp(mu1, mu2, theta)

    if(type == "Matern3_2"):
        if mu2 is None:
            return EMSE.Wijs_mat32_sym_cpp(mu1, theta)
        return EMSE.Wijs_mat32_cpp(mu1, mu2, theta)


def mi(mu1, theta, type):
    '''
    Compute integral of the covariance kernel over a [0,1]^d domain

    Parameters
    ----------
    mu1 : ndarray_like
        input locations considered
    theta : ndarray_like
        lengthscale hyperparameter of the kernel
    type : str 
        kernel type, one of ``"Gaussian"``, ``"Matern5_2"`` or ``"Matern3_2"``
    
    References
    ----------
    Replication or exploration? Sequential design for stochastic simulation experiments,
    Technometrics, 61(1), 7-23. Preprint available on arXiv:1710.03206.
    '''
    if mu1.shape[1] > 1 and len(theta) == 1:
        theta = np.repeat(theta, mu1.shape[1])

    if type == "Gaussian":
        return EMSE.mi_gauss_cpp(mu1, np.sqrt(theta))

    if type == "Matern5_2":
        return EMSE.mi_mat52_cpp(mu1, theta)

    if type == "Matern3_2":
        return EMSE.mi_mat32_cpp(mu1, theta)

## Wrapper functions

def d1(X, x, sigma, type):
  if type == "Gaussian":
    return EMSE.d_gauss_cpp(X = X, x = x, sigma = sigma)
  
  if type == "Matern5_2":
    return EMSE.d_mat52_cpp(X = X, x = x, sigma = sigma)
  
  if type == "Matern3_2":
    return EMSE.d_mat32_cpp(X = X, x = x, sigma = sigma)
  

def c1(X, x, sigma, W, type):
  if type == "Gaussian":
    return EMSE.c1_gauss_cpp(X = X, x = x, sigma = np.sqrt(sigma), W = W)
  
  if type == "Matern5_2":
    return EMSE.c1_mat52_cpp(X = X, x = x, sigma = sigma, W = W)
  
  if type == "Matern3_2":
    return EMSE.c1_mat32_cpp(X = X, x = x, sigma = sigma, W = W)
  
  
def c2(x, sigma, w, type):
  if type == "Gaussian":
    return EMSE.c2_gauss_cpp(x = x, t = np.sqrt(sigma), w = w)
  
  if type == "Matern5_2":
    return EMSE.c2_mat52_cpp(x = x, t = sigma, w = w)
  
  if type == "Matern3_2":
    return EMSE.c2_mat32_cpp(x = x, t = sigma, w = w)


def deriv_crit_IMSPE(x, model, Wijs = None):
    '''
    Derivative of crit_IMSPE
    
    Parameters
    ----------
    
    x : ndarray_like
        matrix for the news design (size 1 x d)
    model : hetGP or homGP
        model
    Wijs : ndarray_like
        optional previously computed matrix of Wijs, see Wij

    Returns
    -------
    
    Derivative of the sequential IMSPE with respect to x
    '''
  
    ## Precalculations
    if Wijs is None: Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)

    if len(x.shape) == 1: x = x.reshape(1,-1)
    kn1 = cov_gen(model.X0, x, theta = model.theta, type = model.covtype)
    
    if TYPE(model)==hetGP: kng1 = cov_gen(model.X0, x, theta = model.theta_g, type = model.covtype)
    new_lambda = model.predict(x = x, nugs_only = True).nugs/model.nu_hat
    k11 = 1 + new_lambda

    W1 = Wij(mu1 = model.X, mu2 = x, theta = model.theta, type = model.covtype)
    W11 = Wij(mu1 = x, theta = model.theta, type = model.covtype)
  
    # compute derivative vectors and scalars
    Kikn1 =  model.Ki @ kn1
    v = np.squeeze(k11 - (kn1.T @ Kikn1))
    g =  - Kikn1 / v
    
    tmp = np.repeat(np.nan, x.shape[1])
    dlambda = 0
    if TYPE(model)==hetGP: KgiD = model.Kgi @ (model.Delta - model.nmean)
    if model.theta.shape[0] < x.shape[1]: model.theta = np.repeat(model.theta, x.shape[1])
    if TYPE(model)==hetGP and model.theta_g.shape[0] < x.shape[1]: model.theta_g = np.repeat(model.theta, x.shape[1])
    
    Wig = Wijs @ g
  
    for m in range(x.shape[1]):
    
        c1_v = c1(X = model.X0[:,m], x = x[:,m], sigma = model.theta[m], W = W1, type = model.covtype)
    
        dis = np.squeeze(d1(X = model.X0[:,m], x = x[:,m], sigma = model.theta[m], type = model.covtype) * kn1)
    
        if TYPE(model)==hetGP:
            dlambda = (d1(X = model.X0[:,m], x = x[:,m], sigma = model.theta_g[m], type = model.covtype) * kng1).T @ KgiD
            if model.logN:
                dlambda = new_lambda *dlambda
    
    v2 = np.squeeze(2 * (- dis @ Kikn1) + dlambda) 
    v1 = -1/v^2 * v2
    h = np.squeeze(-model.Ki @ (v1 * kn1 + 1/v * dis))
    
    tmp[m] <- 2 * (c1_v.T @ g) + c2(x = x[:,m], sigma = model.theta[m], w = W11, type = model.covtype) / v + ((v2 * g + 2 * v * h).T @ Wig) + 2 * (h.T @ W1) + v1 * W11
    return -tmp