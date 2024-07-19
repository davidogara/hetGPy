
import numpy as np
from hetgpy import EMSE
from hetgpy import hetGP, homGP
from hetgpy.covariance_functions import cov_gen
from scipy.linalg.lapack import dtrtri
from joblib import Parallel, delayed
from hetgpy.covariance_functions import euclidean_dist
from hetgpy.utils import duplicated
from scipy.spatial.distance import pdist
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from copy import deepcopy as copy
import warnings
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
    if type(model)==hetGP or type(model)==homGP:
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
    
def crit_IMSPE(x=None, model=None, id = None, Wijs = None):
    ## Precalculations
  
    if Wijs is None: Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)


    if id is not None:
        if not hasattr(model,'Lambda'):
            tmp = model.g
        else:
            tmp = model.Lambda[id]
        out = 1 - (np.sum(model.Ki*Wijs) + model.Ki[:,id].T @ Wijs @ model.Ki[:,id] / ((model.mult[id]*(model.mult[id] + 1)) / (1*tmp) - model.Ki[id, id]))
        if out < -1e-1:
            warnings.warn(f"Numerical errors caused negative IMSPE at design location {id}, suggest investigating")
        return out

    if len(x.shape)==1:
        x = x.reshape(1,-1)
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
  if len(x.shape)==1: x = x.squeeze()
  if type == "Gaussian":
    return EMSE.d_gauss_cpp(X = X, x = x, sigma = sigma)
  
  if type == "Matern5_2":
    return EMSE.d_mat52_cpp(X = X, x = x, sigma = sigma)
  
  if type == "Matern3_2":
    return EMSE.d_mat32_cpp(X = X, x = x, sigma = sigma)
  

def c1(X, x, sigma, W, type):
  if len(x.shape)==1: x = x.squeeze()
  if type == "Gaussian":
    return EMSE.c1_gauss_cpp(X = X, x = x, sigma = np.sqrt(sigma), W = W)
  
  if type == "Matern5_2":
    return EMSE.c1_mat52_cpp(X = X, x = x, sigma = sigma, W = W)
  
  if type == "Matern3_2":
    return EMSE.c1_mat32_cpp(X = X, x = x, sigma = sigma, W = W)
  
  
def c2(x, sigma, w, type):
  if len(x.shape)==1:     x     = x.squeeze()
  if len(sigma.shape)==1: sigma = sigma.squeeze()
  if w.shape==(1,1):     w     = w.squeeze()
  if type == "Gaussian":
    return EMSE.c2_gauss_cpp(x = x, t = np.sqrt(sigma), w = w)
  
  if type == "Matern5_2":
    return EMSE.c2_mat52_cpp(x = x, t = sigma, w = w)
  
  if type == "Matern3_2":
    return EMSE.c2_mat32_cpp(x = x, t = sigma, w = w)


def deriv_crit_IMSPE(x, model, id = None, Wijs = None):
    '''
    Derivative of crit_IMSPE
    
    Parameters
    ----------
    
    x : ndarray_like
        matrix for the news design (size 1 x d)
    model : hetGP or homGP
        model
    id: None
        None (but included for compatibility with crit_IMSPE input structure so it can be used in minimize)
    Wijs : ndarray_like
        optional previously computed matrix of Wijs, see Wij

    Returns
    -------
    
    Derivative of the sequential IMSPE with respect to x
    '''
  
    ## Precalculations
    if Wijs is None: Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)

    if len(x.shape) == 1: x = x.reshape(1,-1)
    kn1 = cov_gen(model.X0, x, theta = model.theta, type = model.covtype).squeeze()
    
    if TYPE(model)==hetGP: kng1 = cov_gen(model.X0, x, theta = model.theta_g, type = model.covtype).squeeze()
    new_lambda = model.predict(x = x, nugs_only = True)['nugs']/model.nu_hat
    k11 = 1 + new_lambda

    W1 = Wij(mu1 = model.X0, mu2 = x, theta = model.theta, type = model.covtype)
    W11 = Wij(mu1 = x, theta = model.theta, type = model.covtype)
  
    # compute derivative vectors and scalars
    Kikn1 =  model.Ki @ kn1
    v = np.squeeze(k11 - (kn1.T @ Kikn1))
    g =  - Kikn1 / v
    
    tmp = np.repeat(np.nan, x.shape[1]).reshape(-1,1)
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
        v1 = -1/v**2 * v2
        h = np.squeeze(-model.Ki @ (v1 * kn1 + 1/v * dis))
        tmp[m] = 2 * (c1_v.T @ g) + c2(x = x[:,m], sigma = model.theta[m], w = W11, type = model.covtype) / v + ((v2 * g + 2 * v * h).T @ Wig) + 2 * (h.T @ W1) + v1 * W11
    return -tmp



def phiP(design,p=50):
    '''
    Implementation of `phiP.R` from `DiceDesign` (necessary for maximinSA_LHS from DiceDesign which is used by IMSPE_search)
   
    From DiceDesign:
    Compute the phiP criterion (Lp norm of the sum of the inverses of the design inter-point distances)
    Reference: Pronzato, L. and Muller, W.,2012, Design of computer experiments: space filling and beyond, Statistics and Computing, 22:681-701.
    A higher phiP corresponds to a more regular scaterring of design points
   
    Arguments
    ---------
    design : nd_arraylike
        design for a computer experiment
    p    : int
        the "p" in the Lp norm which is taken (default=50) 
    
    Returns
    -------
    fi_p : np.float
        the phiP criterion
    '''
    D = pdist(design)
    D = D**(-p)
    fi_p = np.sum(D)**(1/p)
    return fi_p

def lhs_EP(m, seed = None):
    '''
    From DiceDesign: FUNCTION PERFORMING ELEMENTARY PERMUTATION (EP) IN LHD USED IN SA ALGORITHMS

    Parameters
    ----------
    m : nd_arraylike
        the design
    
    Returns
    -------
    out : tuple
        list including design after EP, ligns and columns defining EP
    '''
    rand = np.random.default_rng(seed)
    G = m
    d = G.shape[1]
    n = G.shape[0]
    ligns  = (rand.uniform(size=2,low=0,high=n)).astype(int)      
    column = (rand.uniform(size=1,low=0,high=d)).astype(int)
    x = G[ligns[0],column]
    G[ligns[0],column] = G[ligns[1],column]
    G[ligns[1],column] = x 
    l = (G,ligns[0],ligns[1],column)   
    return l

def maximinSA_LHS(design,T0=10,c=0.95,it=2000,p=50,profile="GEOM",Imax=100, seed = None):
    '''
    Implementation of maximinSA_LHS from DiceDesign
    Only profile="GEOM" is implemented (like in hetGP)

    #####maximinSA_LHS#####
    #####Maximin LHS VIA SIMULATED ANNEALING OPTIMIZATION#####

    #---------------------------------------------------------------------------|
    #args :  m     : the design                                                 |
    #        T0    : the initial temperature                                    |
    #        c     : parameter regulating the temperature                       |
    #        it    : the number of iterations                                   |
    #        p     : power required in phiP criterion                           |
    #      profile : temperature down profile                                   |
    #                "GEOM" or "GEOM_MORRIS" or "LINEAR". By default : "GEOM"   |
    #output        : a list containing all the input arguments plus:            |
    #       a mindist optimized design                                          |
    #       vector of criterion values along the iterations                     |
    #       vector of temperature values along the iterations                   |
    #       vector of acceptation probability values along the iterations       |
    #depends :  phiP,lhs_EP                                                     |
    #---------------------------------------------------------------------------|
    
    '''
    crit = None ; temp = None ; proba = None
  
    if profile=="GEOM":
        m = design
        i = 0
        T = T0
        fi_p = phiP(m,p)
        crit = fi_p
  
        while T>0 & i<it:       
            
            G = lhs_EP(m)[0]
            fi_p_ep = phiP(G,p)
            
            c = np.clip((fi_p-fi_p_ep)/T,a_min=None,a_max=10) # avoids overflows
            diff = np.minimum(np.exp(c),1)
            if (diff == 1):
                m = G
                fi_p = fi_p_ep
            else:
                Bernoulli = np.random.default_rng(seed).binomial(n=1,size=1,p=diff)
                if Bernoulli==1:
                    m = G
                    fi_p = fi_p_ep
            i = i+1
            crit = (crit,fi_p) ; temp = (temp,T) ; proba = (proba,diff)
            T = (c**i)*(T0)
            
    vals = (design,T0,c,it,p,profile,Imax,m,crit,temp,proba)
    keys = ("InitialDesign","TO","c","it","p","profile","Imax","design","critValues","tempValues","probaValues") 
    
    
    return {k:v for k,v in zip(keys,vals)}

def IMSPE_search(model, replicate = False, Xcand = None, 
                        control = dict(tol_dist = 1e-6, tol_diff = 1e-6, multi_start = 20,
                                       maxit = 100, maximin = True, Xstart = None), Wijs = None, seed = None,
                        ncores = 1):
  # Only search on existing designs
    if(replicate):
        ## Discrete optimization
        res = []
        for i in range(model.X0.shape[0]):
            res.append(
                crit_IMSPE(x = None, model = model, id = i, Wijs=Wijs)
            )

        #res = Parallel(n_jobs=ncores)(
        #delayed(crit_IMSPE)(**kw) 
        #    for kw in inputs
        #)
        return dict(
            par = model.X0[np.argmin(res),:].reshape(1,-1), 
            value = np.min(res), 
            new = False, 
            id = np.argmin(res)
        )
    if control is None:
        control = dict(multi_start = 20, maxit = 100)
  
    if control.get('multi_start') is None:
        control['multi_start'] = 20
  
    if control.get('maxit') is None:
        control['maxit'] = 100
  
    if control.get('maximin') is None:
        control['maximin'] = True
  
    if control.get('tol_dist') is None: control['tol_dist'] = 1e-6
    if control.get('tol_diff') is None: control['tol_diff'] = 1e-6
    
    d = model.X0.shape[1]
  
    ## Precalculations
    if Wijs is None: Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)
    

    ## Optimization
    if Xcand is None:
        ## Continuous optimization
        if control.get('Xstart') is not None:
            Xstart = control['Xstart']
        else:
            if seed is None: 
                seed = np.random.default_rng().choice(2**15) ## To be changed?
            if control['maximin']:
                if(d == 1):
                # perturbed 1d equidistant points
                    Xstart = (np.linspace(1/2, control['multi_start'] -1/2, control['multi_start']) + np.random.default_rng(seed).uniform(size=control['multi_start'], low = -1/2, high = 1/2)).reshape(-1,1)/control['multi_start']
                else:
                    sampler = LatinHypercube(d=d,seed=seed)
                    Xstart = maximinSA_LHS(sampler.random(n=control['multi_start']))['design']
                
            else:
                sampler = LatinHypercube(d=d,seed=seed)
                Xstart = sampler(n=control['multi_start'])
            
        res = dict(par = np.array([np.nan]), value = np.inf, new = np.nan)
        
        def local_opt_fun(i):
            out = minimize(
            x0  = Xstart[i,:],
            fun = crit_IMSPE,
            jac = deriv_crit_IMSPE,
            args = (model, None, Wijs),
            method = "L-BFGS-B", 
            bounds = [(0,1) for _ in range(d)],
            options=dict(maxiter=control['maxit'], #,
                                ftol = control.get('factr',10) * np.finfo(float).eps,#,
                                gtol = control.get('pgtol',0) # should map to pgtol
                                )
            )
            if out['fun']<0:
                foo=1
            python_kws_2_R_kws = {
                'x':'par',
                'fun': 'value',
                'nfev': 'counts'
            }
            for key, val in python_kws_2_R_kws.items():
                out[val] = out[key]
            return out
        
        all_res = Parallel(n_jobs=ncores)(delayed(local_opt_fun)(i) for i in range(Xstart.shape[0]))
        all_res_values = [res.value for res in all_res]
        res_min = np.argmin(all_res_values)
        par = all_res[res_min]['x']
        par[par<0] = 0
        par[par>1] = 1
        res = dict(par = par.reshape(1,-1), 
                    value = all_res[res_min]['value'], new = True, id = None)
        
        
        if control['tol_dist'] > 0 and control['tol_diff'] > 0:
            ## Check if new design is not to close to existing design
            dists = np.sqrt(euclidean_dist(res['par'].reshape(1,-1), model.X0))
            if np.min(dists) < control['tol_dist']:
                argmin = np.unravel_index(np.argmin(dists, axis=None), dists.shape)[0]
                res = dict(par = model.X0[[argmin],:],
                            value = crit_IMSPE(x = model.X0[argmin,:], model = model, id = argmin, Wijs = Wijs),
                            new = False, id = argmin)
            else:
                ## Check if IMSPE difference between replication and new design is significative
                id_closest = np.unravel_index(np.argmin(dists, axis=None), dists.shape)[-1] # closest point to new design
                imspe_rep = crit_IMSPE(model = model, id = id_closest, Wijs = Wijs)
                if (imspe_rep - res['value'])/res['value'] < control['tol_diff']:
                    res = dict(par = model.X0[[id_closest],:],
                            value = imspe_rep,
                            new = False, id = id_closest)
        return res
    
    else:
        ## Discrete optimization
        def crit_IMSPE_mcl(i, model, Wijs, Xcand): 
            return crit_IMSPE(x = Xcand[i,:], model = model, Wijs = Wijs)
        
        inputs = []
        for i in range(Xcand.shape[0]):
            kw = {'model':model,'Wijs':Wijs,'Xcand':Xcand}
            kw[i] = i
            inputs.append(kw)
        res = Parallel(n_jobs=ncores)(
            delayed(crit_IMSPE_mcl)(**kw) 
                for kw in inputs
            )
        
        tmp = (duplicated(np.vstack(model.X0, Xcand[np.argmin(res),:]), fromLast = True))
        if len(tmp) > 0: 
            return(dict(par = Xcand[[np.argmin(res)],:,], value = min(res), new = False, id = tmp))
        return(dict(par = Xcand[[np.argmin(res)],:], value = min(res), new = True, id = None))

def allocate_mult(model = None, N = None, Wijs = None, use_Ki = False):
    '''
    Allocation of replicates on existing design locations, based on (29) from (Ankenman et al, 2010)
   
    Parameters
    ----------
    model: hetGP model
        hetGP model
    N : int
        total budget of replication to allocate
    Wijs: ndarray
        optional previously computed matrix of Wijs, see hetgpy.IMSE.Wij
    use_Ki : bool
        should Ki from model be used?

    Returns
    -------
    vector with approximated best number of replicates per design
    
    References
    ----------
    B. Ankenman, B. Nelson, J. Staum (2010), Stochastic kriging for simulation metamodeling, Operations research, pp. 371--382, 58
    '''
    ## Precalculations
    if Wijs is None: 
        Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)
    
    if use_Ki:
        Ci = np.clip(np.diag(model.Ki @ Wijs @ model.Ki),a_min=0,a_max=None) # clip for numerical imprecision
    else:
        Ci = np.linalg.pinv(cov_gen(model.X0, theta = model.theta, type = model.covtype),rcond=np.sqrt(np.finfo(float).eps))
        Ci = np.clip(np.diag(Ci @ Wijs @ Ci),a_min=0, a_max = None)
    
    if type(model)==hetGP: 
        V = model.Lambda
    else: 
        V = np.repeat(model.g, len(model.Z0))
    
    res = (N * np.sqrt(Ci * V) / np.sum(np.sqrt(Ci * V))).squeeze()
    
    # Now get integer results summing to N
    idxs = np.argsort(np.mod(res,1))[::-1]
    bdg = (N - np.sum(np.floor(res))).astype(int)  # remaining number of points to add after truncating
    
    res = np.floor(res)
    if bdg > 0:
        res[idxs[0:bdg]] = res[idxs[0:bdg]] + 1
    return res

def horizon(model, current_horizon = None, previous_ratio = None, target = None, Wijs = None, seed = None):
    r'''
    Adapt the look-ahead horizon depending on the replicate allocation or a target ratio
    
    Parameters
    ----------
    model : hetGP or homGP model
        hetGP or homGP model
    current_horizon : int
        horizon used for the previous iteration, see details
    previous_ratio : float
        ratio before adding the previous new design
    target : float
        scalar in [0,1] for desired n/N
    
    Wijs : nd_array
        optional previously computed matrix of Wijs, see hetgpy.IMSE.Wij

    Returns
    -------
    Randomly selected horizon for next iteration (adpative) if no target is provided,
    otherwise returns the update horizon value.
    
    Details
    -------
    If target is provided, along with previous_ratio and current_horizon:
    \itemize{
        \item the horizon is increased by one if more replicates are needed but a new ppint has been added at the previous iteration,
        \item the horizon is decreased by one if new points are needed but a replicate has been added at the previous iteration,
        \item otherwise it is unchanged.
    }
    If no target is provided, allocate_mult is used to obtain the best allocation of the existing replicates,
    then the new horizon is sampled from the difference between the actual allocation and the best one, bounded below by 0.
    See (Binois et al. 2017).
    
    References
    ----------
    M. Binois, J. Huang, R. B. Gramacy, M. Ludkovski (2019), 
    Replication or exploration? Sequential design for stochastic simulation experiments,
    Technometrics, 61(1), 7-23.\cr 
    Preprint available on arXiv:1710.03206.
    '''

    rand = np.random.default_rng(seed)
    if target is None:
        mult_star = allocate_mult(model = model, N = model.mult.sum(), Wijs = Wijs)
        tab_input = mult_star - model.mult
        tab_input[tab_input<0] = 0
        u, counts = np.unique(tab_input,return_counts=True)
        return rand.choice(u, prob = counts/counts.sum())
    
    if current_horizon is None or previous_ratio is None:
        raise ValueError("Missing arguments to use target \n")
    
    ratio = len(model.Z0)/len(model.Z)
    
    # Ratio increased while too small
    if ratio < target and ratio < previous_ratio:
        return(max(-1, current_horizon - 1))
    # Ratio decreased while too high
    if ratio > target and ratio > previous_ratio:
        return current_horizon + 1
        
    return current_horizon

def IMSPE_optim(model, h = 2, Xcand = None, control = dict(tol_dist = 1e-6, tol_diff = 1e-6, multi_start = 20, maxit = 100),
                        Wijs = None, seed = None, ncores = 1):
    d = model.X0.shape[1]
    if Xcand is None:
        if model.X0.max() > 1 or model.X0.min() < 0:
            raise ValueError("IMSPE works only with [0,1]^d domain for now.")
    else:
        if np.max(np.vstack([model.X0, Xcand])) > 1: raise ValueError("IMSPE works only with [0,1]^d domain for now.")
        if np.min(np.vstack([model.X0, Xcand])) < 0: raise ValueError("IMSPE works only with [0,1]^d domain for now.")
    

    ## Precalculations
    if Wijs is None: 
        Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)
    ## A) Setting to beat: first new point then replicate h times
    IMSPE_A     = IMSPE_search(model = model, control = control, Xcand = Xcand, Wijs = Wijs, ncores = ncores)
    new_designA = IMSPE_A['par'] ## store first considered design to be added
    path_A      = [IMSPE_A]

    if h > 0:
        newmodelA = copy(model)
        
        if IMSPE_A['new']:
            newWijs = Wij(mu1 = model.X0, mu2 = new_designA, theta = model.theta, type = model.covtype)
            WijsA = np.hstack([Wijs, newWijs])
            WijsA = np.vstack((WijsA, 
                               np.concatenate([newWijs, Wij(IMSPE_A['par'], IMSPE_A['par'], theta = model.theta, type = model.covtype)]).T)
                    )
        else:
            WijsA = Wijs

        for i in range(h):
            newmodelA.update(Xnew = IMSPE_A['par'], Znew = np.array([np.nan]), maxit = 0)
            IMSPE_A = IMSPE_search(model = newmodelA, replicate = True, control = control, Wijs = WijsA, ncores = ncores)
            path_A.append(IMSPE_A)
    
    if h == -1: return dict(par = new_designA, value = IMSPE_A['value'], path = path_A)

    ## B) Now compare with waiting to add new point
    newmodelB = copy(model)
  
    if h == 0:
        IMSPE_B = IMSPE_search(model = newmodelB, replicate = True, control = control, Wijs = Wijs, ncores = ncores)
        new_designB = IMSPE_B['par'] ## store considered design to be added
        
        # search from best replicate
        if Xcand is None:
            IMSPE_C = IMSPE_search(model = newmodelB, Wijs = Wijs,
                                control = dict(Xstart = IMSPE_B['par'], maxit = control['maxit'],
                                            tol_dist = control['tol_dist'], tol_diff = control['tol_diff']), ncores = ncores)
        else:
            IMSPE_C = IMSPE_B
        
        if IMSPE_C['value'] < min(IMSPE_A['value'], IMSPE_B['value']): 
            return dict(par = IMSPE_C['par'], value = IMSPE_C['value'], path = [IMSPE_C])
        
        if IMSPE_B['value'] < IMSPE_A['value']:
            return dict(par = IMSPE_B['par'], value = IMSPE_B['value'], path = [IMSPE_B])
    else:
        for i in range(h):
            ## Add new replicate
            IMSPE_B = IMSPE_search(model = newmodelB, replicate = True, control = control, Wijs = Wijs, ncores = ncores)
      
            if i == 0:
                new_designB = IMSPE_B['par'].reshape(1,-1) ##store first considered design to add
                path_B = list()
      
            path_B.append(IMSPE_B)
            newmodelB.update(Xnew = IMSPE_B['par'], Znew = np.array([np.nan]), maxit = 0)
            
            ## Add new design
            IMSPE_C = IMSPE_search(model = newmodelB, control = control, Xcand = Xcand, Wijs = Wijs, ncores = ncores)
            path_C  = [IMSPE_C]

            if i < h:
                newmodelC = copy(newmodelB)
                
                if not any(duplicated(np.vstack([model.X0, IMSPE_C['par']]))):
                    newWijs = Wij(mu1 = model.X0, mu2 = IMSPE_C['par'], theta = model.theta, type = model.covtype)
                    WijsC = np.hstack([Wijs, newWijs])
                    WijsC = np.vstack((WijsC, 
                                       np.concatenate(
                                           [newWijs, 
                                            Wij(mu1 = IMSPE_C['par'], theta = model.theta, type = model.covtype)
                                            ]).T
                                    )
                            )
                
                else:
                    WijsC = Wijs
                
                for j in range(i,h-1):
                    ## Add remaining replicates
                    newmodelC.update(Xnew = IMSPE_C['par'], Znew = np.array([np.nan]), maxit = 0)
                    IMSPE_C = IMSPE_search(model = newmodelC, replicate = True, control = control, Wijs = WijsC, ncores = ncores)
                    path_C.append(IMSPE_C)
            if IMSPE_C['value'] < IMSPE_A['value']: 
                return dict(par = new_designB, value = IMSPE_C['value'], path = path_B + path_C)

    return dict(par = new_designA, value = IMSPE_A['value'], path = path_A)