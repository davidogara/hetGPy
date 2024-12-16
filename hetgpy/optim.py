from __future__ import annotations
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import hetgpy
from hetgpy import hetGP, homGP
from hetgpy.contour import crit_cSUR, crit_ICU, crit_MCU, crit_MEE, crit_tMSE
from hetgpy.covariance_functions import cov_gen, partial_cov_gen, euclidean_dist
from hetgpy.qEI import qEI_cpp
  
from scipy.stats import t, norm
from scipy.special import gamma, erfc, erfcx
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from hetgpy.IMSE import maximinSA_LHS
from hetgpy.utils import duplicated

def crit_EI(x, model, cst = None, preds = None):
  r'''
  Expected Improvement (EI) criteria

  Parameters
  ----------
  x: nd_arraylike
    model designs, one point per row
  model: hetgpy.hetGP
    hetGP or homGP model
  cst: float
    optional plugin value of the mean
  preds: Dict
    model predictions (optional)

  References
  ----------

  Mockus, J.; Tiesis, V. & Zilinskas, A. (1978). The application of Bayesian methods for seeking the extremum Towards Global Optimization, Amsterdam: Elsevier, 2, 2.
  Vazquez E, Villemonteix J, Sidorkiewicz M, Walter E (2008). Global Optimization Based on Noisy Evaluations: An Empirical Study of Two Statistical Approaches, Journal of Physics: Conference Series, 135, IOP Publishing.
  
  
  
  Examples
  --------
  >>> from hetgpy.test_functions import f1d
  >>> from hetgpy.homGP import homGP
  >>> from hetgpy.optim import crit_EI
  >>> import numpy as np
  >>> ftest = f1d
  >>> n_init = 5 # number of unique designs
  >>> X = np.linspace(0, 1, n_init).reshape(-1,1)
  >>> Z = ftest(X)
  >>> xgrid = np.linspace(0,1,51).reshape(-1,1)
  >>> model = homGP()
  >>> model.mle(X = X, Z = Z, lower = np.array([0.01]), upper = np.array([1]), known = dict(g = 2e-8))
  >>> EI = crit_EI(xgrid, model, cst = model.Z0.min())
  '''
  if cst is None: cst = np.min(model.predict(x = model['X0'])['mean'])
  if len(x.shape) == 1: x = x.reshape(-1,model.X0.shape[1])
  if preds is None: preds = model.predict(x = x)
  
  if type(model)== hetgpy.homTP or type(model)== hetgpy.hetTP:
    gamma = (cst - preds['mean'])/np.sqrt(preds['sd2'])
    res = (cst - preds['mean']) * t.cdf(gamma, df = model['nu'] + len(model['Z']))
    res = res + np.sqrt(preds['sd2']) * (1 + (gamma**2 - 1)/(model['nu'] + len(model['Z']) - 1)) * t.pdf(x = gamma, df = model['nu'] + len(model['Z']))
    res[np.where(res < 1e-12)] = 0 # for stability
    return(res)
  
  xcr = (cst - preds['mean'])/np.sqrt(preds['sd2'])
  res = (cst - preds['mean']) * norm.cdf(xcr)
  res = res + np.sqrt(preds['sd2']) * norm.pdf(xcr)
  res[(preds['sd2'] < np.sqrt(np.finfo(float).eps)) | (res < 1e-12)] = 0 # for stability
  return res

def deriv_crit_EI(x, model, cst = None, preds = None):
  r'''
  Derivative for crit_EI, used in crit_optim
  
  Parameters
  ----------
  x: nd_arraylike
    model designs, one point per row
  model: hetgpy.hetGP
    hetGP or homGP model
  cst: float
    optional plugin value of the mean
  preds: Dict
    model predictions (optional)

  References
  ----------
  Ginsbourger, D. Multiples metamodeles pour l'approximation et l'optimisation de fonctions numeriques multivariables Ecole Nationale Superieure des Mines de Saint-Etienne, Ecole Nationale Superieure des Mines de Saint-Etienne, 2009
  Roustant, O., Ginsbourger, D., DiceKriging, DiceOptim: Two R packages for the analysis of computer experiments by kriging-based metamodeling and optimization, Journal of Statistical Software, 2012


  '''
  if cst is None: cst = np.min(model.predict(x = model.X0)['mean'])
  if len(x.shape) == 1: x = x.reshape(-1,model.X0.shape[1])
  if preds is None: preds = model.predict(x = x)
  
  pred_gr = predict_gr(model, x)
  
  z = (cst - preds['mean'])/np.sqrt(preds['sd2'])
  
  if type(model)==homGP or type(model)==hetGP:
    res = pred_gr['sd2'] / (2 * np.sqrt(preds['sd2'])) * norm.pdf(z) - pred_gr['mean'] * norm.cdf(z)
  else:
    # dz = - dm/s - z ds/s = -dm/s - z * ds2/(2s2) 
    dz = -pred_gr['mean']/np.sqrt(preds['sd2']) - z * pred_gr['sd2']/(2 * preds['sd2'].reshape(x.shape[0],x.shape[0]).T)
    
    # d( (cst - m(x)).pt(z(x))) = 
    p1 = -pred_gr['mean'] * t.cdf(z, df = model.nu + len(model.Z)) + (cst - preds['mean']) * dz * t.pdf(z, df = model.nu + len(model.Z))
    
    a = model.nu + len(model.Z) - 1
    # d( s(x) (1 + (z^2-1)/(nu + N -1)) dt(z(x)) (in 2 lines)
    p2 = (pred_gr['sd2']/(2*np.sqrt(preds['sd2'])) * (1 + (z**2 - 1)/a) + 2*np.sqrt(preds['sd2']) * z * dz/a) * t.pdf(z, df = model.nu + len(model.Z))
    p2 = p2 + np.sqrt(preds['sd2']) * (1 + (z**2 - 1)/a) * dz * dlambda(z, model.nu + len(model.Z))
    res = p1 + p2
  
  
  res[np.abs(res) < 1e-12] = 0 # for stability with optim
  
  return res

def predict_gr(model, x):
  r'''
  Gradient of the prediction given a model
  '''
  if len(x.shape) == 1: x = x.reshape(-1,1)
  kvec =  cov_gen(X1 = model.X0, X2 = x, theta = model.theta, type = model.covtype)
  
  dm = np.full(fill_value=np.nan, shape = (x.shape[0], x.shape[1]))
  ds2 = dm.copy()
  
  for i in range(x.shape[0]):
    dkvec = np.full(fill_value=np.nan, shape=(model.X0.shape[0], x.shape[1]))
    for j in range(x.shape[1]):
      dkvec[:, j] = np.squeeze(partial_cov_gen(X1 = x[i:i+1,:], X2 = model.X0, theta = model.theta, i1 = 1, i2 = j + 1, arg = "X_i_j", type = model.covtype)) * kvec[:,i]
    
    dm[i,:] = ((model.Z0 - model.beta0).T @ model.Ki) @ dkvec
    if (type(model)==homGP or type(model)==hetGP) and model.trendtype == "OK": 
      tmp = np.squeeze(1 - (model.Ki.sum(axis=1)) @ kvec[:,i])/(np.sum(model.Ki)) * (model.Ki.sum(axis=1)) @ dkvec 
    else: 
      tmp = 0
    ds2[i,:] = -2 * ((kvec[:,i].T @ model.Ki) @ dkvec + tmp)
  
  if type(model)==homGP or type(model)==hetGP:
    return dict(mean = dm, sd2 = model.nu_hat * ds2)
  else:
    return dict(mean = model.sigma2 * dm, sd2 =  (model.nu + model.psi - 2) / (model.nu + len(model.Z) - 2) * model.sigma2**2 * ds2)
  
def dlambda(z,a):
  r'''
  Derivative of the student-t pdf

  Parameters
  ----------
  z: float
    input location
  a: float
    degree of freedom parameter

  Returns
  -------
  derivative of student-t pdf

  Examples
  --------
  >>> dlambda(0.55, 3.6) # -0.2005827
  '''
  return -(a + 1) * gamma((a + 1)/2)/(np.sqrt(np.pi * a) * a * gamma(a/2)) * z * ((a + z**2)/a)**(-(a +3)/2)


def crit_qEI(x, model, cst = None, preds = None):
  r'''
  Fast approximated batch-Expected Improvement criterion (for minimization)
  Parallel Expected improvement

  Parameters
  ----------
  x: nd_array
    matrix of new designs representing the batch of q points, one point per row (q x d)
  model: homGP/hetGP
    model object including inverve matrices
  cst: nd_array
    optional optional plugin value used in the EI, see details
  preds: dict
    optional predictions at x to avoid recomputing if already done (must include the predictive covariance, i.e., the cov slot)
  
  Returns
  -------
  qEI_cpp  

  Details
  ------- 
  cst is classically the observed minimum in the deterministic case. In the noisy case, the min of the predictive mean works fine. This is a beta version at this point. It may work for for TP models as well.

  References
  ---------- 
  M. Binois (2015), Uncertainty quantification on Pareto fronts and high-dimensional strategies in Bayesian optimization, with applications in multi-objective automotive design.
  Ecole Nationale Superieure des Mines de Saint-Etienne, PhD thesis.
  
  Examples
  --------
  >>> from hetgpy.test_functions import f1d
  >>> from hetgpy.homGP import homGP
  >>> from hetgpy.optim import crit_qEI
  >>> import numpy as np
  >>> ftest = f1d
  >>> n_init = 5 # number of unique designs
  >>> X = np.linspace(0, 1, n_init).reshape(-1,1)
  >>> Z = ftest(X)
  >>> xgrid = np.linspace(0,1,51).reshape(-1,1)
  >>> model = homGP()
  >>> model.mleHomGP(X = X, Z = Z, lower = np.array([0.01]), upper = np.array([1]), known = dict(g = 2e-8))
  >>> xbatch = np.array((0.37, 0.17, 0.7)).reshape(3, 1)
  >>> fqEI = crit_qEI(xbatch, model, cst = model.Z0.min())
  '''
  if cst is None: cst = np.min(model.predict(x = model.X0)['mean'])
  if len(x.shape)==1: x = x.reshape(-1,1)
  if preds is None or preds.get('cov') is None: preds = model.predict(x = x, xprime = x)
  
  
  # cov2cor not available in python, so compute it ourselves
  cov = preds['cov']
  Dinv = np.diag(1.0 / np.sqrt(np.diag(cov))) # inverse of diagonal matrix
  cormat = Dinv @ cov @ Dinv
  res = qEI_cpp(mu = -preds['mean'], s = np.sqrt(preds['sd2']), cor = cormat, threshold = -cst)
  
  return res


def crit_search(model, crit, replicate = False, Xcand = None, 
                        control = dict(tol_dist = 1e-6, tol_diff = 1e-6, multi_start = 20,
                                       maxit = 100, maximin = True, Xstart = None), 
                        seed = None,
                        ncores = 1,
                        **kwargs):
    crit_to_function = {
       'crit_EI': crit_EI,
       'crit_cSUR': crit_cSUR, 
       'crit_ICU':crit_ICU, 
       'crit_MCU':crit_MCU, 
       'crit_MEE':crit_MEE, 
       'crit':crit_tMSE
    }
    
    crit_func = crit_to_function[crit]
    def fn(*args):
       # maximize
       return -1.0 *  crit_func(*args)
    if(replicate):
        ## Discrete optimization
        res = []
        for i in range(model.X0.shape[0]):
            res.append(
                crit_func(x = model.X0[i:i+1,:], model = model, **kwargs)
            )
        return dict(
            par = model.X0[np.argmax(res),:].reshape(1,-1), 
            value = np.max(res), 
            new = False, 
            id = np.argmax(res)
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
  
    if crit == "crit_EI": 
      def grad(*args):
         # maximization
         return -1.0 * deriv_crit_EI(*args)
      gr = grad 
    else: 
       gr = None
    
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
        #model = deepcopy(model)    
        res = dict(par = np.array([np.nan]), value = np.inf, new = np.nan)
        crit_to_args = {
           'crit_EI':   {'model':model,'cst': None,'preds':kwargs.get('preds')},
           'crit_MEE':  {'model':model,'thres':kwargs.get('thres',0),'preds':kwargs.get('preds')},
           'crit_cSUR': {'model':model,'thres':kwargs.get('thres',0),'preds':kwargs.get('preds')},
           'crit_ICU':  {'model':model,'thres':kwargs.get('thres',0), 
                         'Xref':kwargs.get('Xref',None),
                        'w':kwargs.get('w',None), 'preds':kwargs.get('preds',None), 
                        'kxprime':kwargs.get('kxprime',None)},
          'crit_tMSE':  {'model':model,'thres':kwargs.get('thres',0),'preds':kwargs.get('preds'),
                         'seps':kwargs.get('seps',0.05)},
          'crit_MCU':   {'model':model,'thres':kwargs.get('thres',0),
                         'gamma':kwargs.get('gamma',2), 'preds':kwargs.get('preds',None)}
        }
        def local_opt_fun(i):
            out = minimize(
            x0  = Xstart[i,:],
            fun = fn,
            jac = gr,
            args = tuple(crit_to_args[crit].values()),
            method = "L-BFGS-B", 
            bounds = [(0,1) for _ in range(d)],
            options=dict(maxiter=control['maxit'], #,
                                ftol = control.get('factr',10e9) * np.finfo(float).eps,#,
                                gtol = control.get('pgtol',0) # should map to pgtol
                                )
            )
            python_kws_2_R_kws = {
                'x':'par',
                'fun': 'value',
                'nfev': 'counts'
            }
            for key, val in python_kws_2_R_kws.items():
                out[val] = out[key]
            return out
        all_res = Parallel(n_jobs=ncores)(delayed(local_opt_fun)(i) for i in range(Xstart.shape[0]))
        all_res_values = [res['value'] for res in all_res]
        res_min = np.nanargmin(all_res_values)
        par = all_res[res_min]['x']
        par[par<0] = 0
        par[par>1] = 1
        res = dict(par = par.reshape(1,-1), 
                    value = crit_func(par,**crit_to_args[crit]), 
                    new = True, id = None)
        
        
        if control['tol_dist'] > 0 and control['tol_diff'] > 0:
            ## Check if new design is not to close to existing design
            dists = np.sqrt(euclidean_dist(res['par'].reshape(1,-1), model.X0))
            if np.min(dists) < control['tol_dist']:
                argmin = np.argmin(dists)
                res = dict(par = model.X0[[argmin],:],
                            value = crit_func(x = model.X0[argmin,:], **crit_to_args[crit]),
                            new = False, id = argmin)
            else:
                ## Check if crit difference between replication and new design is significative
                id_closest = np.argmin(dists) # closest point to new design
                crit_rep = crit_func(x=model.X0[id_closest],**crit_to_args[crit])
                if (res['value']-crit_rep)/res['value'] < control['tol_diff']:
                    res = dict(par = model.X0[[id_closest],:],
                            value = crit_rep,
                            new = False, id = id_closest)
    else:
      ## Discrete
      print('here')
      res = -1 * crit_func(model=model,x=Xcand)
      tmp = (duplicated(np.vstack([model.X0, Xcand[np.argmin(res),:]]), fromLast = True))
      if len(tmp) > 0: 
          par = Xcand[res.argmin(keepdims=True),:]
          return(dict(par = par, value = crit_func(model=model,x=par), new = False, id = tmp))
      par = Xcand[res.argmin(keepdims=True),:]
      return dict(par = par, value = crit_func(model=model,x=par), new = True, id = None)
    return res

def crit_optim(model, crit, h = 2, Xcand = None, 
               control = dict(multi_start = 10, maxit = 100), seed = None, ncores = 1, **kwargs):
  '''
  crit_optim
  '''
  d = model.X0.shape[1]
  if crit == "crit_IMSPE": raise ValueError("crit_IMSPE is intended to be optimized by IMSPE_optim")
  ## A) Setting to beat: first new point then replicate h times
  crit_A = crit_search(model = model, crit = crit, control = control, Xcand = Xcand, seed = seed, ncores = ncores, **kwargs)
  new_designA = crit_A['par'] ## store first considered design to be added
  path_A = [crit_A]

  if h > 0:
    newmodelA = deepcopy(model)
  for i in range(1,h+1):
    ZnewA = newmodelA.predict(crit_A['par'])['mean']
    newmodelA.update(Xnew = crit_A['par'], Znew = ZnewA, maxit = 0)
    crit_A = crit_search(model = newmodelA, crit = crit, replicate = True, 
                          control = control, seed = seed, ncores = ncores, **kwargs)
    path_A.append(crit_A)
  if h == -1: return dict(par = new_designA, value = crit_A['value'], path = path_A) 

  newmodelB = deepcopy(model)
  if h == 0:
    crit_B = crit_search(model = newmodelB, crit = crit, replicate = True, control = control, ncores = ncores,**kwargs)
    new_designB = crit_B['par'] ## store considered design to be added
    
    # search from best replicate
    if Xcand is None:
       crit_C = crit_search(model = newmodelB,
                            crit = crit,
                            control = dict(Xstart = crit_B['par'], 
                                           maxit = control['maxit'],
                                        tol_dist = control['tol_dist'], 
                                        tol_diff = control['tol_diff']), 
                                        ncores = ncores,**kwargs)
    else:
      crit_C = crit_B.copy()
      
      if crit_C['value'] > max(crit_A['value'], crit_B['value']): 
          return dict(par = crit_C['par'], value = crit_C['value'], path = [crit_C])
      
      if crit_B['value'] > crit_A['value']:
          return dict(par = crit_B['par'], value = crit_B['value'], path = [crit_B])
  else:
      for i in range(h):
        ## Add new replicate
        crit_B = crit_search(model = newmodelB, crit = crit, replicate = True, control = control, ncores = ncores,**kwargs)
  
        if i == 0:
            new_designB = crit_B['par'].reshape(1,-1) ##store first considered design to add
            path_B = list()
  
        path_B.append(crit_B)
        ZnewB = newmodelB.predict(crit_B['par'])['mean']
        newmodelB.update(Xnew = crit_B['par'], Znew = ZnewB, maxit = 0)
        
        ## Add new design
        crit_C = crit_search(model = newmodelB, crit = crit, control = control, Xcand = Xcand, ncores = ncores,**kwargs)
        path_C  = [crit_C]

        if i < h:
          newmodelC = deepcopy(newmodelB)
          for j in range(i,h):
            ## Add remaining replicates
            ZnewC = newmodelC.predict(crit_C['par'])['mean']
            newmodelC.update(Xnew = crit_C['par'], Znew = ZnewC, maxit = 0)
            crit_C = crit_search(model = newmodelC, crit = crit, replicate = True, control = control, 
                                 ncores = ncores,**kwargs)
            path_C.append(crit_C)
        if crit_C['value'] < crit_A['value']: return dict(par = new_designB, value = crit_C['value'], path = path_B + path_C)

  return dict(par = new_designA, value = crit_A['value'], path = path_A)


def log1mexp(x):
    r"""

    References
    ----------
    Maechler, Martin (2012). Accurately Computing log(1-exp(-|a|)). Assessed from the Rmpfr package.

    """
    if np.isnan(x):
        print("x is NA")
        return np.nan

    if np.min(x) < 0:
        print("x < 0")
        return np.nan

    if x <= np.log(2):
        return np.log(-np.expm1(-x))
    else:
        return np.log1p(-np.exp(-x))


#' log_h function from
#' @references Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for bayesian optimization. Advances in Neural Information Processing Systems, 36.
def log_h(z, eps=np.finfo(float).eps):
    r"""
    References
    ----------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for Bayesian optimization. Advances in Neural Information Processing Systems, 36.

    """
    c1 = np.log(2 * np.pi) / 2
    c2 = np.log(np.pi / 2) / 2
    if z > -1:
        return np.log(norm.pdf(z) + z * norm.cdf(z))
    if z < -1 / np.sqrt(eps):
        return -z ** 2 / 2 - c1 - 2 * np.log(np.abs(z))
    res = -z ** 2 / 2 - c1 + log1mexp(-(np.log(erfcx(-z / np.sqrt(2)) * np.abs(z)) + c2))
    if np.isnan(res):
        res = -750
    return res


def crit_logEI(x, model, cst = None, preds = None):
    r"""
    Logarithm of Expected Improvement (EI) criteria

    Parameters
    ----------
    x: nd_arraylike
      model designs, one point per row
    model: hetgpy.hetGP
      hetGP or homGP model
    cst: float
      optional plugin value of the mean
    preds: Dict
      model predictions (optional)

    References
    ----------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for Bayesian optimization. Advances in Neural Information Processing Systems, 36.

    Examples
    --------
    >>> from hetgpy.test_functions import f1d
    >>> from hetgpy.homGP import homGP
    >>> from hetgpy.optim import crit_logEI
    >>> import numpy as np
    >>> ftest = f1d
    >>> n_init = 5 # number of unique designs
    >>> X = np.linspace(0, 1, n_init).reshape(-1,1)
    >>> Z = ftest(X)
    >>> xgrid = np.linspace(0,1,51).reshape(-1,1)
    >>> model = homGP()
    >>> model.mle(X = X, Z = Z, lower = np.array([0.01]), upper = np.array([1]), known = dict(g = 2e-8))
    >>> logEI = crit_logEI(xgrid, model, cst = model.Z0.min())
    """
    if cst is None:
        cst = np.min(model.predict(x=model["X0"])["mean"])
    if len(x.shape) == 1:
        x = x.reshape(-1, model.X0.shape[1])
    if preds is None:
        preds = model.predict(x=x)

    if type(model) == hetgpy.homTP or type(model) == hetgpy.hetTP:
        gamma = (cst - preds["mean"]) / np.sqrt(preds["sd2"])
        res = (cst - preds["mean"]) * t.cdf(gamma, df=model["nu"] + len(model["Z"]))
        res = res + np.sqrt(preds["sd2"]) * (1 + (gamma**2 - 1) / (model["nu"] + len(model["Z"]) - 1)) * t.pdf(x=gamma, df=model["nu"] + len(model["Z"]))
        res[np.where(res < 1e-12)] = 0  # for stability
        return np.log(res)

    xcr = (cst - preds["mean"]) / np.sqrt(preds["sd2"])
    res = np.log(np.sqrt(preds["sd2"]))
    for i in np.arange(x.shape[0]):
        res[i] = np.maximum(-800, res[i] + log_h(xcr[i]))
    res[preds["sd2"] == 0] = -800

    return res

#' Derivative  of log_h function from
#' @references Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for bayesian optimization. Advances in Neural Information Processing Systems, 36.
def dlog_h(z, eps=np.finfo(float).eps):
    r"""
    References
    ----------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for Bayesian optimization. Advances in Neural Information Processing Systems, 36.

    """
    c2 = np.log(np.pi / 2) / 2
    if z > -1:
        return norm.cdf(z) / (norm.pdf(z) + z * norm.cdf(z))
    if z < -1 / np.sqrt(eps):
        return -z - 2 / z
    # res = -z ** 2 / 2 - c1 + log1mexp(-(np.log(erfcx(-z / np.sqrt(2)) * np.abs(z)) + c2))
    # res = -z + (np.exp(c2) * (np.sqrt(np.pi) * (z**2 + 1) * np.exp(z**2 / 2) * (erfc(z / np.sqrt(2)) - 2) - np.sqrt(2) * z)) / (np.sqrt(np.pi) * (z * np.exp((z**2 + 2 * c2) / 2) * (erfc(z / np.sqrt(2)) - 2) - 1))
    res = -z + (np.exp(c2) * (-np.sqrt(np.pi) * (z**2 + 1) * erfcx(-z/np.sqrt(2)) - np.sqrt(2) * z)) / (np.sqrt(np.pi) * (z * np.exp(c2) * -erfcx(-z / np.sqrt(2)) - 1))
    if np.isnan(res):
        res = 0
    return res


def deriv_crit_logEI(x, model, cst=None, preds=None):
    r"""
    Derivative of the logarithm of Expected Improvement (EI) criteria

    Parameters
    ----------
    x: nd_arraylike
      model designs, one point per row
    model: hetgpy.hetGP
      hetGP or homGP model
    cst: float
      optional plugin value of the mean
    preds: Dict
      model predictions (optional)

    References
    ----------
    Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2024). Unexpected improvements to expected improvement for Bayesian optimization. Advances in Neural Information Processing Systems, 36.

    Examples
    --------
    """
    if cst is None:
        cst = np.min(model.predict(x=model["X0"])["mean"])
    if len(x.shape) == 1:
        x = x.reshape(-1, model.X0.shape[1])
    if preds is None:
        preds = model.predict(x=x)

    pred_gr = predict_gr(model, x)
    z = (cst - preds["mean"]) / np.sqrt(preds["sd2"])

    if type(model) == homGP or type(model) == hetGP:
        ds = pred_gr["sd2"] / (2 * np.sqrt(preds["sd2"]).reshape(x.shape[0], x.shape[0]).T)
        dz = -pred_gr["mean"] / np.sqrt(preds["sd2"]) - z * ds / np.sqrt(preds["sd2"])
        return dz * dlog_h(z) + ds / np.sqrt(preds["sd2"])
    else:
        # dz = - dm/s - z ds/s = -dm/s - z * ds2/(2s2)
        dz = -pred_gr["mean"] / np.sqrt(preds["sd2"]) - z * pred_gr["sd2"] / (2 * preds["sd2"].reshape(x.shape[0], x.shape[0]).T)

        # d( (cst - m(x)).pt(z(x))) =
        p1 = -pred_gr["mean"] * t.cdf(z, df=model.nu + len(model.Z)) + (cst - preds["mean"]) * dz * t.pdf(z, df=model.nu + len(model.Z))

        a = model.nu + len(model.Z) - 1
        # d( s(x) (1 + (z^2-1)/(nu + N -1)) dt(z(x)) (in 2 lines)
        p2 = (pred_gr["sd2"] / (2 * np.sqrt(preds["sd2"])) * (1 + (z**2 - 1) / a) + 2 * np.sqrt(preds["sd2"]) * z * dz / a) * t.pdf(z, df=model.nu + len(model.Z))
        p2 = p2 + np.sqrt(preds["sd2"]) * (1 + (z**2 - 1) / a) * dz * dlambda(z, model.nu + len(model.Z))
        res = p1 + p2
        res[np.abs(res) < 1e-12] = 0  # for stability with optim

        eitmp = (cst - preds["mean"]) * t.cdf(z, df=model["nu"] + len(model["Z"]))
        eitmp = eitmp + np.sqrt(preds["sd2"]) * (1 + (z**2 - 1) / (model["nu"] + len(model["Z"]) - 1)) * t.pdf(x=z, df=model["nu"] + len(model["Z"]))
        res = res/eitmp
    return res
