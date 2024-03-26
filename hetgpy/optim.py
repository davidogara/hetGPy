import numpy as np
from hetgpy import hetGP, homGP
from hetgpy.covariance_functions import cov_gen, partial_cov_gen
from scipy.stats import t, norm
def crit_EI(x, model, cst = None, preds = None):
  if cst is None: cst = np.min(model.predict(x = model['X0'])['mean'])
  if len(x.shape) == 1: x = x.reshape(-1,1)
  if preds is None: preds = model.predict(x = x)
  
  if type(model)== homGP.homTP or type(model)== hetGP.hetTP:
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
  if cst is None: cst = np.min(model.predict(x = model.X0)['mean'])
  if len(x.shape) == 1: x = x.reshape(-1,1)
  if preds is None: preds = model.predict(x = x)
  
  pred_gr = predict_gr(model, x)
  
  z = (cst - preds['mean'])/np.sqrt(preds['sd2'])
  
  if type(model)==homGP.homGP or type(model)==hetGP.hetGP:
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
  if len(x.shape) == 1: x = x.reshape(-1,1)
  kvec =  cov_gen(X1 = model.X0, X2 = x, theta = model.theta, type = model.covtype)
  
  dm = np.full(fill_value=np.nan, shape = (x.shape[0], x.shape[0]))
  ds2 = dm.copy()
  
  for i in range(x.shape[0]):
    dkvec = np.full(fill_value=np.nan, shape=(model.X0.shape[0], x.shape[1]))
    for j in range(x.shape[1]):
      dkvec[:, j] = np.squeeze(partial_cov_gen(X1 = x[i:i+1,:], X2 = model.X0, theta = model.theta, i1 = 1, i2 = j, arg = "X_i_j", type = model.covtype)) * kvec[:,i]
    
    dm[i,:] = (model.Z0 - model.beta0.T @ model.Ki) @ dkvec
    if (type(model)==homGP.homGP or type(model)==hetGP.hetGP) and model.trendtype == "OK": 
      tmp = np.squeeze(1 - (model.Ki.sum(axis=1)) @ kvec[:,i])/(np.sum(model.Ki)) * (model.Ki.sum(axis=1)) @ dkvec 
    else: 
      tmp = 0
    ds2[i,:] <- -2 * ((kvec[:,i].T @ model.Ki) @ dkvec + tmp)
  
  if type(model)==homGP.homGP or type(model)==hetGP.hetGP:
    return dict(mean = dm, sd2 = model.nu_hat * ds2)
  else:
    return dict(mean = model.sigma2 * dm, sd2 =  (model.nu + model.psi - 2) / (model.nu + len(model.Z) - 2) * model.sigma2**2 * ds2)