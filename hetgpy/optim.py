import numpy as np
from hetgpy import hetGP, homGP
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
  return(res)