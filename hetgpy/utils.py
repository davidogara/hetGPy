# utils.py
# utility functions that are not vectorized (will likely be sped up with numba)
import warnings
import numpy as np
from scipy.linalg.lapack import dtrtri
from hetgpy.covariance_functions import cov_gen
from scipy.stats.qmc import scale

MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)

def fast_tUY2(mult,Y2):
  r'''
  aggregate array by replicates

  Parameters
  ----------
  mult: nd_arraylike
    replicates for each unique Y2
  Y2: nd_arraylike
    array to be summed

  Returns
  -------
  res: summation at each unique design location
  '''
  # to do: speed this up
  res = np.zeros(shape=mult.shape[0])
  idx = 0
  idxtmp = 0
  for i in range(len(Y2)):
      res[idx]+=Y2[i]
      idxtmp+=1
      if idxtmp == mult[idx]:
          idx+=1
          idxtmp = 0
  return res


def rho_AN(xx, X0, theta_g, g, sigma = 1, type = "Gaussian", SiNK_eps = 1e-4, eps = MACHINE_DOUBLE_EPS, mult = None):
  r'''
  Rho function for SiNK prediction, anistropic case
  
  Parameters
  ----------
  covtype: str 
    covariance kernel type, either 'Gaussian' or 'Matern5_2'
  '''
  if len(xx.shape)==1:
    xx = xx.reshape(-1)
  K = sigma * cov_gen(X1 = X0, theta = theta_g, type = type) + np.diag(eps + g/mult, X0.shape[0])
  
  k = sigma * cov_gen(X1 = xx, X2 = X0, theta = theta_g, type = type)
  
  Kinv = dtrtri(np.linalg.cholesky(K).T)[0]
  Kinv = Kinv @ Kinv.T
  return np.amax(SiNK_eps, np.sqrt(np.diag(k @ Kinv @ k.T))/sigma**2)

def crossprod(X,Y):
  r'''
  Alias for `crossprod` in R

  Parameters
  ----------
  X: ndarray_like
  Y: ndarray_like

  Returns
  -------
  X.T @ Y
  '''
  return X.T @ Y
def duplicated(X,fromLast = False):
  r'''
  Function to match `duplicated` in base R

  Examples
  --------
  from rpy2.robjects import r
  r("x <- c(9:20, 1:5, 3:7, 0:8)")
  x = np.array(r("x"))
  
  (duplicated(x) == np.array(r("duplicated(x)"))).all()
  '''
  arr = np.ones(shape=X.shape[0],dtype=bool)
  # if we are working with model.Z, axis is None
  # otherwise if we are working with model.X0, axis is 0
  # why? because Z can have NAs
  # this is a known issue in numpy: https://github.com/numpy/numpy/issues/23286
  axis = 0 if len(X.shape)==2 else None
  if not fromLast:
    _, i = np.unique(X,return_index=True, axis=axis)
    arr[i] = False
  else:
    # flip the array and put it back
    _, i = np.unique(X[::-1],return_index=True, axis = axis)
    arr[i] = False
    arr = arr[::-1]

  return arr



def scale_01(X, l_bounds=None, u_bounds=None):
    '''Scale native units -> [0,1]^d. Wraps scipy.stats.qmc.scale(reverse=True).
    If bounds are omitted, they are inferred from the column-wise min/max of X.'''
    if X.ndim < 2:
        X = X.reshape(-1, 1)
    if l_bounds is None or u_bounds is None:
        l_bounds = X.min(axis=0)
        u_bounds = X.max(axis=0)
        warnings.warn(f'Inferring bounds from data:\nl_bounds: {l_bounds}\nu_bounds: {u_bounds}')
    return scale(X, l_bounds=l_bounds, u_bounds=u_bounds, reverse=True)
def scale_native(X, l_bounds, u_bounds):
    '''Scale [0,1]^d -> native units. Wraps scipy.stats.qmc.scale(reverse=False).'''
    if X.ndim < 2:
        X = X.reshape(-1, 1)
    return scale(X, l_bounds=l_bounds, u_bounds=u_bounds, reverse=False)