# utils.py
# utility functions that are not vectorized (will likely be sped up with numba)

import numpy as np
from scipy.linalg.lapack import dtrtri
from hetgpy.covariance_functions import cov_gen
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)

def fast_tUY2(mult,Y2):
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

# Rho function for SiNK prediction, anistropic case
## @param covtype covariance kernel type, either 'Gaussian' or 'Matern5_2'
def rho_AN(xx, X0, theta_g, g, sigma = 1, type = "Gaussian", SiNK_eps = 1e-4, eps = MACHINE_DOUBLE_EPS, mult = None):
  if len(xx.shape)==1:
    xx = xx.reshape(-1)
  K = sigma * cov_gen(X1 = X0, theta = theta_g, type = type) + np.diag(eps + g/mult, X0.shape[0])
  
  k = sigma * cov_gen(X1 = xx, X2 = X0, theta = theta_g, type = type)
  
  Kinv = dtrtri(np.linalg.cholesky(K).T)[0]
  Kinv = Kinv @ Kinv.T
  return np.amax(SiNK_eps, np.sqrt(np.diag(k @ Kinv @ k.T))/sigma**2)

def crossprod(X,Y):
      return X.T @ Y
def duplicated(X):
      '''
      Function to match `duplicated` in base R

      Examples
      --------
      from rpy2.robjects import r
      r("x <- c(9:20, 1:5, 3:7, 0:8)")
      x = np.array(r("x"))
      
      (duplicated(x) == np.array(r("duplicated(x)"))).all()
      '''
      _, i   = np.unique(X,return_index=True)
      arr    = np.ones_like(X, dtype = bool)
      arr[i] = False
      return arr