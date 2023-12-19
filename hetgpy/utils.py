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