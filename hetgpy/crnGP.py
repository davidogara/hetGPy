'''
command random number GP
'''

from __future__ import annotations
import numpy as np
from itertools import chain
import warnings
from time import time
from scipy.linalg.lapack import dtrtri
from scipy import optimize
from scipy.stats import norm
from hetgpy.covariance_functions import cov_gen, partial_cov_gen, euclidean_dist
from hetgpy.utils import fast_tUY2, rho_AN
from hetgpy.find_reps import find_reps
from hetgpy.auto_bounds import auto_bounds
from hetgpy.find_reps import find_reps
from hetgpy.utils import duplicated
from hetgpy.update_covar import update_Ki, update_Ki_rep
from hetgpy.plot import plot_diagnostics, plot_optimization_iterates
from copy import deepcopy
import contextlib
from numpy.typing import ArrayLike, NDArray
NDArrayInt = NDArray[np.int_]
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)

class crnGP():
    def __init__(self):
        self.mle = self.mlecrnGP
        return
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self,item,value):
        self.__dict__[item] = value
    def get(self,key):
        r'''
        General `get` item (retrives key from self.__dict__)
        
        '''
        return self.__dict__.get(key)
    
    def loglik(self,X0, S0, Z, theta, g, rho, stype, beta0 = None, covtype = "Gaussian", eps = MACHINE_DOUBLE_EPS):
        n = X0.shape[0]
        d = X0.shape[1]

        Cx = cov_gen(X1 = X0, theta = theta, type = covtype)
        if self.ids is None:
            # mimics R's outer(S0,S0,'==')
            self.ids = S0 == S0[:,None]
        Cs = np.full(shape=(n,n),fill_value=rho,dtype=float)
        Cs[self.ids] = 1.0
        C = Cx * Cs
        jitter = (eps+g)*np.eye(n)
        Ki = np.linalg.cholesky(C + jitter).T
        ldetKi = - 2.0 * np.sum(np.log(np.diag(Ki)))
        # to mirror R's chol2inv: do the following:
        # expose dtrtri from lapack (for fast cholesky inversion of a triangular matrix)
        # use result to compute Ki (should match chol2inv)
        Ki = dtrtri(Ki)[0] #  -- equivalent of chol2inv -- see https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
        Ki = Ki @ Ki.T     #  -- equivalent of chol2inv
        
        self.Ki = Ki
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z0 / Ki.sum()
        self.beta0 = beta0
        return
    
    def dloglik(self):
        return
    
    def mlecrnGP(self):
        return
    