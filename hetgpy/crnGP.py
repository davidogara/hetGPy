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
        
        self.C = C
        self.Cx = Cx
        self.Cs = Cs
        
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
            beta0 = Ki.sum(axis=1) @ Z / Ki.sum()
        self.beta0 = beta0
        
        psi = (((Z - beta0).T @ Ki) @ (Z - beta0)) / n
        loglik = -0.5 * n * np.log(np.pi) - 0.5 * n + np.log(psi) + 0.5 * ldetKi - 0.5 * n
        return loglik
    
    def dloglik(self,X0, S0, Z, theta, g, rho, stype, beta0 = None, covtype = "Gaussian",
                          eps = MACHINE_DOUBLE_EPS, components = ("theta", "g", "rho")):
        n, d = X0.shape
        C = self.C
        Cx = self.Cx
        Cs = self.Cs
        Ki = self.Ki

        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z / Ki.sum()

        Z = (Z - beta0).copy()
        KiZ = Ki @ Z

        psi = (Z @ KiZ).squeeze() / n

        tmp1 = np.array([])
        tmp2 = np.array([])
        tmp3 = np.array([])

        if 'theta' in components:
            tmp1 = np.full(shape=len(theta),fill_value=np.nan)
            for i in range(len(theta)):
                dC_dthetak = partial_cov_gen(X1=X0,theta=theta[i],type=covtype,arg="theta_k") * C
                tmp1[i] = 0.5 * (KiZ.T @ dC_dthetak) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_dthetak)
        if 'g' in components:
            tmp2 = 0.5 * np.sum(KiZ**2) / psi - 0.5 * np.sum(np.diag(Ki))
            tmp2 = np.array([tmp2])
        if 'rho' in components:
            dC_drho = np.ones(shape=(n,n))
            ids = self.ids
            dC_drho[ids] = 0
            dC_drho *= Cx
            tmp3 = 0.5 * (KiZ.T @ dC_drho) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_drho)
        return np.concatenate([tmp1,tmp2,tmp3]).squeeze()
    
    def mlecrnGP(self,X, Z, T0 = None, 
                stype = "none", 
                lower = None, upper = None, 
                known = None,
                noiseControl = dict(g_bounds = (10*MACHINE_DOUBLE_EPS, 1e2),
                                    rho_bounds = (0.001, 0.9)),
                init = None,
                covtype = "Gaussian",
                maxit = 100, 
                eps = MACHINE_DOUBLE_EPS, 
                settings = dict(return_Ki = True, factr = 1e7)):
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        if T0 is None and X.shape[0] != Z.shape[0]:
            raise ValueError(f"Dimension mismatch between Z and X: {Z.shape=}, {X.shape=}")
        if T0 is not None and X.shape[0] != Z.shape[0]:
            raise ValueError(f"Dimension mismatch between Z and X: {Z.shape=}, {X.shape=}")
        
        stypes = ("none", "XS")
        if stype not in stypes:
            raise ValueError(f"stype must be one of {stypes}")

        covtypes = ("Gaussian", "Matern5_2", "Matern3_2")
        if covtype not in covtypes:
            raise ValueError(f"covtype must be one of {covtypes}")
        
        return
    