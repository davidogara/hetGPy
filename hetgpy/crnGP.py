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
                init = dict(),
                covtype = "Gaussian",
                maxit = 100, 
                eps = MACHINE_DOUBLE_EPS, 
                settings = dict(return_Ki = True, factr = 1e7)):
        # copy on instantiation
        init = init.copy()
        
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
        
        X0 = X[:,0:-1]
        S0 = X[:,-1]

        if T0 is not None and len(T0.shape)==1:
            T0 = T0.reshape(-1,1)
        d = X0.shape[1]
        if np.abs(S0 - S0.astype(int)) > 0:
            raise ValueError(f"Last col of X is assumed to contain integer-valued seed information.")
        S0 = S0.astype(int)

        if lower is None or upper is None:
            auto_thetas = auto_bounds(X = X0, covtype = covtype)
            if lower is not None:
                lower = auto_thetas['lower']
            if upper is not None:
                upper = auto_thetas['upper']
            if init.get('theta') is None or len(init['theta']) < len(lower):
                init['theta'] = np.sqrt(upper * lower)
        if T0 is not None:
            auto_thetasT = auto_bounds(X = T0, covtype = covtype)
            lower = np.concatenate([lower,auto_thetasT['lower']])
            upper = np.concatenate([upper,auto_thetasT['upper']])
            if init.get('theta') is None or len(init['theta']) < len(lower):
                init['theta'] = np.sqrt(upper * lower)
        tic = time()
        if settings.get('return_Ki') is None:
            settings['return_Ki'] = True
        if noiseControl.get('g_bounds') is None:
            noiseControl['g_bounds'] = np.array([MACHINE_DOUBLE_EPS, 1e2])
        if noiseControl.get('rho_bounds') is None:
            noiseControl['rho_bounds'] = np.array([0, 0.9])
        
        g_min, g_max = noiseControl['g_bounds']
        
        beta0 = known.get('beta0')

        n = X0.shape[0]
        
        if known.get('theta') is None and init.get('theta') is None:
            init['theta'] = 0.9*lower + 0.1 * upper
        if known.get('g') is None and init.get('g') is None:
            init['g'] = 0.1
        if known.get('rho') is None and init.get('rho') is None:
            init['rho'] = 0.1
        
        trendtype = 'OK'
        if beta0 is not None:
            trendtype = 'SK'
        
        self.max_loglik = float('inf')
        self.arg_max = np.nan
        def fn(par, X0, S0, T0, Z, beta0, theta, g, rho):
            idx = 0
            if theta is None:
                theta = par[0:len(theta)]
                idx = idx + len(init['theta'])
            if g is None:
                g = par[idx]
                idx = idx + 1
            if rho is None:
                rho = par[idx]
            if T0 is None:
                loglik = self.loglik(X0 = X0, S0 = S0,
                                     Z = Z, theta = theta, g = g,
                                     rho = rho, stype = stype, beta0 = beta0,
                                     covtype = covtype,eps = eps)
            if not np.isnan(loglik):
                if loglik > self.max_loglik:
                    self.max_loglik = loglik
                    self.arg_max = par      
            return loglik
        def gr(par, X0, S0, T0, Z, beta0, theta, g, rho):
            idx = 0
            components = []

            if theta is None:
                theta = par[0:len(theta)]
                idx = idx + len(init['theta'])
                components.append('theta')
            if g is None:
                g = par[idx]
                idx = idx + 1
                components.append('g')
            if rho is None:
                rho = par[idx]
                components.append('rho')
            if T0 is None:
                return self.dloglik(X0 = X0,S0 = S0, Z = Z,
                                    theta = theta, g = g, rho = rho, 
                                    stype = stype, beta0 = beta0, covtype = covtype, 
                                    eps = eps, components = components)
        
        if known.get('g') is not None and known.get('theta') is not None and known.get('rho') is not None:
            theta_out = known['theta']
            g_out = known['g']
            rho_out = known['rho']
            if T0 is None:
                out = dict(
                    value = self.loglik(X0 = X0, S0 = S0,
                                     Z = Z, theta = theta_out, g = g_out,
                                     rho = rho_out, stype = stype, 
                                     beta0 = beta0,
                                     covtype = covtype,eps = eps),
                    message = 'All hyperparameters given',
                    counts = 0,
                    time = time() - tic
                )
        else:
            parinit, lowerOpt, upperOpt = np.array([]), np.array([]), np.array([])
            if known.get('theta') is None:
                parinit = init['theta']
                lowerOpt = np.concatenate([lowerOpt, lower])
                upperOpt = np.concatenate([upperOpt, upper])
            if known.get('g') is None:
                parinit = np.concatenate(parinit, init['g'])
                lowerOpt = np.concatenate([lowerOpt, g_min])
                upperOpt = np.concatenate([upperOpt, g_max])
            if known.get('rho') is None:
                parinit = np.concatenate(parinit, init['rho'])
                lowerOpt = np.concatenate([lowerOpt, noiseControl['rho_bounds'][[0]]])
                upperOpt = np.concatenate([upperOpt, noiseControl['rho_bounds'][[1]]])
            
            bounds = [(l,u) for l,u in zip(lowerOpt,upperOpt)]
            
            out = optimize.minimize(
                    fun=fn, # for maximization
                    args = (X0, S0, T0, Z, beta0, known.get('theta'), known.get('g'), known.get('rho')),
                    x0 = parinit,
                    jac=gr,
                    method="L-BFGS-B",
                    bounds = bounds,
                    options=dict(maxiter=maxit,
                                ftol = settings.get('factr',10) * np.finfo(float).eps,#,
                                gtol = settings.get('pgtol',0) # should map to pgtol
                                )
            )
            python_kws_2_R_kws = {
                'x':'par',
                'fun': 'value',
                'nit': 'counts'
            }
            out['counts'] = dict(nfev=out['nfev'],njev=out['njev'])
            for key, val in python_kws_2_R_kws.items():
                out[val] = out[key]
            if out.success == False:
                out = dict(par = self.arg_max, value = -1.0 * self.max_loglik, counts = np.nan,
                message = "Optimization stopped due to NAs, use best value so far")
            
            idx = 0
            theta_out = out['par'][0:len(init['theta'])] if known.get('theta') is None else known['theta']
            idx = idx + len(init['theta'])
            g_out = out['par'][idx] if known.get('g') is None else known.get('g')
            
            rho_out = out['par'][-1] if known.get('rho') is None else known.get('rho')

            

            Cx = cov_gen(X1 = X0, theta = theta_out, type = covtype)
            if self.ids is None:
                # mimics R's outer(S0,S0,'==')
                self.ids = S0 == S0[:,None]
            Cs = np.full(shape=(n,n),fill_value=rho_out,dtype=float)
            Cs[self.ids] = 1.0
            C = Cx * Cs
            
            self.C = C
            self.Cx = Cx
            self.Cs = Cs
            
            jitter = (eps+g_out)*np.eye(n)
            Ki = np.linalg.cholesky(C + jitter).T
            # to mirror R's chol2inv: do the following:
            # expose dtrtri from lapack (for fast cholesky inversion of a triangular matrix)
            # use result to compute Ki (should match chol2inv)
            Ki = dtrtri(Ki)[0] #  -- equivalent of chol2inv -- see https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
            Ki = Ki @ Ki.T     #  -- equivalent of chol2inv
            self.Ki = Ki

            self.nu = (Z - beta0).T @ self.Ki
            return