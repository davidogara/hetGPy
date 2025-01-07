'''
Implementation of homGP using jax

**Note: currently experimental
'''
import jax
jax.config.update("jax_enable_x64", True)
from hetgpy import homGP
from numpy.typing import ArrayLike, NDArray
import jax.numpy as np
from hetgpy.cov_jax import cov_gen_jax as cov_gen
from jax.scipy.linalg import solve_triangular
import warnings
from time import time
from hetgpy.utils import fast_tUY2
from hetgpy.auto_bounds import auto_bounds
from scipy import optimize
NDArrayInt = NDArray[np.int_]
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)
from jax.flatten_util import ravel_pytree

class homGP_jax(homGP):
    def __init__(self):
        super().__init__()
    def logLikHom(self,X0: ArrayLike, Z0: ArrayLike, Z: ArrayLike, mult: NDArrayInt, theta: ArrayLike, g: float, beta0: float | None = None, covtype: str = "Gaussian", eps: float = MACHINE_DOUBLE_EPS) -> float:
        r'''
        logLikHom but with autodiff
        '''
        n = X0.shape[0]
        N = Z.shape[0]

        C = cov_gen(X1 = X0, theta = theta, type = covtype)
        self.C = C
        Ki = np.linalg.cholesky(C + np.diag(eps + g / mult) ).T
        ldetKi = - 2.0 * np.sum(np.log(np.diag(Ki)))
        Ki = solve_triangular(Ki,np.eye(n)) #  -- equivalent of chol2inv -- see https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
        Ki = Ki @ Ki.T     #  -- equivalent of chol2inv
        
        self.Ki = Ki
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z0 / Ki.sum()
        self.beta0 = beta0

        psi_0 = (Z0 - beta0).T @ Ki @ (Z0 - beta0)
        psi = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g) + psi_0)
        
        # check for divide by zero warnings
        #if psi <=0: return np.nan
        #if g <= 0: return np.nan
        loglik = (-N / 2.0) * np.log(2*np.pi) - (N / 2.0) * np.log(psi) + (1.0 / 2.0) * ldetKi - (N - n)/2.0 * np.log(g) - (1.0 / 2.0) * np.log(mult).sum() - (N / 2.0)
            
        
        return loglik
    
    def mle_jax(self,X: ArrayLike, Z: ArrayLike, lower: ArrayLike | None = None, upper: ArrayLike | None = None, known: dict = dict(),
                        noiseControl: dict = dict(g_bounds = (MACHINE_DOUBLE_EPS, 1e2)),
                        init: dict = {},
                        covtype: str = "Gaussian",
                        maxit: int = 100, eps: float = MACHINE_DOUBLE_EPS, settings: dict = dict(returnKi = True, factr = 1e7)) -> None:
        r'''
        MLE, but auto-diffed
        '''
        known = known.copy()
        init = init.copy()
        if type(X) == dict:
            X0 = X['X0']
            Z0 = X['Z0']
            mult = X['mult']
            if sum(mult) != len(Z):    raise ValueError(f"Length(Z) should be equal to sum(mult): they are {len(Z)} \n and {sum(mult)}")
            if len(X0.shape) == 1:      warnings.warn(f"Coercing X0 to shape {len(X0)} x 1"); X0 = X0.reshape(-1,1)
            if len(Z0) != X0.shape[0]: raise ValueError("Dimension mismatch between Z0 and X0")
        else:
            if len(X.shape) == 1:    warnings.warn(f"Coercing X to shape {len(X)} x 1"); X = X.reshape(-1,1)
            if X.shape[0] != len(Z): raise ValueError("Dimension mismatch between Z and X")
            elem = find_reps(X, Z, return_Zlist = False)
            X0   = elem['X0']
            Z0   = elem['Z0']
            Z    = elem['Z']
            mult = elem['mult']

            
        covtypes = ("Gaussian", "Matern5_2", "Matern3_2")
        covtype = [c for c in covtypes if c==covtype][0]

        if lower is None or upper is None:
            auto_thetas = auto_bounds(X = X0, covtype = covtype)
            if lower is None: lower = auto_thetas['lower']
            if upper is None: upper = auto_thetas['upper']
            if known.get("theta") is None and init.get('theta') is None:  init['theta'] = np.sqrt(upper * lower)
        lower = np.array(lower).reshape(-1)
        upper = np.array(upper).reshape(-1)
        if len(lower) != len(upper): raise ValueError("upper and lower should have the same size")

        tic = time()

        if settings.get('return_Ki') is None: settings['return_Ki'] = True
        if noiseControl.get('g_bounds') is None: noiseControl['g_bounds'] = (MACHINE_DOUBLE_EPS, 1e2)
        
        g_min = noiseControl['g_bounds'][0]
        g_max = noiseControl['g_bounds'][1]

        beta0 = known.get('beta0')

        N = len(Z)
        n = X0.shape[0]

        if len(X0.shape) == 1: raise ValueError("X0 should be a matrix. \n")

        if known.get("theta") is None and init.get("theta") is None: init['theta'] = 0.9 * lower + 0.1 * upper # useful for mleHetGP
        
        
        if known.get('g') is None and init.get('g') is None: 
            if any(mult > 2):
                #t1 = mult.T
                #t2 = (Z.squeeze() - np.repeat(Z0,mult))**2
                init['g'] = np.mean(
                    (
                        (fast_tUY2(mult.T,(Z.squeeze() - np.repeat(Z0,mult))**2)/mult)[np.where(mult > 2)]
                    ))/np.var(Z0,ddof=1) 
            else:
                init['g'] = 0.1
        trendtype = 'OK'
        if beta0 is not None:
            trendtype = 'SK'
        
        ## General definition of fn and gr
        self.max_loglik = float('-inf')
        self.arg_max = np.array([np.nan]).reshape(-1)
        def fn(par, X0, Z0, Z, mult, beta0, theta, g):
            idx = 0 # to store the first non used element of par

            if theta is None: 
                theta = par[0:len(init['theta'])]
                idx   = idx + len(init['theta'])
            if g is None:
                g = par[idx]
            
            loglik = self.logLikHom(X0 = X0, Z0 = Z0, Z = Z, mult = mult, theta = theta, g = g, beta0 = beta0, covtype = covtype, eps = eps)
            
            return -1.0 * loglik # for maximization
        
        ## Both known
        if known.get('g') is not None and known.get("theta") is not None:
            theta_out = known["theta"]
            g_out = known['g']
            out = dict(value = self.logLikHom(X0 = X0, Z0 = Z0, Z = Z, mult = mult, theta = theta_out, g = g_out, beta0 = beta0, covtype = covtype, eps = eps),
                        message = "All hyperparameters given", counts = 0, time = time() - tic)
        else:
            parinit = lowerOpt = upperOpt = []
            if known.get("theta") is None:
                parinit = init['theta']
                lowerOpt = np.array(lower)
                upperOpt = np.array(upper)

            if known.get('g') is None:
                parinit = np.hstack((parinit,init.get('g')))
                lowerOpt = np.append(lowerOpt,g_min)
                upperOpt = np.append(upperOpt,g_max)
            bounds = [(l,u) for l,u in zip(lowerOpt,upperOpt)]
            # convert to numpy for interface with scipy
            # from GPJax.fit
            #x0, scipy_to_jnp = ravel_pytree()
            @jax.jit
            def wrapper(x0,*args):
                value, grads = jax.value_and_grad(fn,argnums=0)(x0,*args)
                return value, grads
            start = time()
            out = optimize.minimize(
                    fun=wrapper, # for maximization
                    args = (X0, Z0, Z, mult, beta0, known.get('theta'), known.get('g')),
                    x0 = parinit,
                    jac=True,
                    method="L-BFGS-B",
                    bounds = bounds,
                    #tol=1e-8,
                    options=dict(maxiter=maxit, #,
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

            g_out = out['par'][-1] if known.get('g') is None else known.get('g')
            theta_out = out['par'][0:len(init['theta'])] if known.get('theta') is None else known['theta']
        
        ki = np.linalg.cholesky(
            cov_gen(X1 = X0, theta = theta_out, type = covtype) + np.diag(eps + g_out / mult)
            ).T
        ki = solve_triangular(ki,np.eye(n))
        Ki = ki @ ki.T
        self.Ki = Ki
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z0 / Ki.sum()
        
        psi_0 = ((Z0 - beta0).T @ Ki) @ (Z0 - beta0)

        nu = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g_out) + psi_0)


        self.theta = theta_out
        self.g = g_out
        self.nu_hat = nu
        self.ll = -1.0 * out['value']
        self.nit_opt = out['counts']
        self.beta0 = beta0
        self.trendtype = trendtype
        self.covtype = covtype 
        self.msg = out['message'] 
        self.eps = eps
        self.X0 = X0
        self.Z0 = Z0 
        self.Z = Z
        self.mult = mult
        self.used_args = dict(lower = lower, upper = upper, known = known, noiseControl = noiseControl)
        self.time = time() - tic
        
        if settings["return_Ki"]: self.Ki  = Ki
        return self
        
        
    
        