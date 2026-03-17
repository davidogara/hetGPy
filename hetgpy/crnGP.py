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
from itertools import combinations, product
import math
from numpy.typing import ArrayLike, NDArray
NDArrayInt = NDArray[np.int_]
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)

class crnGP():
    def __init__(self):
        self.mle = self.mlecrnGP
        self.ids = None
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
    def pairwise_rho(self,S0r, S0c = None,rho = None):
        rho = np.atleast_1d(rho)

        if S0c is None:
            S0c = S0r.copy()
        nr, nc = S0r.shape[0], S0c.shape[0]
        rho_indices = np.array(list(combinations(np.unique(S0r),2)))

        # iterate over rho indices and make pointer to rho and its reverse
        rho_mapper = {}
        for i, idx in enumerate(rho_indices):
            s = tuple(sorted(idx))
            s_rev = tuple(reversed(s))
            rho_mapper[s] = rho[i]
            rho_mapper[s_rev] = rho[i]
        
        seed_pairs = np.array(list(product(S0r,S0c)))
        seed_pairs = np.concatenate([seed_pairs,np.arange(len(seed_pairs)).reshape(-1,1)],axis=1)

        out = []
        default_rho = rho.mean()
        for row1 in seed_pairs:
            s1, s2, idx = tuple(row1.astype(int))
            if s1 != s2:
                r = rho_mapper.get((s1,s2),default_rho)
                out.append([s1,s2,idx,r])

        out = np.array(out)
        Cs = np.zeros(shape = (nr,nc),dtype=float).flatten()
        Cs[out[:,-2].astype(int)] = out[:,-1]
        Cs = Cs.reshape(nr,nc)
        mask = S0r[:,None]==S0c
        Cs[mask] = 1.0
        return Cs
    def loglik(self,X0, S0, Z, theta, g, rho, stype, beta0 = None, covtype = "Gaussian", eps = MACHINE_DOUBLE_EPS):
        r'''
        Log Likelihood under correlated noise

        Parameters
        ----------
        X0: ndarray_like
            unique design matrix (of size nxd)
        S0: ndarray_like
            vector of seed information (size nx1)
        Z: ndarray_like
            observation (output) vector (size N)
        theta: ndarray_like
            lengthscale
        g: float
            noise variance
        rho: float
            seed correlation strength
        stype: str
            type of kronecker structure of the data
        beta0: float
            trend
        covtype: str
            covariance kernel to use
        eps: float
            amount of jitter on diagonal of covariance matrix

        Returns
        -------
        loglik: float
            log-likelihood
        '''
        n = X0.shape[0]
        d = X0.shape[1]

        Cx = cov_gen(X1 = X0, theta = theta, type = covtype)
        if self.ids is None:
            # mimics R's outer(S0,S0,'==')
            self.ids = S0 == S0[:,None]
        if isinstance(rho,float) or rho.size==1:
            Cs = np.full(shape=(n,n),fill_value=rho,dtype=float)
        else:
            Cs = self.pairwise_rho(S0c=S0,S0r=S0,rho=rho)
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
        loglik = -n/2 * np.log(2*np.pi) - n/2 * np.log(psi) + 1/2 * ldetKi - n/2
        return loglik
    
    def dloglik(self,X0, S0, Z, theta, g, rho, stype, beta0 = None, covtype = "Gaussian",
                          eps = MACHINE_DOUBLE_EPS, components = ("theta", "g", "rho")):
        r'''
        Gradient of log-likelihood under correlated noise

        Parameters
        ----------
        X0: ndarray_like
            matrix of designs (one point per row)
        S0: ndarray_like
            integer vector containing seed information
        Z: ndarray_like
            vector of outputs
        theta: ndarray_like
            scalar (isotropic) or vector (anisotropic) lengthscale values
        g: float
            noise variance
        rho: float
            seed correlation
        stype: str
            type of kronecker structure of the data
        beta0: float
            trend
        covtype: str
            covariance kernel to use
        eps: float
            amount of jitter on diagonal of covariance matrix
        components: tuple
            directions for partial derivatives, one or several of 'theta' (lengthscales) or 'g' (for noise) or 'rho' (seed correlation strength)


        Returns
        -------
        gradient with respect to hyperparameters (specified via ``components``)
        '''
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
            tmp2 = np.atleast_1d(tmp2)
        if 'rho' in components:
            dC_drho = np.ones(shape=(n,n))
            ids = self.ids
            dC_drho[ids] = 0
            dC_drho *= Cx
            tmp3 = 0.5 * (KiZ.T @ dC_drho) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_drho)
            tmp3 = np.atleast_1d(tmp3)
        return np.concatenate([tmp1,tmp2,tmp3]).squeeze()
    
    def loglikT(self,X0, S0, T0, Z, theta, g, rho, beta0 = None, covtype = "Gaussian", eps = MACHINE_DOUBLE_EPS):
        n = X0.shape[0]
        N = Z.size
        d = X0.shape[1]
        
        Cx = cov_gen(X1 = X0, theta = theta[0:d], type = covtype)
        if self.ids is None:
            # mimics R's outer(S0,S0,'==')
            self.ids = S0 == S0[:,None]
        Cs = np.full(shape=(n,n),fill_value=rho,dtype=float)
        Cs[self.ids] = 1.0

        Ct = cov_gen(X1 = T0, theta = np.atleast_1d(theta[d]), type = covtype)
        self.Cx = Cx
        self.Cs = Cs
        self.Ct = Ct

        SCxs = np.linalg.svd(Cx * Cs)
        SCt = np.linalg.svd(Ct)

        term1 = np.kron(SCt.U,SCxs.U) * np.repeat(1.0 / (np.kron(SCt.S,SCxs.S) + g),N).reshape(N,N).T
        term2 = np.kron(SCt.U,SCxs.U)
        Ki = term1 @ term2.T
        ldetKi = -1.0 * np.sum(np.log(np.kron(SCt.S,SCxs.S) + g))
        self.Ki = Ki
        
        Z = Z.flatten(order='F')
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z / Ki.sum()
        
        psi = (((Z - beta0).T @ Ki) @ (Z - beta0)) / N
        loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi) + 1/2 * ldetKi - N/2
        return loglik
    def dloglikT(self,X0, T0, S0, Z, theta, g, rho, beta0 = None, covtype = "Gaussian",
                           eps = MACHINE_DOUBLE_EPS, components = ("theta", "g", "rho")):
        n = X0.shape[0]
        N = Z.size
        d = X0.shape[1]

        Cx = self.Cx
        Cs = self.Cs
        Ct = self.Ct
        Ki = self.Ki

        Z = Z.flatten(order='F').copy()
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z / Ki.sum()

        Z = (Z - beta0).copy()
        KiZ = Ki.T @ Z

        psi = (Z.T @ KiZ).squeeze()

        tmp1, tmp2, tmp3 = np.array([]), np.array([]), np.array([])

        if 'theta' in components:
            tmp1 = np.full(shape=len(theta)-1,fill_value=np.nan)
            for i in range(len(theta)-1):
                dC_dthetak = partial_cov_gen(X1=X0,theta=theta[i],type=covtype,arg="theta_k") * Cx * Cs
                dC_dthetak = np.kron(Ct,dC_dthetak)
                tmp1[i] = 0.5 * (KiZ.T @ dC_dthetak) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_dthetak)
            # now for time
            dC_dthetak = partial_cov_gen(X1=T0,theta=theta[d],type=covtype,arg="theta_k") * Ct
            dC_dthetak = np.kron(dC_dthetak,Cx * Cs)
            tmp1 = np.concatenate([tmp1,
                                   np.atleast_1d(N/2 * (KiZ.T @ dC_dthetak) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_dthetak))])

        if 'g' in components:
            tmp2 = (N/2) * np.sum(KiZ**2) / psi - 0.5 * np.sum(np.diag(Ki))
            tmp2 = np.atleast_1d(tmp2)
        if 'rho' in components:
            dC_drho = np.ones(shape=(n,n))
            ids = self.ids
            dC_drho[ids] = 0
            dC_drho *= Cx
            dC_drho = np.kron(Ct,dC_drho)
            tmp3 = (N/2) * (KiZ.T @ dC_drho) @ KiZ / psi - 0.5 * np.trace(Ki @ dC_drho)
            tmp3 = np.atleast_1d(tmp3)

        return np.concatenate([tmp1,tmp2,tmp3])








    def mlecrnGP(self,X, Z, T0 = None,
                rhotype = "simple", 
                stype = "none", 
                lower = None, upper = None, 
                known = dict(),
                noiseControl = dict(g_bounds = np.array((10*MACHINE_DOUBLE_EPS, 1e2)),
                                    rho_bounds = None),
                init = dict(),
                covtype = "Gaussian",
                maxit = 100, 
                eps = MACHINE_DOUBLE_EPS, 
                settings = dict(return_Ki = True, factr = 1e7)):
        r'''
        Gaussian process modeling with correlated noise.

        You may also call this function as `crnGP.mle`

        Gaussian process regression under correlated noise based on maximum likelihood estimation of the hyperparameters.
        
        Parameters
        ----------
        X : ndarray_like
            matrix of all designs, one per row, or list with elements. The last column is assumed to contain the integer seed value.
        Z : ndarray_like
            Z vector of all observations.
        T0 : ndarray_like:
            T0 optional vector of times (same for all ``X's``). Not currently supported.
        lower,upper : ndarray_like 
            optional bounds for the ``theta`` parameter (see :func: covariance_functions.cov_gen for the exact parameterization).
            In the multivariate case, it is possible to give vectors for bounds (resp. scalars) for anisotropy (resp. isotropy)
        noiseControl : dict
            dict with element:
                - ``g_bounds`` vector providing minimal and maximal noise to signal ratio (default to ``(sqrt(MACHINE_DOUBLE_EPS), 100)``).
        settings : dict 
                dict for options about the general modeling procedure, with elements:
                    - ``return_Ki`` boolean to include the inverse covariance matrix in the object for further use (e.g., prediction).
                    - ``factr`` (default to 1e7) and ``pgtol`` are available to be passed to `options` for L-BFGS-B in :func: ``scipy.optimize.minimize``.   
        eps : float
            jitter used in the inversion of the covariance matrix for numerical stability
        known : dict
            optional dict of known parameters (e.g. ``beta0``, ``theta``, ``g``)
        init :  dict
            optional lists of starting values for mle optimization:
                - ``theta_init`` initial value of the theta parameters to be optimized over (default to 10% of the range determined with ``lower`` and ``upper``)
                - ``g_init`` vector of nugget parameter to be optimized over
        covtype : str 
                covariance kernel type, either ``'Gaussian'``, ``'Matern5_2'`` or ``'Matern3_2'``, see :func: ``~covariance_functions.cov_gen``
        maxit : int
                maximum number of iterations for `L-BFGS-B` of :func: ``scipy.optimize.minimize`` dedicated to maximum likelihood optimization
    
        Notes
        -------
        The global covariance matrix of the model is parameterized as ``nu_hat * (Cx + g Id) * Cs = nu_hat * K``,
        with ``Cx`` the spatial correlation matrix between unique designs, depending on the family of kernel used
        (see ``~covariance_functions.cov_gen`` for available choices) and values of lengthscale parameters. Cs is the correlation matrix between seed values,
        equal to 1 if the seeds are equal, ``rho`` otherwise. 
        ``nu_hat`` is the plugin estimator of the variance of the process.

        It is generally recommended to rescale the inputs to the unit cube and to normalize the outputs.
        
        Returns
        -------
        self, with the following attributes: 

            - ``theta``: unless given, maximum likelihood estimate (mle) of the lengthscale parameter(s),
            - ``nu_hat``: plugin estimator of the variance,
            - ``g``: unless given, mle of the nugget of the noise/log-noise process,
            - ``trendtype``: either ``"SK"`` if ``beta0`` is provided, else ``"OK"``,
            - ``beta0``: constant trend of the mean process, plugin-estimator unless given,
            - ``ll``: log-likelihood value,
            - ``nit_opt``, ``msg``: counts and message returned by :func:``scipy.optimize.minimize``
            - ``used_args``: list with arguments provided in the call to the function,
                - ``Ki``, ``Kgi``: inverse of the covariance matrices of the mean and noise processes (not scaled by ``nu_hat`` and ``nu_hat_var``),  
                - ``X0``, ``Z0``, ``Z``, ``eps``, ``logN``, ``covtype``: values given in input,
            - ``time``: time to train the model, in seconds.

        References
        ----------
        A Fadikar, M Binois, N Collier, A Stevens, KB Toh, J Ozik. Trajectory-oriented optimization of stochastic epidemiological models. 2023 Winter Simulation Conference (WSC), San Antonio, TX, USA, 2023, pp. 1244-1255, doi: 10.1109/WSC60868.2023.10408258

        Examples
        --------
        >>> from hetgpy import crnGP
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> pps = 10 # points per seed
        >>> x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
        >>> X = np.vstack([x,x])
        >>> seeds = ([1] * pps) + ([5] * pps)
        >>> X = np.hstack([X,np.array(seeds).reshape(-1,1)])
        >>> # amplitude and phase shift
        >>> Z = np.sin(X[:,0] + (np.pi/2)*(X[:,-1]==5)) + X[:,-1]
        >>> GP = crnGP()
        >>> GP.mle(X,Z,covtype="Matern5_2")
        '''
        # copy on instantiation
        init = init.copy()
        known = known.copy()
        noiseControl = noiseControl.copy()
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
        if np.abs(S0 - S0.astype(int)).sum() > 0:
            raise ValueError(f"Last col of X is assumed to contain integer-valued seed information.")
        S0 = S0.astype(int)

        if rhotype not in ['simple']:
            raise ValueError("rhotype must be one of ['simple']")
        if lower is None or upper is None:
            auto_thetas = auto_bounds(X = X0, covtype = covtype)
            if lower is None:
                lower = auto_thetas['lower']
            if upper is None:
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
            if rhotype == 'simple':
                noiseControl['rho_bounds'] = np.array([0.001, 0.9])
        g_min, g_max = noiseControl['g_bounds']
        
        beta0 = known.get('beta0')
        n = X0.shape[0]
        
        if known.get('theta') is None and init.get('theta') is None:
            init['theta'] = 0.9*lower + 0.1 * upper
        if known.get('g') is None and init.get('g') is None:
            init['g'] = 0.1  
        if known.get('rho') is None and init.get('rho') is None and rhotype == "simple":
            init['rho'] = 0.1

        
        trendtype = 'OK'
        if beta0 is not None:
            trendtype = 'SK'
        
        self.max_loglik = -1.0 * float('inf')
        self.arg_max = np.nan
        def fn(par, X0, S0, T0, Z, beta0, theta, g, rho):
            idx = 0
            if theta is None:
                theta = par[0:len(init['theta'])]
                idx = idx + len(init['theta'])
            if g is None:
                g = par[idx]
                idx = idx + 1
            if rho is None:
                rho = par[idx:].squeeze()
            if T0 is None:
                loglik = self.loglik(X0 = X0, S0 = S0,
                                     Z = Z, theta = theta, g = g,
                                     rho = rho, stype = stype, beta0 = beta0,
                                     covtype = covtype,eps = eps)
            else:
                loglik = self.loglikT(X0 = X0, S0 = S0, T0 = T0, Z = Z, 
                                      theta = theta, g = g, rho = rho, beta0 = beta0, 
                                      covtype = covtype, eps = eps)
            if not np.isnan(loglik):
                if loglik > self.max_loglik:
                    self.max_loglik = loglik
                    self.arg_max = par      
            return -1.0 * loglik
        
        def gr(par, X0, S0, T0, Z, beta0, theta, g, rho):
            idx = 0
            components = []

            if theta is None:
                theta = par[0:len(init['theta'])]
                idx = idx + len(init['theta'])
                components.append('theta')
            if g is None:
                g = par[idx]
                idx = idx + 1
                components.append('g')
            if rho is None:
                rho = par[idx:].squeeze()
                components.append('rho')
            if T0 is None:
                return -1.0 * self.dloglik(X0 = X0,S0 = S0, Z = Z,
                                    theta = theta, g = g, rho = rho, 
                                    stype = stype, beta0 = beta0, covtype = covtype, 
                                    eps = eps, components = components)
            else:
                return -1.0 * self.dloglikT(X0 = X0, T0 = T0, S0 = S0, Z = Z, 
                                            theta = theta, g = g, rho = rho, 
                                            beta0 = beta0, covtype = covtype,
                                            eps = eps, components = components)
        
        if known.get('g') is not None and known.get('theta') is not None and known.get('rho') is not None:
            theta_out = known['theta']
            g_out = known['g']
            rho_out = known['rho']
            if T0 is None:
                out = dict(
                    value = -1.0 * self.loglik(X0 = X0, S0 = S0,
                                     Z = Z, theta = theta_out, g = g_out,
                                     rho = rho_out, stype = stype, 
                                     beta0 = beta0,
                                     covtype = covtype,eps = eps),
                    message = 'All hyperparameters given',
                    counts = 0,
                    time = time() - tic
                )
            else:
                out = dict(
                    value = -1.0 * self.loglikT(X0 = X0, S0 = S0, T0 = T0, Z = Z, 
                                                theta = theta_out, g = g_out, rho = rho_out,
                                                beta0 = beta0, 
                                                covtype = covtype, eps = eps),
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
                parinit = np.concatenate([parinit, np.array([init['g']])])
                g_min = np.atleast_1d(g_min)
                g_max = np.atleast_1d(g_max)
                lowerOpt = np.concatenate([lowerOpt, g_min])
                upperOpt = np.concatenate([upperOpt, g_max])
            if known.get('rho') is None:   
                parinit = np.concatenate([parinit, np.atleast_1d(init['rho'])])
                lowerOpt = np.concatenate([
                                lowerOpt,
                                np.atleast_1d(noiseControl['rho_bounds'][0])
                            ])
                upperOpt = np.concatenate([
                                upperOpt,
                                np.atleast_1d(noiseControl['rho_bounds'][1])
                            ])
            
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
        if known.get('theta') is None:
            idx += len(init['theta'])
            theta_out = out['par'][0:idx]
            
        else:
            theta_out = known['theta']
        g_out = out['par'][idx] if known.get('g') is None else known.get('g')
        
        rho_out = out['par'][(idx+1):] if known.get('rho') is None else known.get('rho')

        

        Cx = cov_gen(X1 = X0, theta = np.atleast_1d(theta_out), type = covtype)
        if self.ids is None:
            # mimics R's outer(S0,S0,'==')
            self.ids = S0 == S0[:,None]
        if rhotype == 'simple':
            Cs = np.full(shape=(n,n),fill_value=rho_out,dtype=float)
        Cs[self.ids] = 1.0
        C = Cx * Cs
        
        self.C = C
        self.Cx = Cx
        self.Cs = Cs
        
        if T0 is not None:
            Ct = cov_gen(X1 = T0, theta = np.atleast_1d(theta_out[d]), type = covtype)
            SCxs = np.linalg.svd(Cx*Cs)
            SCt = np.linalg.svd(Ct)
            term1 = np.kron(SCt.U,SCxs.U) * (1.0 / np.kron(SCt.S,SCxs.S) + g_out)
            term2 = np.kron(SCt.U,SCxs.U)
            Ki = term1 @ term2.T
        else:
            jitter = (eps+g_out)*np.eye(n)
            Ki = np.linalg.cholesky(C + jitter).T
            # to mirror R's chol2inv: do the following:
            # expose dtrtri from lapack (for fast cholesky inversion of a triangular matrix)
            # use result to compute Ki (should match chol2inv)
            Ki = dtrtri(Ki)[0] #  -- equivalent of chol2inv -- see https://stackoverflow.com/questions/6042308/numpy-inverting-an-upper-triangular-matrix
            Ki = Ki @ Ki.T     #  -- equivalent of chol2inv
            self.Ki = Ki
        
        Zf = Z.flatten(order='F')
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Zf / Ki.sum()
        
        self.X0 = X0
        self.Z = Z
        self.covtype = covtype
        self.S0 = S0
        self.trendtype = trendtype
        self.nu_hat = (Zf - beta0).T @ self.Ki @ (Zf - beta0) / Zf.shape[0]
        self.ll = -1.0 * out['value']
        self.theta = theta_out
        self.g = g_out
        self.rho = rho_out
        self.time = time() - tic
        return
    
    def predict(self,x,xprime = None,t0 = None, interval: str | None = None, interval_lower: float | None  = None, interval_upper: float | None = None,**kw):
        r'''
        Prediction under correlated noise

        Parameters
        ----------
        x : ndarray_like
            matrix of designs locations to predict at (one point per row)
        xprime : ndarray_like
            optional second matrix of predictive locations to obtain the predictive covariance matrix between ``x`` and ``xprime``
        t0: ndarray_like
            vector of times (not currently supported)
        interval: str
            one of 'confidence' or 'predictive' which is a convenience method to return confidence/predictive intervals corresponding to `interval_lower` and `interval_upper`
        interval_lower: float
            lower of confidence/predictive interval
        interval_upper: float
            upper of confidence/predictive interval
        nugs_only : bool (default False)  
            if ``True``, only return noise variance prediction
        kwargs : dict
            optional additional elements (only used for nugs_only)

        Returns
        -------

        dict with elements:
            - ``mean``: kriging mean;
            - ``sd2``: kriging variance (filtered, e.g. without the nugget values)
            - ``nugs``: noise variance prediction
            - ``cov``: (returned if ``xprime`` is given) predictive covariance matrix between ``x`` and ``xprime``
            - ``confidence_interval``: prediction with kriging variance only
            - ``predictive_interval``: prediction with kriging and noise variance
        '''
        if len(x.shape)==1:
            x = x.reshape(-1,1)
            if (x.shape[1]-1) != self.X0.shape[1]:
                raise ValueError(f"Problem with x format")
        s = x[:,-1]
        x = x[:,0:-1]
        if xprime is not None:
            if len(xprime.shape)==1:
                xprime = xprime.reshape(-1,1)
            if (xprime.shape[1]-1) != self.X0.shape[1]:
                raise ValueError(f"Problem with xprime format")
            sp = xprime[:,-1]
            xprime = xprime[:,0:-1]
        
        interval_types = [None,'confidence','predictive']
        return_interval = False
        if interval is not None:
            list_interval = [interval] if type(interval)==str else interval
            if 'confidence' not in list_interval and 'predictive' not in list_interval:
                raise ValueError(f"interval must be one of 'confidence' or 'predictive' not {interval}")
            return_interval = True

        if "nugs_only" in kw and kw["nugs_only"]:
            return dict(nugs = np.repeat(self['nu_hat'] * self['g'], x.shape[0]))
        d = x.shape[1]

        self['Ki'] /= self['nu_hat']

        kx = self['nu_hat'] * cov_gen(X1=x,X2 = self['X0'],theta = self['theta'], type=self['covtype'])
        tmp = s[:,None] == self['S0']
        if isinstance(self['rho'],float) or self['rho'].size==1:
            ks = np.full(shape = (x.shape[0],self['X0'].shape[0]),fill_value=self['rho'])
        else:
            ks = self.pairwise_rho(S0r=s,S0c=self['S0'],rho=self['rho'])
        
        ks[tmp] = 1
        kx = kx * ks
        

        nugs = np.repeat(self['nu_hat'] * self['g'],x.shape[0])
        mean = self['beta0'] + kx @ (self['Ki'] @ (self['Z'] - self['beta0']))

        if self['trendtype'] == 'SK':
            sd2 = self['nu_hat'] - np.diag(kx @ (self['Ki'] @ kx.T))
        else:
            sd2 = self['nu_hat'] - np.diag(kx @ ((self['Ki'] @ kx.T))) + (1- (self['Ki'].sum(axis=0))@ kx.T)**2/self['Ki'].sum()
        
        if (sd2 < 0).any():
            warnings.warn('Numerical errors caused some negative predictive variances to be thresholded to zero. Consider using np.inv via rebuild.CRNGP')  
            sd2[sd2 < 0] = 0
        if xprime is not None:
            kxprime = self['nu_hat'] * cov_gen(X1 = self['X0'],X2=xprime,theta = self['theta'],type = self['covtype'])
            tmp =  self['S0'][:,None] == s
            ksprime = np.full(shape = (self['X0'].shape[0],xprime.shape[0]),fill_value=self['rho'])
            ksprime[tmp] = 1
            kxprime = kxprime * ksprime
        
            kxxprime = self['nu_hat'] * cov_gen(X1 = x,X2=xprime,theta = self['theta'],type = self['covtype'])       
            ksxxprime = np.full(shape = (x.shape[0],xprime.shape[0]),fill_value=self['rho'])
            tmp = s == sp[:,None]
            ksxxprime[tmp] = 1
            kxxprime = kxxprime * ksxxprime

            if self['trendtype'] == 'SK':
                if x.shape[0] < xprime.shape[0]:
                    cov = kxxprime - kx @ self['Ki'] @ kxprime
                else:
                    cov = kxxprime - kx @ (self['Ki'] @ kxprime)
            else:
                if x.shape[0] < xprime.shape[0]:
                    cov = kxxprime - kx @ self['Ki'] @ kxprime + ((1-(self['Ki'].sum(axis=0,keepdims=True))@ kx.T).T @ (1-self['Ki'].sum(axis=0,keepdims=True) @ kxprime))/self['Ki'].sum() 
                else:
                    cov = kxxprime - kx @ (self['Ki'] @ kxprime) + ((1-(self['Ki'].sum(axis=0,keepdims=True))@ kx.T).T @ (1-self['Ki'].sum(axis=0,keepdims=True) @ kxprime))/self['Ki'].sum()
        else:
            cov = None
        # rescale Ki
        self['Ki'] *= self['nu_hat']

        preds = dict(
            mean = mean,
            sd2 = sd2,
            nugs = nugs,
            cov = cov
        )
        if return_interval:
            preds['confidence_interval'] = {}
            if 'confidence' in list_interval:
                preds['confidence_interval']['lower'] = norm.ppf(interval_lower, loc = preds['mean'], scale = np.sqrt(preds['sd2'])).squeeze()
                preds['confidence_interval']['upper'] = norm.ppf(interval_upper, loc = preds['mean'], scale = np.sqrt(preds['sd2'])).squeeze()
            preds['predictive_interval'] = {}
            if 'predictive' in list_interval:
                preds['predictive_interval']['lower'] = norm.ppf(interval_lower, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()
                preds['predictive_interval']['upper'] = norm.ppf(interval_upper, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

        return preds
            