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
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)

class homGP():
    def __init__(self):
        self.mle = self.mleHomGP
        return
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self,item,value):
        self.__dict__[item] = value
    def get(self,key):
        return self.__dict__.get(key)
    
    def logLikHom(self,X0, Z0, Z, mult, theta, g, beta0 = None, covtype = "Gaussian", eps = MACHINE_DOUBLE_EPS, env = None):
        n = X0.shape[0]
        N = Z.shape[0]

        C = cov_gen(X1 = X0, theta = theta, type = covtype)
        self.C = C
        Ki = np.linalg.cholesky(C + np.diag(eps + g / mult) ).T
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

        psi_0 = (Z0 - beta0).T @ Ki @ (Z0 - beta0)
        psi = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g) + psi_0)
        
        # check for divide by zero warnings
        if psi <=0: return np.nan
        if g <= 0: return np.nan
        loglik = (-N / 2.0) * np.log(2*np.pi) - (N / 2.0) * np.log(psi) + (1.0 / 2.0) * ldetKi - (N - n)/2.0 * np.log(g) - (1.0 / 2.0) * np.log(mult).sum() - (N / 2.0)
            
        #print('loglik: ', loglik,'\n')
        return loglik
    

        
    def dlogLikHom(self,X0, Z0, Z, mult, theta, g, beta0 = None, covtype = "Gaussian",
                        eps = MACHINE_DOUBLE_EPS, components = ("theta", "g")):
        k = len(Z)
        n = X0.shape[0]
        
        C     = self.C # assumes these have been instantiated by a call to `logLikHom` first
        Ki    = self.Ki
        beta0 = self.beta0
        
        Z0 = Z0 - beta0
        Z  = Z - beta0

        KiZ0 = Ki @ Z0 ## to avoid recomputing  
        psi  = Z0.T @ KiZ0
        tmp1 = None
        tmp2 = None
        # First component, derivative with respect to theta
        if "theta" in components:
            tmp1 = np.repeat(np.nan, len(theta))
            if len(theta)==1:
                dC_dthetak = partial_cov_gen(X1 = X0, theta = theta, type = covtype, arg = "theta_k") * C
                tmp1 = k/2 * (KiZ0.T @ dC_dthetak) @ KiZ0 /(((Z.T @ Z) - (Z0 * mult).T @ Z0)/g + psi) - 1/2 * np.trace(Ki @ dC_dthetak) # replaces trace_sym
                tmp1 = np.array(tmp1).squeeze()
            else:
                for i in range(len(theta)):
                    # use i:i+1 to preserve vector structure -- see "An integer, i, returns the same values as i:i+1 except the dimensionality of the returned self is reduced by 1"
                    ## at: https://numpy.org/doc/stable/user/basics.indexing.html
                    # tmp1[i] <- k/2 * crossprod(KiZ0, dC_dthetak) %*% KiZ0 /((crossprod(Z) - crossprod(Z0 * mult, Z0))/g + psi) - 1/2 * trace_sym(Ki, dC_dthetak)
                    dC_dthetak = partial_cov_gen(X1 = X0[:,i:i+1], theta = theta[i], type = covtype, arg = "theta_k") * C
                    tmp1[i] = (k/2 * (KiZ0.T @ dC_dthetak) @ KiZ0 /(((Z.T @ Z) - (Z0 * mult).T @ Z0)/g + psi) - 1/2 * np.trace(Ki @ dC_dthetak)).squeeze() # replaces trace_sym
        # Second component derivative with respect to g
        if "g" in components:
            tmp2 = k/2 * ((Z.T @ Z - (Z0 * mult).T @ Z0)/g**2 + np.sum(KiZ0**2/mult)) / ((Z.T @ Z - (Z0 * mult).T @ Z0)/g + psi) - (k - n)/ (2*g) - 1/2 * np.sum(np.diag(Ki)/mult)
            tmp2 = np.array(tmp2).squeeze()
        
        out = np.hstack((tmp1, tmp2)).squeeze()
        out = out[~(out==None)].astype(float).reshape(-1,1)
        #print('dll', out, '\n')
        return out    

    def mleHomGP(self,X, Z, lower = None, upper = None, known = dict(),
                        noiseControl = dict(g_bounds = (MACHINE_DOUBLE_EPS, 1e2)),
                        init = {},
                        covtype = "Gaussian",
                        maxit = 100, eps = MACHINE_DOUBLE_EPS, settings = dict(returnKi = True, factr = 1e7)):
        r'''
        Gaussian process modeling with homoskedastic noise.

        You may also call this function as `model.mle`

        Gaussian process regression under homoskedastic noise based on maximum likelihood estimation of the hyperparameters. This function is enhanced to deal with replicated observations.
        
        Parameters
        ----------
        X : ndarray_like
            matrix of all designs, one per row, or list with elements:
            - ``X0`` matrix of unique design locations, one point per row
            - ``Z0`` vector of averaged observations, of length ``len(X0)``
            - ``mult`` number of replicates at designs in ``X0``, of length ``len(X0)``
        Z : ndarray_like
            Z vector of all observations. If using a list with ``X``, ``Z`` has to be ordered with respect to ``X0``, and of length ``sum(mult)``
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
        The global covariance matrix of the model is parameterized as ``nu_hat * (C + g * diag(1/mult)) = nu_hat * K``,
        with ``C`` the correlation matrix between unique designs, depending on the family of kernel used (see :func: `~hetgpy.covariance_functions.cov_gen` for available choices) and values of lengthscale parameters.
        ``nu_hat`` is the plugin estimator of the variance of the process.

        It is generally recommended to use :func: ``~find_reps.find_reps`` to pre-process the data, to rescale the inputs to the unit cube and to normalize the outputs.
        
        Returns
        -------
        self, with the following attributes: 

            - ``theta``: unless given, maximum likelihood estimate (mle) of the lengthscale parameter(s),
            - ``nu_hat``: plugin estimator of the variance,
            - ``g``: unless given, mle of the nugget of the noise/log-noise process,
            - ``trendtype``: either ``"SK"`` if ``beta0`` is provided, else ``"OK"``,
            - ``beta0``: constant trend of the mean process, plugin-estimator unless given,
            - ``ll``: log-likelihood value, (``ll_non_pen``) is the value without the penalty,
            - ``nit_opt``, ``msg``: counts and message returned by :func:``scipy.optimize.minimize``
            - ``used_args``: list with arguments provided in the call to the function,
                - ``Ki``, ``Kgi``: inverse of the covariance matrices of the mean and noise processes (not scaled by ``nu_hat`` and ``nu_hat_var``),  
                - ``X0``, ``Z0``, ``Z``, ``eps``, ``logN``, ``covtype``: values given in input,
            - ``time``: time to train the model, in seconds.
            
            See also `~hetgpy.homGP.homGP.predict` for predictions, `~hetgpy.homGP.update` for updating an existing model.
            ``summary`` and ``plot`` functions are available as well.
            `~homTP.mleHomTP` provide a Student-t equivalent.
        
        Examples
        --------
        >>> from hetgpy import homGP
        >>> from hetgpy.example_data import mcycle
        >>> m = mcycle()
        >>> model = homGP()
        >>> model.mle(m['times'],m['accel'],lower=[1.0],upper=[10.0],covtype="Matern5_2")
        
        References
        ----------
        M. Binois, Robert B. Gramacy, M. Ludkovski (2018), Practical heteroskedastic Gaussian process modeling for large simulation experiments,
        Journal of Computational and Graphical Statistics, 27(4), 808--821.
        Preprint available on arXiv:1611.05902.
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
            
            if np.isnan(loglik) == False:
                if loglik > self.max_loglik:
                    self.max_loglik = loglik
                    self.arg_max = par
            
            return -1.0 * loglik # for maximization
        
        def gr(par,X0, Z0, Z, mult, beta0, theta, g):
            
            idx = 0
            components = []

            if theta is None:
                theta = par[0:len(init['theta'])]
                idx = idx + len(init['theta'])
                components.append('theta')
            if g is None:
                g = par[idx]
                components.append('g')
            dll = self.dlogLikHom(X0 = X0, Z0 = Z0, Z = Z, mult = mult, theta = theta, g = g, beta0 = beta0, covtype = covtype, eps = eps,
                            components = components)
            return -1.0 * dll # for maximization
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
            
            out = optimize.minimize(
                    fun=fn, # for maximization
                    args = (X0, Z0, Z, mult, beta0, known.get('theta'), known.get('g')),
                    x0 = parinit,
                    jac=gr,
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
        ki = dtrtri(ki)[0]
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
    def predict(self, x, xprime = None,interval = None, interval_lower = None, interval_upper = None,**kw):

        if len(x.shape) == 1:
            x = x.reshape(-1,1)
            if x.shape[1] != self['X0'].shape[1]: raise ValueError("x is not a matrix")
        if xprime is not None and len(xprime.shape)==1:
            xprime = xprime.reshape(-1,1)
            if xprime.shape[1] != self['X0'].shape[1]: raise ValueError("xprime is not a matrix")
        
        interval_types = [None,'confidence','predictive']
        return_interval = False
        if interval is not None:
            list_interval = [interval] if type(interval)==str else interval
            if 'confidence' not in list_interval and 'predictive' not in list_interval:
                raise ValueError(f"interval must be one of 'confidence' or 'predictive' not {interval}")
            return_interval = True

        if "nugs_only" in kw and kw["nugs_only"]:
            return dict(nugs = np.repeat(self['nu_hat'] * self['g'], x.shape[0]))

        if self.get('Ki') is None:
            # these should be replaced with calls to self instead of self
            ki = np.linalg.cholesky(
            cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype']) + np.diag(self['eps'] + self['g'] / self['mult'])
            ).T
            ki = dtrtri(ki)[0]
            self['Ki'] = ki @ ki.T
        self['Ki'] /= self['nu_hat'] # this is a subtle difference between R and Python. 
        kx = self['nu_hat'] * cov_gen(X1 = x, X2 = self['X0'], theta = self['theta'], type = self['covtype'])
        nugs = np.repeat(self['nu_hat'] * self['g'], x.shape[0])
        mean = self['beta0'] + kx @ (self['Ki'] @ (self['Z0'] - self['beta0']))
        
        if self['trendtype'] == 'SK':
            sd2 = self['nu_hat'] - np.diag(kx @ (self['Ki'] @ kx.T))
        else:
            sd2 = self['nu_hat'] - np.diag(kx @ ((self['Ki'] @ kx.T))) + (1- (self['Ki'].sum(axis=0))@ kx.T)**2/self['Ki'].sum()
        
        if (sd2<0).any():
            sd2[sd2<0] = 0
            warnings.warn("Numerical errors caused some negative predictive variances to be thresholded to zero. Consider using ginv via rebuild.homGP")

        if xprime is not None:
            kxprime = self['nu_hat'] * cov_gen(X1 = self['X0'], X2 = xprime, theta = self['theta'], type = self['covtype'])
            if self['trendtype'] == 'SK':
                if x.shape[0] < xprime.shape[0]:
                    cov = self['nu_hat'] *  cov_gen(X1 = x, X2 = xprime, theta = self['theta'], type = self['covtype']) - kx @ self['Ki'] @ kxprime
                else:
                    cov = self['nu_hat'] *  cov_gen(X1 = x, X2 = xprime, theta = self['theta'], type = self['covtype']) - kx @ (self['Ki'] @ kxprime)
            else:
                if x.shape[0] < xprime.shape[0]:
                    cov = self['nu_hat'] *  cov_gen(X1 = x, X2 = xprime, theta = self['theta'], type = self['covtype']) - kx @ self['Ki'] @ kxprime + ((1-(self['Ki'].sum(axis=0,keepdims=True))@ kx.T).T @ (1-self['Ki'].sum(axis=0,keepdims=True) @ kxprime))/self['Ki'].sum() 
                else:
                    cov = self['nu_hat'] *  cov_gen(X1 = x, X2 = xprime, theta = self['theta'], type = self['covtype']) - kx @ (self['Ki'] @ kxprime) + ((1-(self['Ki'].sum(axis=0,keepdims=True))@ kx.T).T @ (1-self['Ki'].sum(axis=0,keepdims=True) @ kxprime))/self['Ki'].sum() 
            
        else:
            cov = None
        

        # re-modify self so Ki is preserved (because R does not modify lists in place)
        self['Ki']*=self['nu_hat']
        
        preds = dict(mean = mean, sd2 = sd2, nugs = nugs, cov = cov)
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

    def rebuild_homGP(self, robust = False):
        if robust :
            self['Ki'] <- np.linalg.pinv(
                cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype']) + np.diag(self['eps'] + self['g'] / self['mult'])
            ).T
            self['Ki'] /= self['nu_hat']
        else:
            ki = np.linalg.cholesky(
            cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype']) + np.diag(self['eps'] + self['g'] / self['mult'])
            ).T
            ki = dtrtri(ki)[0]
            self['Ki'] = ki @ ki.T
        return self

    def strip(self):
        keys  = ('Ki','Kgi','modHom','modNugs')
        for key in keys:
            if key in self.keys():
                del self[key]
        return self
    
    def update(self,Xnew, Znew, ginit = 1e-2, lower = None, upper = None, noiseControl = None, settings = None,
                         known = {}, maxit = 100):
        # first reduce Xnew/Znew in case of potential replicates
        newdata = find_reps(Xnew, Znew, normalize = False, rescale = False)
  
        if duplicated(np.vstack([self.X0, newdata['X0']])).any():
            id_exists = []
            for i in range(newdata['X0'].shape[0]):
                tmp = duplicated(np.vstack([newdata['X0'][i,:], self.X0]))
                if tmp.any():
                    id_exists.append(i)
                    id_X0 = tmp.nonzero()[0] - 1
                    self.Z0[id_X0] = (self.mult[id_X0] * self.Z0[id_X0] + newdata['Z0'][i] * newdata['mult'][i])/(self.mult[id_X0] + newdata['mult'][i])
                    idZ = np.cumsum(self.mult)+1
                    self.Z = np.insert(self.Z, values = newdata['Zlist'][i], obj = min(idZ[id_X0],len(idZ)))
                    
                    ## Inverse matrices are updated if MLE is not performed 
                    if maxit == 0:
                        self.Ki  = update_Ki_rep(id_X0, self, nrep = newdata['mult'][i])
                        self.nit_opt = 0
                        self.msg = "Not optimized \n"
                    
                    self.mult[id_X0] = self.mult[id_X0] + newdata['mult'][i]
                # remove duplicates now
            idxs = np.delete(np.arange(newdata['X0'].shape[0]),id_exists)
            newdata['X0']    = newdata['X0'][idxs,:]
            newdata['Z0']    = newdata['Z0'][idxs]
            newdata['mult']  = newdata['mult'][idxs]
            if type(newdata['Zlist'])==dict:
                newdata['Zlist'] = {k:v for k,v in newdata['Zlist'].items() if k in idxs}
                # decrement key indices
                Zlist = {}
                for i, val in enumerate(newdata['Zlist'].values()):
                    Zlist[i] = val
                newdata['Zlist'] = Zlist.copy()
                foo=1
            else:
                newdata['Zlist'] = newdata['Zlist'][idxs]
        if newdata['X0'].shape[0] > 0 and maxit==0:
            for i in np.arange(newdata['X0'].shape[0]):
                self.Ki = update_Ki(newdata['X0'][i:i+1,:], self, nrep = newdata['mult'][i], new_lambda = None)
                self.X0    = np.vstack([self.X0, newdata['X0']])        
                self.Z0    = np.hstack([self.Z0, newdata['Z0']])
                self.mult  = np.hstack([self.mult, newdata['mult']])
                if type(newdata['Zlist'])==dict:
                    self.Z     = np.hstack([self.Z, np.hstack(list(newdata['Zlist'].values()))])
                else:
                    self.Z     = np.hstack([self.Z, newdata['Zlist']])
            self.nit_opt = 0
            self.msg = "Not optimized \n"
        if maxit > 0:
            if upper is None: upper = self.used_args['upper']
            if lower is None: lower = self.used_args['lower']
            if noiseControl is None:
                noiseControl = self.used_args['noiseControl']
            init = {}
            if settings is None: settings = self.used_args.get('settings')
            if known == {}: known = self.used_args['known'].copy()
            if known.get('theta') is None: init['theta'] = self.theta
            if known.get('g') is None: init['g'] = self.g

             
            self.mleHomGP(X = dict(X0 = np.vstack([self.X0, newdata['X0']]), 
                                   Z0 = np.hstack([self.Z0, newdata['Z0']]), 
                                   mult = np.hstack([self.mult, newdata['mult']])), 
                                   Z = np.hstack([self.Z, 
                                                  np.array(list(chain(*newdata['Zlist'].values())))]
                                    ), 
                       lower = lower, 
                       upper = upper, 
                       noiseControl = noiseControl, 
                       covtype = self.covtype, 
                       init = init, 
                       known = known, 
                       eps = self.eps, 
                       maxit = maxit
            )
        return self
    def copy(self):
        '''
        Make a copy of the model, which is useful in tandem with the update function
        
        Parameters
        ----------
        None

        Returns
        -------
        newmodel: (deep) copy of model
        '''

        newmodel = deepcopy(self)
        return newmodel
    def plot(self,type='diagnostics'):
        ptypes = ('diagnostics','iterates')
        if type not in ptypes:
            raise ValueError(f"{type} not found, select one of {ptypes}")
        if type=='diagnostics':
            return plot_diagnostics(model=self)
        if type=='iterates':
            return plot_optimization_iterates(model=self)
    
    def summary(self):
        print("N = ", len(self.Z), " n = ", len(self.Z0), " d = ", self.X0.shape[1], "\n")
        print("Homoskedastic nugget value: ", self.g, "\n")
        print(self.covtype, " covariance lengthscale values: ", self.theta, "\n")
        print("Variance/scale hyperparameter: ", self.nu_hat, "\n")
        if self.trendtype == "SK":
            print("Given constant trend value: ", self.beta0, "\n")
        else:
            print("Estimated constant trend value: ", self.beta0, "\n")
        print("MLE optimization: \n", "Log-likelihood = ", self.ll, "; Nb of evaluations (obj, gradient) by L-BFGS-B: ", self.nit_opt, "; message: ", self.msg, "\n")

class homTP():
    pass
