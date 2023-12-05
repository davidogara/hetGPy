import numpy as np
import warnings
from time import time
from scipy.linalg.lapack import dtrtri
from scipy import optimize
from hetgpy.covariance_functions import cov_gen, partial_cov_gen
MACHINE_DOUBLE_EPS = np.sqrt(2.220446e-16) # From David's RStudio .Machine$double_eps

class hetGP:
    def __init__(self):
        return
    
    def auto_bounds(self):
        pass
    def find_reps(self,X,Z, return_Zlist = True, rescale = False, normalize = False, inputBounds = None):
        
        if type(X) != np.ndarray:
            raise ValueError(f"X must be a numpy array, is currently: {type(X)}")
        if X.shape[0] == 1:
            if return_Zlist:
                return dict(X0=X,Z0=Z,mult = 1, Z = Z, Zlist = dict(Z))
            return(dict(X0 = X, Z0 = Z, mult = 1, Z = Z))
        if len(X.shape) == 1: # if x is a 1D series
            raise ValueError(f"X appears to be a 1D array. Suggest reshaping with X.reshape(-1,1)")
        if rescale:
            if inputBounds is None:
                inputBounds = np.array([X.min(axis=0),
                                        X.max(axis=0)])
            X = (X - inputBounds[0,:]) @ np.diag(1/(inputBounds[1,:] - inputBounds[0,:]))
        outputStats = None
        if normalize:
            outputStats = np.array([Z.mean(), Z.var()])
            Z = (Z - outputStats[0])/np.sqrt(outputStats[1])
        X0 = np.unique(X, axis = 0)
        if X0.shape[0] == X.shape[0]:
            if return_Zlist:
                return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z, Zlist = Z,
                  inputBounds = inputBounds, outputStats = outputStats)
            return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z,
                inputBounds = inputBounds, outputStats = outputStats)
        
        # TODO: consider numba-ing this part. Replicating *split* in R is a bit tricky
        # consider something like: Zsplit = np.split(Z, np.unique(corresp, return_index=True)[1][1:])
        _, corresp = np.unique(X,axis=0,return_inverse=True)
        Zlist = {}
        Z0    = np.zeros(X0.shape[0], dtype=X0.dtype)
        mult  = np.zeros(X0.shape[0], dtype=X0.dtype)
        for ii in np.unique(corresp):
            out = Z[(ii==corresp).nonzero()[0]]
            
            Zlist[ii] = out
            Z0[ii]    = out.mean()
            mult[ii]  = len(out)
  
        if return_Zlist:
            return dict(X0 = X0, Z0 = Z0, mult = mult, Z = Z,
                Zlist = Zlist, inputBounds = inputBounds, outputStats = outputStats)
        return dict(X0 = X0, Z0 = Z0, mult = mult, Z = Z, inputBounds = inputBounds,
              outputStats = outputStats)
    

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
        #  psi <- 1/N * ((crossprod(Z - beta0) - crossprod((Z0 - beta0) * mult, Z0 - beta0))/g + psi_0)
        #t1 = (Z-beta0).T @ (Z-beta0)
        #t2 = ((Z0-beta0)*mult).T @ (Z0-beta0)
        #psi = (1.0 / N) * (((t1 - t2) / g) + psi_0)
        psi = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g) + psi_0)
        # loglik <- -N/2 * log(2*pi) - N/2 * log(psi) + 1/2 * ldetKi - (N - n)/2 * log(g) - 1/2 * sum(log(mult)) - N/2
        loglik = (-N / 2.0) * np.log(2*np.pi) - (N / 2.0) * np.log(psi) + (1.0 / 2.0) * ldetKi - (N - n)/2.0 * np.log(g) - (1.0 / 2.0) * np.sum(np.log(mult)) - (N / 2.0)
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
        tmp1 = tmp2 = None

        # First component, derivative with respect to theta
        if "theta" in components:
            tmp1 = np.repeat(np.nan, len(theta))
            if len(theta)==1:
                dC_dthetak = partial_cov_gen(X1 = X0, theta = theta, type = covtype, arg = "theta_k") * C
                tmp1 = k/2 * (KiZ0.T @ dC_dthetak) @ KiZ0 /(((Z.T @ Z) - (Z0 * mult).T @ Z0)/g + psi) - 1/2 * np.trace(Ki @ dC_dthetak) # replaces trace_sym
                tmp1 = np.array(tmp1)
            else:
                for i in range(len(theta)):
                    # use i:i+1 to preserve vector structure -- see "An integer, i, returns the same values as i:i+1 except the dimensionality of the returned object is reduced by 1"
                    ## at: https://numpy.org/doc/stable/user/basics.indexing.html
                    # tmp1[i] <- k/2 * crossprod(KiZ0, dC_dthetak) %*% KiZ0 /((crossprod(Z) - crossprod(Z0 * mult, Z0))/g + psi) - 1/2 * trace_sym(Ki, dC_dthetak)
                    dC_dthetak = partial_cov_gen(X1 = X0[:,i:i+1], theta = theta[i], type = covtype, arg = "theta_k") * C
                    tmp1[i] = k/2 * (KiZ0.T @ dC_dthetak) @ KiZ0 /(((Z.T @ Z) - (Z0 * mult).T @ Z0)/g + psi) - 1/2 * np.trace(Ki @ dC_dthetak) # replaces trace_sym
        # Second component derivative with respect to g
        if "g" in components:
            tmp2 = k/2 * ((Z.T @ Z - (Z0 * mult).T @ Z0)/g**2 + np.sum(KiZ0**2/mult)) / ((Z.T @ Z - (Z0 * mult).T @ Z0)/g + psi) - (k - n)/ (2*g) - 1/2 * np.sum(np.diag(Ki)/mult)
            tmp2 = np.array(tmp2).reshape(-1,1)
        
        out = np.hstack((tmp1.reshape(-1,1), tmp2.reshape(-1,1)))
        return out
    
    def mleHomGP(self,X, Z, lower = None, upper = None, known = dict(),
                     noiseControl = dict(g_bounds = (MACHINE_DOUBLE_EPS, 1e2)),
                     init = dict(),
                     covtype = ("Gaussian", "Matern5_2", "Matern3_2"),
                     maxit = 100, eps = MACHINE_DOUBLE_EPS, settings = dict(returnKi = True, factr = 1e7)):
        
        if type(X) == dict:
            X0 = X['X0']
            Z0 = X['Z0']
            mult = X['mult']
            if sum(mult) != len(Z):    raise ValueError(f"Length(Z) should be equal to sum(mult): they are {len(Z)} \n and {sum(mult)}")
            if len(X.shape) == 1:      warnings.warn(f"Coercing X0 to shape {len(X0)} x 1"); X0 = X0.reshape(-1,1)
            if len(Z0) != X0.shape[0]: raise ValueError("Dimension mismatch between Z0 and X0")
        else:
            if len(X.shape) == 1:    warnings.warn(f"Coercing X to shape {len(X)} x 1"); X = X.reshape(-1,1)
            if X.shape[0] != len(Z): raise ValueError("Dimension mismatch between Z and X")
            elem = self.find_reps(X, Z, return_Zlist = False)
            X0   = elem['X0']
            Z0   = elem['Z0']
            Z    = elem['Z']
            mult = elem['mult']

            # might need to change this
            covtypes = ("Gaussian", "Matern5_2", "Matern3_2")
            covtype = [c for c in covtypes if c in covtype][0]

            if lower is None or upper is None:
                auto_thetas = self.auto_bounds(X = X0, covtype = covtype)
                if lower is None: lower = auto_thetas['lower']
                if upper is None: upper = auto_thetas['upper']
                if known.get("theta") is None and init.get('theta') is None:  init['theta'] = np.sqrt(upper * lower)
            
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
            if known.get('g') is None and init.get('g') is None: 
                if any(mult > 2):
                    #t1 = mult.T
                    #t2 = (Z.squeeze() - np.repeat(Z0,mult.astype(int)))**2
                    init['g'] = np.mean(
                        (
                            (fast_tUY2(mult.T,(Z.squeeze() - np.repeat(Z0,mult.astype(int)))**2)/mult)[np.where(mult > 2)]
                        ))/np.var(Z0,ddof=1) 
                else:
                    init['g'] = 0.1
            trendtype = 'OK'
            if beta0 is not None:
               trendtype = 'SK'
            
            ## General definition of fn and gr
            self.max_loglik = float('-inf')
            self.arg_max = None
            self.init = init # save this as a global variable for dLoglikeHom
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
                        self.max_loglike = loglik
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
                # take max of mins and min of maxs
                bounds = [[lowerOpt.max(), upperOpt.min()]]
                out = optimize.minimize(
                    fun=fn, # for maximization
                    args = (X0, Z0, Z, mult, beta0, known.get('theta'), known.get('g')),
                    x0 = parinit,
                    jac=gr,
                    method="L-BFGS-B",
                    bounds = bounds,
                    tol=1e-8,
                    options=dict(maxiter=maxit, #,
                                ftol = settings.get('factr') * np.finfo(float).eps,#,
                                #gtol = settings.get('pgtol') # should map to pgtol
                                )
                    )
                    # out <- try(optim(par = parinit, fn = fn, gr = gr, method = "L-BFGS-B", lower = lowerOpt, upper = upperOpt, theta = known[["theta"]], g = known$g,
                    #         X0 = X0, Z0 = Z0, Z = Z, mult = mult, beta0 = beta0,
                    #         control = list(fnscale = -1, maxit = maxit, factr = settings$factr, pgtol = settings$pgtol), env = envtmp))
                python_kws_2_R_kws = {
                    'x':'par',
                    'fun': 'value',
                    'nit': 'counts'
                }
                for key, val in python_kws_2_R_kws.items():
                    out[val] = out[key]
                if out.success == False:
                    out = dict(par = self.arg_max, value = self.max_loglik, counts = np.nan,
                    message = "Optimization stopped due to NAs, use best value so far")

                g_out = out['par'][-1] if known.get('g') is None else known.get('g')
                theta_out = out['par'][0:len(init['theta'])] if known.get('theta') is None else known['theta']
            
            #Ki <- chol2inv(chol(add_diag(cov_gen(X1 = X0, theta = theta_out, type = covtype), eps + g_out/ mult)))

            ki = np.linalg.cholesky(
                cov_gen(X1 = X0, theta = theta_out, type = covtype) + np.diag(eps + g_out / mult)
                ).T
            ki = dtrtri(ki)[0]
            Ki = ki @ ki.T

            if beta0 is None:
                beta0 = Ki.sum(axis=1) @ Z0 / Ki.sum()
            
            psi_0 = (Z0 - beta0).T @ Ki @ (Z0 - beta0)

            nu = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g_out) + psi_0)
        
            res = dict(theta = theta_out, g = g_out, nu_hat = nu,
                        ll = -1.0 * out['fun'], nit_opt = out['counts'],
                        beta0 = beta0, trendtype = trendtype, covtype = covtype, msg = out['message'], eps = eps,
              X0 = X0, Z0 = Z0, Z = Z, mult = mult,
              used_args = dict(lower = lower, upper = upper, known = known, noiseControl = noiseControl),
              time = time() - tic)
            
            if settings["return_Ki"]: res['Ki']  = Ki
            return res
        def predict_hom_GP(self, object, x, xprime = NULL): return

