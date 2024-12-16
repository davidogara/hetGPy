from __future__ import annotations
import numpy as np
import warnings
from time import time
from scipy.linalg.lapack import dtrtri
from scipy import optimize
from scipy.special import digamma, polygamma
from scipy.stats import norm
from hetgpy.covariance_functions import cov_gen, partial_cov_gen
from hetgpy.utils import fast_tUY2, rho_AN, duplicated
from hetgpy.find_reps import find_reps
from hetgpy.auto_bounds import auto_bounds
from hetgpy import homGP
from hetgpy.plot import plot_optimization_iterates, plot_diagnostics
from hetgpy.LOO import LOO_preds
from hetgpy.update_covar import update_Ki, update_Ki_rep, update_Kgi, update_Kgi_rep
from copy import copy, deepcopy
import contextlib
from numpy.typing import ArrayLike, NDArray
NDArrayInt = NDArray[np.int_]
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)


class hetGP:
    def __init__(self):
        self.iterates = [] # for saving iterates during MLE
        self.use_torch = False
        self.mle = self.mleHetGP
        return
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self,item,value):
        self.__dict__[item] = value
    def get(self,key):
        return self.__dict__.get(key)
    # Part II: hetGP functions

    def logLikHet(self, X0: ArrayLike, Z0: ArrayLike, 
                        Z: ArrayLike, mult: NDArrayInt,
                        Delta: ArrayLike, theta: ArrayLike, 
                        g: ArrayLike, k_theta_g: ArrayLike | None = None, 
                        theta_g: ArrayLike | None = None, logN: bool | None = None, 
                        SiNK: bool = False,beta0: float | None = None, 
                      pX: ArrayLike | None = None, eps: float = MACHINE_DOUBLE_EPS, 
                      covtype: str = "Gaussian", SiNK_eps: float = 1e-4,
                      penalty: bool = True, hom_ll: float | None = None,
                      trace: int = 0) -> float:
        r'''
        log-likelihood in the anisotropic case - one lengthscale by variable
        Model: K = nu2 * (C + Lambda) = nu using all observations using the replicates information nu2 is replaced by its plugin estimator in the likelihood
        
        Parameters
        ----------

        X0 : ndarray_like
            unique designs
        Z0 : ndarray_like
            averaged observations
        Z  : ndarray_like
            replicated observations (sorted with respect to X0)
        mult : ndarray_like 
            number of replicates at each Xi
        Delta : ndarray_like
          vector of nuggets corresponding to each X0i or pXi, that are smoothed to give Lambda
        logN : bool
            should exponentiated variance be used
        SiNK : bool
            should the smoothing come from the SiNK predictor instead of the kriging one
        theta : ndarray_like
            scale parameter for the mean process, either one value (isotropic) or a vector (anistropic)
        k_theta_g: ndarray_like
            constant used for linking nuggets lengthscale to mean process lengthscale, i.e., theta_g[k] = k_theta_g * theta[k], alternatively theta_g can be used
        theta_g : ndarray_like
            either one value (isotropic) or a vector (anistropic), alternative to using k_theta_g
        g : ndarray_like
            nugget of the nugget process
        pX : ndarray_like
            matrix of pseudo inputs locations of the noise process for Delta (could be replaced by a vector to avoid double loop)
        beta0 : float
            mean, if not provided, the MLE estimator is used
        eps : float
            minimal value of elements of Lambda
        covtype: str
            covariance kernel type
        penalty : bool 
            should a penalty term on Delta be used?
        hom_ll : float 
            reference homoskedastic likelihood
        '''
        n = X0.shape[0]
        N = Z.shape[0]
  
        if theta_g is None: theta_g = k_theta_g * theta


        if pX is None or pX.size==0:
            Cg = cov_gen(X1 = X0, theta = theta_g, type = covtype)
            Kg_c = np.linalg.cholesky(Cg + np.diag(eps + g / mult) ).T
            Kgi = dtrtri(Kg_c)[0]
            Kgi = Kgi @ Kgi.T
            nmean = np.squeeze(Kgi.sum(axis=0) @ Delta / Kgi.sum()) # oridinary kriging mean
        
            if SiNK:
                rhox = 1 / rho_AN(xx = X0, X0 = X0, theta_g = theta_g, g = g, type = covtype, eps = eps, SiNK_eps = SiNK_eps, mult = mult)
                M =  rhox * Cg @ (Kgi @ (Delta - nmean))
            else:
                M = Cg @ (Kgi @ (Delta - nmean))
        
        else:
            Cg = cov_gen(X1 = X0, theta = theta_g, type = covtype)
            Kg_c = np.linalg.cholesky(Cg + np.diag(eps + g / mult) ).T
            Kgi = dtrtri(Kg_c)[0]
            Kgi = Kgi @ Kgi.T
            
            kg = cov_gen(X1 = X0, X2 = pX, theta = theta_g, type = covtype)
            
            nmean = np.squeeze(Kgi.sum(axis=0) @ Delta / Kgi.sum()) # oridinary kriging mean
        
            if SiNK:
                rhox = 1 / rho_AN(xx = X0, X0 = pX, theta_g = theta_g, g = g, type = covtype, eps = eps, SiNK_eps = SiNK_eps, mult = mult)
                M =  rhox * kg @ (Kgi @ (Delta - nmean))
            else:
                M = kg.squeeze() @ (Kgi @ (Delta - nmean))
        
        Lambda = np.squeeze(nmean + M)
  
        if logN :
            Lambda = np.exp(Lambda)
        
        else:
            Lambda[Lambda <= 0] = eps
        
        LambdaN = np.repeat(Lambda, repeats= mult)

        # Temporarily store Cholesky transform of K in Ki
        C = cov_gen(X1 = X0, theta = theta, type = covtype)
        self.C = C
        Ki = np.linalg.cholesky(C + np.diag(Lambda/mult + eps) ).T
        ldetKi = - 2.0 * np.sum(np.log(np.diag(Ki)))
        Ki = dtrtri(Ki)[0]
        Ki = Ki @ Ki.T
        
        self.Cg = Cg
        self.Kg_c = Kg_c
        self.Kgi = Kgi
        self.ldetKi = ldetKi
        self.Ki = Ki
        if beta0 is None:
            beta0 = Ki.sum(axis=1) @ Z0 / Ki.sum()

        psi_0 = (Z0 - beta0).T @ Ki @ (Z0 - beta0)
        
        psi = (1.0 / N)*((((Z-beta0)/ LambdaN).T @ (Z-beta0)) - ((Z0 - beta0)*mult/Lambda).T @ (Z0-beta0) + psi_0)
        try:
            loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi) + 1/2 * ldetKi - 1/2 * np.sum((mult - 1) * np.log(Lambda) + np.log(mult)) - N/2
        except RuntimeWarning as e:
            return np.nan
        if penalty:
            nu_hat_var = np.squeeze((Delta - nmean).T @ Kgi @ (Delta - nmean))/ len(Delta)
    
            ## To avoid 0 variance, e.g., when Delta = nmean
            if nu_hat_var < eps:
                return loglik
    
            pen = - n/2 * np.log(nu_hat_var) - np.sum(np.log(np.diag(Kg_c))) - n/2*np.log(2*np.pi) - n/2
            if(loglik < hom_ll and pen > 0):
                if trace > 0: warnings.warn("Penalty is deactivated when unpenalized likelihood is lower than its homGP equivalent")
                return loglik
            return loglik + pen
        return loglik
            
        
    
    def dlogLikHet(self,
                   X0: ArrayLike, Z0: ArrayLike, Z: ArrayLike, mult: NDArrayInt, Delta: ArrayLike, 
                   theta: ArrayLike, g: float, k_theta_g: ArrayLike | None = None, theta_g: ArrayLike| None = None, 
                   beta0: ArrayLike | None = None, pX: ArrayLike | None = None,
                   logN: bool = True, SiNK: bool = False, components: list = None, 
                    eps: float = MACHINE_DOUBLE_EPS, covtype: str = "Gaussian", 
                    SiNK_eps: float = 1e-4,penalty: bool = True, 
                    hom_ll: float = None) -> ArrayLike:
        '''
        derivative of log-likelihood for logLikHet_Wood with respect to theta and Lambda with all observations
        Model: K = nu2 * (C + Lambda) = nu using all observations using the replicates information 
        nu2 is replaced by its plugin estimator in the likelihood

        Parameters
        ----------
        X0 : ndarray_like
            unique designs
        Z0 : ndarray_like
            averaged observations
        Z  : ndarray_like
            replicated observations (sorted with respect to X0)
        mult : ndarray_like 
            number of replicates at each Xi
        Delta : ndarray_like
          vector of nuggets corresponding to each X0i or pXi, that are smoothed to give Lambda
        logN : bool
            should exponentiated variance be used
        SiNK : bool
            should the smoothing come from the SiNK predictor instead of the kriging one
        theta : ndarray_like
            scale parameter for the mean process, either one value (isotropic) or a vector (anistropic)
        k_theta_g: ndarray_like
            constant used for linking nuggets lengthscale to mean process lengthscale, i.e., theta_g[k] = k_theta_g * theta[k], alternatively theta_g can be used
        theta_g : ndarray_like
            either one value (isotropic) or a vector (anistropic), alternative to using k_theta_g
        g : ndarray_like
            nugget of the nugget process
        pX : ndarray_like
            matrix of pseudo inputs locations of the noise process for Delta (could be replaced by a vector to avoid double loop)
        components : ndarray_like
            components to determine which variable are to be taken in the derivation: None for all, otherwise list with elements from 'theta', 'Delta', 'theta_g', 'k_theta_g', 'pX' and 'g'.
        beta0 : float
            mean, if not provided, the MLE estimator is used
        eps : float
            minimal value of elements of Lambda
        covtype: str
            covariance kernel type
        penalty : bool 
            should a penalty term on Delta be used?
        hom_ll : float 
            reference homoskedastic likelihood
        '''
        ## Verifications
        if k_theta_g is None and theta_g is None:
            print("Either k_theta_g or theta_g must be provided \n")
        
        ## Initialisations
        
        # Which terms need to be computed
        if components is None:
            components = ["theta", "Delta", "g"]
            if k_theta_g is None:
                components.append("theta_g")
            else:
                components.append("k_theta_g")
            
            if pX is not None:
                components.append("pX")
        
        if theta_g is None:
            theta_g = k_theta_g * theta
        
        n = X0.shape[0]
        N = Z.shape[0]

        if not (self.Cg is None and self.Kg_c is None and self.Kgi is None):
            Cg   = self.Cg
            Kg_c = self.Kg_c
            Kgi  = self.Kgi
            if pX is None or pX.size == 0: 
                M = (Kgi * (-eps - g / mult)[:,None]) + np.diag(np.ones(n))
        else:
            if pX is None or pX.size == 0:
                Cg   = cov_gen(X1 = X0, theta = theta_g, type = covtype)
                Kg_c = dtrtri(np.linalg.cholesky(Cg + np.diag(eps + g/mult)).T)[0]
                Kgi  = Kg_c @ Kg_c.T
                M = (Kgi * (-eps - g / mult)[:,None]) + np.diag(np.ones(n))
            else:
                Cg   = cov_gen(X1 = X0, theta = theta_g, type = covtype)
                Kg_c = dtrtri(np.linalg.cholesky(Cg + np.diag(eps + g/mult)).T)[0]
                Kgi  = Kg_c @ Kg_c.T
                kg   = cov_gen(X1 = X0, X2 = pX, theta = theta_g, type = covtype)
        
        ## Precomputations for reuse
        rSKgi = Kgi.sum(axis=1)
        sKgi = Kgi.sum()
        
        nmean = np.squeeze(rSKgi @ Delta / sKgi) ## ordinary kriging mean
        
        ## Precomputations for reuse
        KgiD = Kgi @ (Delta - nmean)
        
        if SiNK:
            rhox = 1 / rho_AN(xx = X0, X0 = pX, theta_g = theta_g, g = g, type = covtype, eps = eps, SiNK_eps = SiNK_eps, mult = mult)
            M =  rhox * M
        
        Lambda = np.squeeze(nmean + M @ (Delta - nmean))
        if logN:
            Lambda = np.exp(Lambda)
        
        else:
            Lambda[Lambda <= 0] = eps
        
        LambdaN = np.repeat(Lambda, mult)

        if not (self.C is None and self.Ki is None and self.ldetKi is None):
            C = self.C
            Ki = self.Ki
            ldetKi = self.ldetKi
        else:
            C = cov_gen(X1 = X0, theta = theta, type = covtype)
            Ki = np.linalg.cholesky(C + np.diag(Lambda/mult + eps)).T
            ldetKi = - 2 * np.sum(np.log(np.diag(Ki))) # log determinant from Cholesky
            Ki = dtrtri(Ki)[0]
            Ki = Ki @ Ki.T
        if beta0 is None:
            beta0 = np.squeeze(Ki.sum(axis=1) @ Z0 / Ki.sum())
        
        ## Precomputations for reuse
        KiZ0 = Ki @ (Z0 - beta0)
        rsM = M.sum(axis=1)
        

        psi_0 = np.squeeze(KiZ0.T @(Z0 - beta0))
        # psi = (crossprod((Z - beta0)/LambdaN, Z - beta0) - crossprod((Z0 - beta0) * (mult/Lambda), Z0 - beta0) + psi_0)
        psi = ((((Z-beta0)/ LambdaN).T @ (Z-beta0)) - ((Z0 - beta0)*mult/Lambda).T @ (Z0-beta0) + psi_0)

        if penalty:
            nu_hat_var = np.squeeze((KgiD.T @ (Delta - nmean)))/len(Delta) 
            
            # To prevent numerical issues when Delta = nmean, resulting in divisions by zero
            if nu_hat_var < eps:
                penalty = False
            else:
                loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi/N)  + 1/2 * ldetKi - 1/2 * np.sum((mult - 1) * np.log(Lambda) + np.log(mult)) - N/2
                pen = - n/2 * np.log(nu_hat_var) - np.sum(np.log(np.diag(Kg_c))) - n/2*np.log(2*np.pi) - n/2
                if loglik < hom_ll and pen > 0: penalty = False
        
        dLogL_dtheta = dLogL_dDelta = dLogL_dkthetag = dLogL_dthetag = dLogL_dg = dLogL_dpX = None

        if "theta" in components:
            dLogL_dtheta = np.repeat(np.nan, len(theta))
            for i in range(len(theta)):
                if len(theta) ==1:
                    dC_dthetak = partial_cov_gen(X1 = X0, theta = theta, arg = "theta_k", type = covtype) * C # partial derivative of C with respect to theta
                else:
                    dC_dthetak = partial_cov_gen(X1 = X0[:, i:i+1], theta = theta[i], arg = "theta_k", type = covtype) * C # partial derivative of C with respect to theta
            
            
                if("k_theta_g" in components):
                
                    if pX is None:
                        if len(theta) == 1:
                            dCg_dthetak = partial_cov_gen(X1 = X0, theta = k_theta_g * theta, arg = "theta_k", type = covtype) * k_theta_g * Cg # partial derivative of Cg with respect to theta[i]
                        else:
                            dCg_dthetak = partial_cov_gen(X1 = X0[:, i:i+1], theta = k_theta_g * theta[i], arg = "theta_k", type = covtype) * k_theta_g * Cg # partial derivative of Cg with respect to theta[i]
                        
                
                        # Derivative Lambda / theta_k (first part)
                        if SiNK == False:
                            dLdtk = (dCg_dthetak @ KgiD).reshape(n,1) - M @ (dCg_dthetak @ KgiD).reshape(n,1)
                        
                        # Derivative Lambda / theta_k (first part)
                        if SiNK == True:
                            Kgitkg = M.T # for reuse
                            d_irho_dtheta_k = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dCg_dthetak @ Kgitkg) - np.diag(M @ (dCg_dthetak @ Kgitkg)) + np.diag(M @ dCg_dthetak.T))
                            dLdtk = (d_irho_dtheta_k * kg + rhox * (dCg_dthetak - (M) @ dCg_dthetak)) @ KgiD
                        
                    else:
                        if len(theta) == 1:
                            dCg_dthetak = partial_cov_gen(X1 = pX, theta = k_theta_g * theta, arg = "theta_k", type = covtype) * k_theta_g * Cg # partial derivative of Cg with respect to theta[i]
                            dkg_dthetak = partial_cov_gen(X1 = X0, X2 = pX, theta = k_theta_g * theta, arg = "theta_k", type = covtype) * k_theta_g * kg # partial derivative of kg with respect to theta[i]
                        else:
                            dCg_dthetak = partial_cov_gen(X1 = pX[:, i:i+1], theta = k_theta_g * theta[i], arg = "theta_k", type = covtype) * k_theta_g * Cg # partial derivative of Cg with respect to theta[i]
                            dkg_dthetak = partial_cov_gen(X1 = X0[:, i:i+1], X2 = pX[:, i:i+1], theta = k_theta_g * theta[i], arg = "theta_k", type = covtype) * k_theta_g * kg # partial derivative of kg with respect to theta[i]
                        
                        
                        # Derivative Lambda / theta_k (first part)
                        if SiNK == False:
                            dLdtk = (dkg_dthetak @ KgiD).reshape(n,1) - M @ (dCg_dthetak @ KgiD).reshape(n,1)
                        
                        # Derivative Lambda / theta_k (first part)
                        if SiNK == True:
                            Kgitkg = M.T # for reuse
                            d_irho_dtheta_k = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dkg_dthetak @ Kgitkg) - np.diag(M @ (dCg_dthetak @ Kgitkg)) + np.diag(M @ dkg_dthetak.T))
                            dLdtk = (d_irho_dtheta_k * kg + rhox * (dkg_dthetak - (M) @ dCg_dthetak)) @ KgiD
                        
                
                
                    # (second part)
                    dLdtk = dLdtk.squeeze() - (1 - rsM) * np.squeeze(rSKgi @ dCg_dthetak @ (Kgi @ Delta) * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dthetak @ rSKgi))/sKgi**2
                    
                    if logN:
                        dLdtk = dLdtk * Lambda
                    
                    dK_dthetak = dC_dthetak + np.diag(np.squeeze(dLdtk)/mult) # dK/dtheta[k]

                    t1 = (((Z - beta0)/LambdaN) * np.repeat(dLdtk,mult)).T @ ((Z - beta0)/LambdaN)
                    t2 = ((Z0 - beta0)/Lambda * mult * dLdtk).T @ ((Z0 - beta0)/Lambda)
                    t3 = (KiZ0.T @ dK_dthetak) @ KiZ0
                    t4 = 1/2 * np.trace(Ki @ dK_dthetak)
                    dLogL_dtheta[i] = N / 2 * (t1 - t2 + t3)/psi - t4
                    dLogL_dtheta[i] = dLogL_dtheta[i] - 1/2 * np.sum((mult - 1) * dLdtk/Lambda) # derivative of the sum(a_i - 1)log(lambda_i)
                    
                    if penalty:
                        dLogL_dtheta[i] = dLogL_dtheta[i]  + 1/2 * (KgiD.T @ dCg_dthetak) @ KgiD / nu_hat_var  - 1/2 * np.trace(Kgi @ dCg_dthetak)
                    
                else:
                    dLogL_dtheta[i] = N/2 * (KiZ0.T @ dC_dthetak) @ KiZ0/psi - 1/2 * np.trace(Ki @ dC_dthetak)
                    
        if len([comp in components for comp in ("Delta", "g", "k_theta_g", "theta_g", "pX")])>0:
            dLogLdLambda = N/2 * ((fast_tUY2(mult, (Z - beta0)**2) - (Z0 - beta0)**2 * mult)/Lambda**2 + KiZ0**2/mult)/psi - (mult - 1)/(2*Lambda) - 1/(2*mult) * np.diag(Ki)
            
            if logN:
                dLogLdLambda = Lambda * dLogLdLambda
        ## Derivative of Lambda with respect to Delta
        if "Delta" in components:
            dLogL_dDelta = (M.T @ dLogLdLambda) + rSKgi/sKgi*sum(dLogLdLambda) -  rSKgi / sKgi * np.sum((M.T @ dLogLdLambda)) #chain rule
        
        # Derivative Lambda / k_theta_g
        if "k_theta_g" in components:
            if pX is None or pX.size==0:
                dCg_dk = partial_cov_gen(X1 = X0, theta = theta, k_theta_g = k_theta_g, arg = "k_theta_g", type = covtype) * Cg
            
                if SiNK:
                    d_irho_dkthetag = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dCg_dk @ Kgitkg) - np.diag(M @ (dCg_dk @ Kgitkg)) + np.diag(M, dCg_dk.T))
                    dLogL_dkthetag = (d_irho_dkthetag * kg + rhox*(dCg_dk - M @ dCg_dk)) @ KgiD - \
                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dk @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dk @ rSKgi))/sKgi**2
                else:
                    dLogL_dkthetag = np.squeeze((dCg_dk @ KgiD).reshape(n,1) - M @(dCg_dk @ KgiD).reshape(n,1)) - \
                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dk @ (Kgi @ Delta) * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dk @ rSKgi))/sKgi**2
                    foo = 1
            else:
                dCg_dk = partial_cov_gen(X1 = pX, theta = theta, k_theta_g = k_theta_g, arg = "k_theta_g", type = covtype) * Cg
                dkg_dk = partial_cov_gen(X1 = X0, X2 = pX, theta = theta, k_theta_g = k_theta_g, arg = "k_theta_g", type = covtype) * kg
                
                if SiNK:
                    d_irho_dkthetag = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dkg_dk @ Kgitkg) - np.diag(M, (dCg_dk @ Kgitkg)) + np.diag(M @ dkg_dk.T))
                    dLogL_dkthetag = (d_irho_dkthetag * kg + rhox*(dkg_dk - M @ dCg_dk)) @ KgiD - \
                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dk @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dk @ rSKgi))/sKgi**2
                else:
                    dLogL_dkthetag = dkg_dk @ KgiD - M @ (dCg_dk @ KgiD) - \
                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dk @ (Kgi @ Delta) * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dk @ rSKgi))/sKgi**2
                
            
            dLogL_dkthetag = dLogL_dkthetag.T @ dLogLdLambda ## chain rule
                
        if "theta_g" in components:
            dLogL_dthetag = np.repeat(np.nan, len(theta_g))
        
            for i in range(len(theta_g)):
                if pX is None:
                    
                    if len(theta_g) == 1:
                        dCg_dthetagk = partial_cov_gen(X1 = X0, theta = theta_g, arg = "theta_k", type = covtype) * Cg # partial derivative of Cg with respect to theta
                    else:
                        dCg_dthetagk = partial_cov_gen(X1 = X0[:, i:i+1], theta = theta_g[i], arg = "theta_k", type = covtype) * Cg # partial derivative of Cg with respect to theta
                    
                    
                    if SiNK:
                        d_irho_dtheta_gk = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dCg_dthetagk @ Kgitkg) - np.diag(M @ (dCg_dthetagk @ Kgitkg)) + np.diag(M @ dCg_dthetagk.T))
                        dLogL_dthetag[i] = ((d_irho_dtheta_gk * kg + rhox * (dCg_dthetagk - M @ dCg_dthetagk)) @ KgiD - \
                                                    (1 - rsM) * (rSKgi @ dCg_dthetagk @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dthetagk @ rSKgi))/sKgi**2).T @ dLogLdLambda #chain rule
                    else:
                        dLogL_dthetag[i] = (dCg_dthetagk @ KgiD - M @ (dCg_dthetagk @ KgiD) - \
                                                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dthetagk @ (Kgi @ Delta) * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dthetagk @ rSKgi))/sKgi**2).T @ dLogLdLambda #chain rule
                    
                else:
                    if len(theta_g) == 1:
                        dCg_dthetagk = partial_cov_gen(X1 = pX, theta = theta_g, arg = "theta_k", type = covtype) * Cg # partial derivative of Cg with respect to theta
                        dkg_dthetagk = partial_cov_gen(X1 = X0, X2 = pX, theta = theta_g, arg = "theta_k", type = covtype) * kg # partial derivative of Cg with respect to theta
                    else:
                        dCg_dthetagk = partial_cov_gen(X1 = pX[:, i:i+1], theta = theta_g[i], arg = "theta_k", type = covtype) * Cg # partial derivative of Cg with respect to theta
                        dkg_dthetagk = partial_cov_gen(X1 = X0[:, i:i+1], X2 = pX[:, i:i+1], theta = theta_g[i], arg = "theta_k", type = covtype) * kg # partial derivative of Cg with respect to theta
                    
                    
                    if SiNK:
                        d_irho_dtheta_gk = -1/2 * (np.diag(M @(kg.T))**(-3/2) * (np.diag(dkg_dthetagk @ Kgitkg) - np.diag(M @ (dCg_dthetagk @ Kgitkg)) + np.diag(M @ dkg_dthetagk.T)))
                        dLogL_dthetag[i] = ((d_irho_dtheta_gk * kg + rhox * (dkg_dthetagk - M @ dCg_dthetagk)) @ KgiD - \
                                                    (1 - rsM) * (rSKgi @ dCg_dthetagk @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dthetagk @ rSKgi))/sKgi**2).T @ dLogLdLambda #chain rule
                    else:
                        dLogL_dthetag[i] = (dkg_dthetagk @ KgiD - M @ (dCg_dthetagk @ KgiD) - \
                                                    (1 - rsM) * np.squeeze(rSKgi @ dCg_dthetagk @ (Kgi @ Delta) * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dthetagk @ rSKgi))/sKgi**2).T @ dLogLdLambda #chain rule
        
                # Penalty term
                if penalty: dLogL_dthetag[i] = dLogL_dthetag[i] + 1/2 * (KgiD.T @ dCg_dthetagk) @ KgiD/nu_hat_var - np.trace(Kgi @ dCg_dthetagk)/2 
            
        ## Derivative Lambda / g
        if "g" in components:
            if SiNK:
                A0 = np.diag(np.repeat(1/mult, n))
                d_irho_dg = -1/2 * (np.diag(M @ kg.T))**(-3/2) * np.diag((-M @ A0) @ Kgitkg)
                dLogL_dg = ((d_irho_dg * M - rhox * M @ A0 @ Kgi) @ (Delta - nmean) - (1 - rsM) * ((Kgi @ A0 @ Kgi).sum(axis=0) @ Delta * sKgi - rSKgi @ Delta * np.sum(rSKgi**2/mult))/sKgi**2).T @ dLogLdLambda #chain rule
            else:
                dLogL_dg = (-M @ (KgiD/mult) - (1 - rsM) * np.squeeze(Delta @ (Kgi @ (rSKgi/mult)) * sKgi - rSKgi @ Delta * np.sum(rSKgi**2/mult))/sKgi**2).T @ dLogLdLambda #chain rule
        # Derivative Lambda/pX
        if "pX" in components:
            dLogL_dpX = np.repeat(np.nan, len(pX))
            for i in np.arange(pX.shape[0]):
                for j in np.arange(pX.shape[1]):
                    dCg_dpX = partial_cov_gen(X1 = pX, theta = theta_g, i1 = i, i2 = j, arg = "X_i_j", type = covtype) * Cg
                    dkg_dpX = (partial_cov_gen(X1 = pX, X2 = X0, theta = theta_g, i1 = i, i2 = j, arg = "X_i_j", type = covtype)).T * kg
                    
                    if SiNK:
                        d_irho_dX_i_j = -1/2 * (np.diag(M @ kg.T))**(-3/2) * (np.diag(dkg_dpX @ Kgitkg) - np.diag(M @ (dCg_dpX @ Kgitkg)) + np.diag(M @ dkg_dpX.T))
                        dLogL_dpX[(j-1)*pX.shape[0] + i] = ((d_irho_dX_i_j * kg + rhox * (dkg_dpX - M @ dCg_dpX)) @ KgiD - \
                                                                (1 - rsM) * (rSKgi @ dCg_dpX @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dpX @ rSKgi))/sKgi**2).T @ dLogLdLambda
                    else:   
                        dLogL_dpX[(j-1)*pX.shape[0] + i] = ((dkg_dpX - M @ dCg_dpX) @ KgiD - \
                                                                (1 - rsM) * (rSKgi @ dCg_dpX @ Kgi @ Delta * sKgi - rSKgi @ Delta * (rSKgi @ dCg_dpX @ rSKgi))/sKgi**2).T @ dLogLdLambda
                    
                    
                    if penalty: dLogL_dpX[(j-1)*pX.shape[0] + i] = dLogL_dpX[(j-1)*pX.shape[0] + i] - 1/2 * (KgiD.T @ dCg_dpX) @ KgiD / nu_hat_var - np.trace(Kgi@ dCg_dpX)/2         
        # Additional penalty terms on Delta
        if penalty:
            if "Delta" in components:
                dLogL_dDelta = dLogL_dDelta - KgiD / nu_hat_var
            
            if "k_theta_g" in components:
                dLogL_dkthetag = dLogL_dkthetag + 1/2 * (KgiD.T @ dCg_dk) @ KgiD / nu_hat_var - np.trace(Kgi @ dCg_dk)/2 
            
            if "g" in components:
                dLogL_dg = dLogL_dg + 1/2 * ((KgiD/mult).T @ KgiD) / nu_hat_var - np.sum(np.diag(Kgi)/mult)/2

        out = np.hstack([dLogL_dtheta,
           dLogL_dDelta,
           dLogL_dkthetag,
           dLogL_dthetag,
           dLogL_dg,
           dLogL_dpX]).squeeze()
        return out[~(out==None)].astype(float)
            
        
        

    
    def mleHetGP(self,X: ArrayLike, 
                 Z: ArrayLike, 
                 lower: ArrayLike | None = None, upper: ArrayLike | None = None,known: dict = dict(),
                noiseControl: dict = dict(k_theta_g_bounds = (1, 100), g_max = 100, g_bounds = (1e-06, 1)),
                init: dict = {},
                covtype: str = "Gaussian",
                maxit: int = 100, 
                eps: float = MACHINE_DOUBLE_EPS, 
                settings: dict = dict(returnKi = True, factr = 1e9,ignore_MLE_divide_invalid = True),use_torch: bool=False) -> None:
        r'''
        Gaussian process modeling with heteroskedastic noise

        You may also call this function as `model.mle`

        Gaussian process regression under input dependent noise based on maximum likelihood estimation of the hyperparameters. 
        A second GP is used to model latent (log-) variances. This function is enhanced to deal with replicated observations.

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
            dict with elements related to optimization of the noise process parameters:
                - ``g_min``, ``g_max`` minimal and maximal noise to signal ratio (of the mean process)
                - ``lowerDelta``, ``upperDelta`` optional vectors (or scalars) of bounds on ``Delta``, of length ``len(X0)`` (default to ``np.repeat(eps, X0.shape[0])`` and ``np.repeat(noiseControl["g_max"], X0.shape[0])`` resp., or their ``log``) 
                - ``lowerpX``, ``upperpX`` optional vectors of bounds of the input domain if `pX` is used.
                - ``lowerTheta_g``, ``upperTheta_g`` optional vectors of bounds for the lengthscales of the noise process if ``linkThetas == 'none'``. Same as for ``theta`` if not provided.
                - ``k_theta_g_bounds`` if ``linkThetas == 'joint'``, vector with minimal and maximal values for ``k_theta_g`` (default to ``(1, 100)``). See Notes.
                - ``g_bounds`` vector for minimal and maximal noise to signal ratios for the noise of the noise process, i.e., the smoothing parameter for the noise process. (default to ``(1e-6, 1)``).
        settings : dict 
                dict for options about the general modeling procedure, with elements:
                    - ``linkThetas`` defines the relation between lengthscales of the mean and noise processes. Either ``'none'``, ``'joint'``(default) or ``'constr'``, see Notes.
                    - ``logN``, when ``True`` (default), the log-noise process is modeled.
                    - ``initStrategy`` one of ``'simple'``, ``'residuals'`` (default) and ``'smoothed'`` to obtain starting values for ``Delta``, see Notes
                    - ``penalty``  when ``True``, the penalized version of the likelihood is used (i.e., the sum of the log-likelihoods of the mean and variance processes, see References).
                    - ``hardpenalty`` is ``True``, the log-likelihood from the noise GP is taken into account only if negative (default if ``maxit > 1000``).
                    - ``checkHom`` when ``True``, if the log-likelihood with a homoskedastic model is better, then return it.
                    - ``trace`` optional scalar (default to ``0``). If positive, tracing information on the fitting process. If ``1``, information is given about the result of the heterogeneous model optimization. Level ``2`` gives more details. Level ``3`` additionaly displays all details about initialization of hyperparameters.
                    - ``return_matrices`` boolean to include the inverse covariance matrix in the object for further use (e.g., prediction).
                    - ``return_hom`` boolean to include homoskedastic GP models used for initialization (i.e., ``modHom`` and ``modNugs``).
                    - ``factr`` (default to 1e9) and ``pgtol`` are available to be passed to `options` for L-BFGS-B in :func: ``scipy.optimize.minimize``.   
        eps : float
            jitter used in the inversion of the covariance matrix for numerical stability
        init,known :  dict
            optional lists of starting values for mle optimization or that should not be optimized over, respectively.
            Values in ``known`` are not modified, while it can happen to these of ``init``, see Notes. 
            One can set one or several of the following:
                - ``theta`` lengthscale parameter(s) for the mean process either one value (isotropic) or a vector (anistropic)
                - ``Delta`` vector of nuggets corresponding to each design in ``X0``, that are smoothed to give ``Lambda`` (as the global covariance matrix depends on ``Delta`` and ``nu_hat``, it is recommended to also pass values for ``theta``)
                - ``beta0`` constant trend of the mean process
                - ``k_theta_g`` constant used for link mean and noise processes lengthscales, when ``settings['linkThetas'] == 'joint'``
                - ``theta_g`` either one value (isotropic) or a vector (anistropic) for lengthscale parameter(s) of the noise process, when ``settings['linkThetas'] != 'joint'``
                - ``g`` scalar nugget of the noise process
                - ``g_H`` scalar homoskedastic nugget for the initialisation with a :func: homGP.mleHomGP. See Notes.
                - ``pX`` matrix of fixed pseudo inputs locations of the noise process corresponding to Delta
        covtype : str 
                covariance kernel type, either ``'Gaussian'``, ``'Matern5_2'`` or ``'Matern3_2'``, see :func: ``~covariance_functions.cov_gen``
        maxit : int
                maximum number of iterations for `L-BFGS-B` of :func: ``scipy.optimize.minimize`` dedicated to maximum likelihood optimization
    
        Notes
        -------
        The global covariance matrix of the model is parameterized as ``nu_hat * (C + Lambda * np.diag(1/mult)) = nu_hat * K``,
        with ``C`` the correlation matrix between unique designs, depending on the family of kernel used (see :func: `~covariance_functions.cov_gen` for available choices) and values of lengthscale parameters.
        ``nu_hat`` is the plugin estimator of the variance of the process.
        ``Lambda`` is the prediction on the noise level given by a second (homoskedastic) GP:
        .. math:: \Lambda = C_g(C_g + \mathrm{diag}(g/\mathrm{mult}))^{-1} \Delta}
        with ``C_g`` the correlation matrix between unique designs for this second GP, with lengthscales hyperparameters ``theta_g`` and nugget ``g``
        and ``Delta`` the variance level at ``X0`` that are estimated.
        
        It is generally recommended to use :func: ``~find_reps.find_reps`` to pre-process the data, to rescale the inputs to the unit cube and to normalize the outputs.
        
        The noise process lengthscales can be set in several ways:
            - using ``k_theta_g`` (``settings['linkThetas'] == 'joint'``), supposed to be greater than one by default. In this case lengthscales of the noise process are multiples of those of the mean process.
            - if ``settings['linkThetas'] == 'constr``, then the lower bound on ``theta_g`` correspond to estimated values of a homoskedastic GP fit.
            - else lengthscales between the mean and noise process are independent (both either anisotropic or not).
        
        When no starting nor fixed parameter values are provided with ``init`` or ``known``, 
        the initialization process consists of fitting first an homoskedastic model of the data, called ``modHom``.
        ``init['theta']``, initial lengthscales are taken at 10\% of the range determined with ``lower`` and ``upper``,
        while ``init['g_H']`` may be use to pass an initial nugget value.
        The resulting lengthscales provide initial values for ``theta`` (or update them if given in ``init``).
        
        If necessary, a second homoskedastic model, ``modNugs``, is fitted to the empirical residual variance between the prediction
        given by ``modHom`` at ``X0`` and ``Z`` (up to ``modHom['nu_hat]'``).
        Note that when specifying ``settings['linkThetas'] == 'joint', then this second homoskedastic model has fixed lengthscale parameters.
        Starting values for ``theta_g`` and ``g`` are extracted from ``modNugs`` 
        
        Three initialization schemes for ``Delta`` are available with ``settings['initStrategy']``: 
            - for ``settings['initStrategy'] == 'simple'``, ``Delta`` is simply initialized to the estimated ``g`` value of ``modHom``. 
            - Note that this procedure may fail when ``settings['penalty'] == True``.
            - for ``settings['initStrategy'] == 'residuals'``, ``Delta`` is initialized to the estimated residual variance from the homoskedastic mean prediction.
            - for  ``settings['initStrategy'] == 'smoothed'``, ``Delta`` takes the values predicted by ``modNugs`` at ``X0``.
            
        Notice that ``lower`` and ``upper`` bounds cannot be equal for ``:func: scipy.optimize.minimize``.
        To use pseudo-input locations for the noise process, one can either provide ``pX`` if they are not to be optimized.
        Otherwise, initial values are given with ``pXinit``, and optimization bounds with ``lowerpX``, ``upperpX`` in ``init``.
        Automatic initialization of the other parameters without restriction is available for now only with method 'simple',
        otherwise it is assumed that pXinit points are a subset of X0.

        Returns
        -------
        self, with the following attributes: 

            - ``theta``: unless given, maximum likelihood estimate (mle) of the lengthscale parameter(s),
            - ``Delta``: unless given, mle of the nugget vector (non-smoothed),
            - ``Lambda``: predicted input noise variance at ``X0``, 
            - ``nu_hat``: plugin estimator of the variance,
            - ``theta_g``: unless given, mle of the lengthscale(s) of the noise/log-noise process,
            - ``k_theta_g``: if ``settings['linkThetas'] == 'joint'``, mle for the constant by which lengthscale parameters of ``theta`` are multiplied to get ``theta_g``,
            - ``g``: unless given, mle of the nugget of the noise/log-noise process,
            - ``trendtype``: either ``"SK"`` if ``beta0`` is provided, else ``"OK"``,
            - ``beta0``: constant trend of the mean process, plugin-estimator unless given,
            - ``nmean``: plugin estimator for the constant noise/log-noise process mean,
            - ``pX``: if used, matrix of pseudo-inputs locations for the noise/log-noise process,
            - ``ll``: log-likelihood value, (``ll_non_pen``) is the value without the penalty,
            - ``nit_opt``, ``msg``: counts and message returned by :func:``scipy.optimize.minimize``
            - ``modHom``: homoskedastic GP model of class ``homGP`` used for initialization of the mean process,
            - ``modNugs``: homoskedastic GP model of class ``homGP`` used for initialization of the noise/log-noise process,
            - ``nu_hat_var``: variance of the noise process,
            - ``used_args``: list with arguments provided in the call to the function,
                - ``Ki``, ``Kgi``: inverse of the covariance matrices of the mean and noise processes (not scaled by ``nu_hat`` and ``nu_hat_var``),  
                - ``X0``, ``Z0``, ``Z``, ``eps``, ``logN``, ``covtype``: values given in input,
            - ``time``: time to train the model, in seconds.
            
            See also `~hetgpy.hetGP.hetGP.predict` for predictions, `~hetgpy.hetGP.update` for updating an existing model.
            ``summary`` and ``plot`` functions are available as well.
            `~hetTP.mleHetTP` provide a Student-t equivalent.

        Examples
        --------
        >>> from hetgpy import hetGP
        >>> from hetgpy.example_data import mcycle
        >>> m = mcycle()
        >>> model = hetGP()
        >>> model.mle(m['times'],m['accel'],lower=[1.0],upper=[10.0],covtype="Matern5_2")    
        
        References
        ----------
        M. Binois, Robert B. Gramacy, M. Ludkovski (2018), Practical heteroskedastic Gaussian process modeling for large simulation experiments,
        Journal of Computational and Graphical Statistics, 27(4), 808--821.
        Preprint available on arXiv:1611.05902.
        '''

        # copy dicts upon import to make sure they aren't passed around between model runs
        known = known.copy()
        init  = init.copy()
        noiseControl = noiseControl.copy()
        if type(X) == dict:
            X0 = X['X0']
            Z0 = X['Z0']
            mult = X['mult']
            if np.sum(mult) != len(Z):    raise ValueError(f"Length(Z) should be equal to sum(mult): they are {len(Z)} \n and {sum(mult)}")
            if len(X0.shape) == 1:      warnings.warn(f"Coercing X0 to shape {len(X0)} x 1"); X0 = X0.reshape(-1,1)
            if len(Z0) != X0.shape[0]: raise ValueError("Dimension mismatch between Z0 and X0")
        else:
            if len(X.shape) == 1:    warnings.warn(f"Coercing X to shape {len(X)} x 1"); X = X.reshape(-1,1)
            if X.shape[0] != len(Z): raise ValueError("Dimension mismatch between Z and X")
            elem = find_reps(X, Z, return_Zlist = False,use_torch=use_torch)
            X0   = elem['X0']
            Z0   = elem['Z0']
            Z    = elem['Z']
            mult = elem['mult']

            
        covtypes = ("Gaussian", "Matern5_2", "Matern3_2")
        if covtype not in covtypes:
            raise ValueError(f"covtype must be one of {covtypes}")

        if isinstance(lower,float) or isinstance(lower,int) or isinstance(lower,float):
            lower = np.array(lower)
        if isinstance(upper,float) or isinstance(upper,int) or isinstance(upper,float):
            upper = np.array(upper)
        if lower is None or upper is None:
            auto_thetas = auto_bounds(X = X0, covtype = covtype)
            if lower is None: lower = auto_thetas['lower']
            if upper is None: upper = auto_thetas['upper']
        lower = np.array(lower).reshape(-1)
        upper = np.array(upper).reshape(-1)
        if len(lower) != len(upper): raise ValueError("upper and lower should have the same size")

        tic = time()

        ## Initial checks
  
        n = X0.shape[0]
  
        if len(X0.shape)==1:
            raise ValueError("X0 should be a matrix. \n")
        jointThetas  = False
        constrThetas = False
        if known.get('theta_g')is not None: 
            settings['linkThetas'] = False
        if settings.get('linkThetas') is None: 
            jointThetas = True
        else:
            if settings.get('linkThetas') == 'joint':
                jointThetas = True
            if settings.get('linkThetas') == 'constr':
                constrThetas = True
    
        logN = True
        if settings.get('logN') is not None:
            logN = settings['logN']
        if settings.get('return_matrices') is None:
            settings['return_matrices'] = True
        
        if settings.get('return_hom') is None:
            settings['return_hom'] = False
        
        if jointThetas and noiseControl.get('k_theta_g_bounds') is None:
            noiseControl['k_theta_g_bounds'] = (1, 100)
        
        if settings.get('initStrategy') is None:
            settings['initStrategy'] = 'residuals'
        
        if settings.get('factr') is None:
            settings['factr']= 1e9
        
        penalty = True
        if settings.get('penalty') is not None:
            penalty = settings['penalty']
        
        if settings.get('checkHom') is None: 
            settings['checkHom'] = True
        
        trace = 0
        if settings.get('trace') is not None:
            trace = settings['trace']
        
        components = []
        if known.get("theta") is None:
            components.append('theta')
        else:
            init['theta'] = known["theta"]

        if known.get('Delta') is None: 
            components.append('Delta')
        else:
            init['Delta'] = known['Delta']
        
        if jointThetas:
            if known.get('k_theta_g') is None:
                components.append('k_theta_g')
            else:
                init['k_theta_g'] = known['k_theta_g']
        if not jointThetas and known.get('theta_g') is None:
            components.append('theta_g') 
        else:
            if not jointThetas: init['theta_g'] = known['theta_g']

        if known.get('g') is None:
            components.append('g')
        else:
            init['g'] = known['g']
        
        if init.get('pX') is not None: 
            components.append('pX')
            # replicates idcs_pX <- which(duplicated(rbind(init$pX, X0))) - nrow(init$pX) ## Indices of starting points in pX
            tmpX    = np.vstack((init['pX'],X0))
            _, i    = np.unique(tmpX,return_index = True)
            arr     = np.ones_like(tmpX,dtype=bool)
            arr[i]  = False
            idcs_pX = arr.nonzero()[0] -  - init['pX'].shape[0]
            
        else:
            if known.get('pX') is not None:
                init['pX'] = known['pX']
            else:
                init['pX'] = None
        if penalty and "pX" in components:
            penalty = False
            warnings.warn("Penalty not available with pseudo-inputs for now")

        trendtype = 'OK'
        if known.get('beta0') is not None:
            trendtype = 'SK'

        if noiseControl.get('g_bounds') is None:
            noiseControl['g_bounds'] = (1e-6, 1)
        
        if len(components)==0 and known.get('theta_g') is None:
            known['theta_g'] = known['k_theta_g'] * known['theta']

        ## Advanced option Single Nugget Kriging model for the noise process
        SiNK_eps = None
        if noiseControl.get('SiNK') is not None and noiseControl.get('SiNK'):
            SiNK = True
            if noiseControl.get('SiNK_eps') is None:
                SiNK_eps = 1e-4
            else:
                SiNK_eps = noiseControl['SiNK_eps']
        else:
            SiNK = False
        
        ### Automatic Initialisation
        modHom = modNugs = None
        if init.get("theta") is None or init.get('Delta') is None:
            ## A) homoskedastic mean process
            if known.get('g_H') is not None:
                g_init = None
            else:
                g_init = init.get('g_H')
            ## Initial value for g of the homoskedastic process: based on the mean variance at replicates compared to the variance of Z0
            if any(mult > 5):
                mean_var_replicates = (
                        (fast_tUY2(mult.T,(Z - np.repeat(Z0,mult))**2)/mult)[np.where(mult > 5)]
                    ).mean()
                if g_init is None: 
                    denom = Z0.var(ddof=1)
                    g_init = mean_var_replicates / denom
                
                if noiseControl.get('g_max') is None:
                    noiseControl['g_max'] = max(1e2, 100 * g_init)
                if noiseControl.get('g_min') is None:
                    noiseControl['g_min'] = eps
                
            else:
                if g_init is None: g_init = 0.1
                
                if noiseControl.get('g_max') is None:
                    noiseControl['g_max'] = 1e2
                
                if noiseControl.get('g_min') is None:
                    noiseControl['g_min'] = eps
                
            if settings['checkHom']:
                rKI = True #return.Ki
            else:
                rKI = False
            modHom = homGP()
            with (np.errstate(divide='ignore',invalid='ignore') 
                  if settings.get('ignore_MLE_divide_invalid',True) 
                  else contextlib.nullcontext()):
                modHom.mleHomGP(
                            X = dict(X0 = X0, Z0 = Z0, mult = mult), Z = Z, lower = lower,
                            known = dict(theta = known.get("theta"), g = known.get('g_H'), beta0 = known.get('beta0')),
                            upper = upper, init = dict(theta = init.get('theta'), g = g_init), 
                            covtype = covtype, maxit = maxit,
                            noiseControl = dict(g_bounds = (noiseControl['g_min'], noiseControl['g_max'])), eps = eps,
                            settings = dict(return_Ki = rKI))
                
            if known.get("theta") is None:
                init['theta'] = modHom['theta']
            
            if init.get('Delta') is None:
                predHom  = modHom.predict(x = X0)['mean']
                nugs_est = (Z.squeeze() - np.repeat(predHom, mult))**2 #squared deviation from the homoskedastic prediction mean to the actual observations
                nugs_est =  nugs_est / modHom['nu_hat'].squeeze()  # to be homegeneous with Delta
            
                if logN:
                    #nugs_est = np.max(nugs_est, MACHINE_DOUBLE_EPS) # to avoid problems on deterministic test functions
                    nugs_est[nugs_est<MACHINE_DOUBLE_EPS] = MACHINE_DOUBLE_EPS
                    nugs_est = np.log(nugs_est)
                    nugs_est0 = np.squeeze(fast_tUY2(mult, nugs_est))/mult # average
            
            else:
                nugs_est0 = init['Delta']
            
            if constrThetas:
                noiseControl['lowerTheta_g'] = modHom['theta']
    
            if settings['initStrategy'] == 'simple':
                if logN:
                    init['Delta'] = np.repeat(np.log(modHom['g']), X0.shape[0])
                else:
                    init['Delta'] = np.repeat(modHom['g'], X0.shape[0])
            
            if settings['initStrategy'] == 'residuals':
                init['Delta'] = nugs_est0
        
        if (init.get('theta_g') is None and init.get('k_theta_g') is None) or init.get('g') is None:
    
            ## B) Homegeneous noise process
            if jointThetas: 
                if init.get('k_theta_g') is None:
                    init['k_theta_g'] = 1
                    init['theta_g'] = init['theta']
                else:
                    init['theta_g'] = init['k_theta_g'] * init['theta']
            
            
                if noiseControl.get('lowerTheta_g') is None: 
                    noiseControl['lowerTheta_g'] = init['theta_g'] - eps
                
                
                if noiseControl.get('upperTheta_g') is None:
                    noiseControl['upperTheta_g'] = init['theta_g'] + eps
                
            
            
            if not jointThetas and init.get('theta_g') is None: 
                init['theta_g'] = init['theta']
            
            if noiseControl.get('lowerTheta_g') is None:
                noiseControl['lowerTheta_g'] = lower
            
            if noiseControl.get('upperTheta_g') is None:
                noiseControl['upperTheta_g'] = upper
            
            ## If an homegeneous process of the mean has already been computed, it is used for estimating the parameters of the noise process
            if "nugs_est" in locals():
            
                if init.get('g') is None:
                    mean_var_replicates_nugs = np.mean((fast_tUY2(mult, (nugs_est - np.repeat(nugs_est0, mult))**2)/mult))
                    if np.var(nugs_est0,ddof=1)==0: 
                        init['g'] = MACHINE_DOUBLE_EPS
                    else:
                        init['g'] = mean_var_replicates_nugs / np.var(nugs_est0,ddof=1)
                    if np.isnan(init['g']):
                        init['g'] = MACHINE_DOUBLE_EPS
                modNugs = homGP()
                settings_tmp = settings.copy()
                settings_tmp['return_Ki'] = False
                with (np.errstate(divide='ignore',invalid='ignore') 
                  if settings.get('ignore_MLE_divide_invalid',True) 
                  else contextlib.nullcontext()):
                    modNugs.mleHomGP(X = dict(X0 = X0, Z0 = nugs_est0, mult = mult), Z = nugs_est,
                                    lower = noiseControl.get('lowerTheta_g'), upper = noiseControl.get('upperTheta_g'),
                                    init = dict(theta = init.get('theta_g'), g =  init.get('g')), 
                                    known = dict(),
                                    covtype = covtype, 
                                    noiseControl = noiseControl,
                                    maxit = maxit, eps = eps, settings = settings_tmp)
                prednugs = modNugs.predict(x = X0)
            
            else:
                if "nugs_est0" not in locals(): nugs_est0 = init['Delta']
                
                if init.get('g') is None:
                    init['g'] = 0.05
                
                modNugs = homGP()
                settings_tmp = settings.copy()
                settings_tmp['return_Ki'] = False
                with (np.errstate(divide='ignore',invalid='ignore') 
                  if settings.get('ignore_MLE_divide_invalid',True) 
                  else contextlib.nullcontext()):
                    modNugs.mleHomGP(X = dict(X0 = X0, Z0 = nugs_est0, mult = np.repeat(1, X0.shape[0])), 
                                            Z = nugs_est0,
                                        lower = noiseControl.get('lowerTheta_g'), upper = noiseControl.get('upperTheta_g'),
                                        init = dict(theta = init.get('theta_g'), g =  init.get('g')), 
                                        covtype = covtype, noiseControl = noiseControl,
                                        maxit = maxit, eps = eps, settings = settings_tmp)
                prednugs = modNugs.predict(x = X0)
                
            
            if settings.get('initStrategy') == 'smoothed':
                init['Delta'] = prednugs['mean']
            
            if known.get('g') is None:
                init['g'] = modNugs['g']
            
            if jointThetas and init.get('k_theta_g') is None:
                init['k_theta_g'] = 1
            
            if not jointThetas and init.get('theta_g') is None:
                init['theta_g'] = modNugs['theta']
        if noiseControl.get('lowerTheta_g') is None:
            noiseControl['lowerTheta_g'] = lower
        
        if noiseControl.get('upperTheta_g') is None:
            noiseControl['upperTheta_g'] = upper
        ### Start of optimization of the log-likelihood
        self.max_loglik = float('-inf')
        self.arg_max = np.array([np.nan]).reshape(-1)
        if settings.get('save_iterates',False): self.iterates = []
        def fn(par, X0, Z0, Z, mult, Delta = None, theta = None, g = None, k_theta_g = None, theta_g = None, logN = False, SiNK = False,
                        beta0 = None, pX = None, hom_ll = None):
            
            idx = 0 # to store the first non used element of par
            
            if theta is None:
                idx   = idx + len(init['theta'])
                theta = par[0:idx]
                
            if Delta is None:
                Delta = par[idx:(idx + len(init['Delta']))]
                idx   = idx + len(init['Delta'])
            
            if jointThetas and k_theta_g is None:
                k_theta_g = par[idx]
                idx = idx + 1
            
            if not jointThetas and theta_g is None:
                theta_g = par[idx:(idx+ len(init['theta_g']))]
                idx = idx + len(init['theta_g'])
            if g is None:
                g = par[idx]
                idx = idx + 1
            
            if idx != (len(par)):
                pX = par[idx:len(par)].reshape(-1,X0.shape[1])
                if pX.size == 0: pX = None
            
            loglik = self.logLikHet(X0 = X0, Z0 = Z0, Z = Z, mult = mult, Delta = Delta, theta = theta, g = g, k_theta_g = k_theta_g, theta_g = theta_g,
                                logN = logN, SiNK = SiNK, beta0 = beta0, pX = pX, covtype = covtype, eps = eps, SiNK_eps = SiNK_eps, penalty = penalty,
                                hom_ll = hom_ll, trace = trace)
            
            if np.isnan(loglik) == False:
                if loglik > self.max_loglik:
                    self.max_loglik = loglik
                    self.arg_max = par
                    if settings.get('save_iterates',False): 
                        self.iterates.append({'ll':loglik,
                                              'theta':theta,'g':g,
                                              'k_theta_g':k_theta_g,'theta_g':theta_g,'Delta':Delta})
            
            return -1.0 * loglik # for maximization
        # gradient
        def gr(par, X0, Z0, Z, mult, Delta = None, theta = None, g = None, k_theta_g = None, theta_g = None, logN = False, SiNK = False,
                 beta0 = None, pX = None, hom_ll = None):
    
            idx = 0 # to store the first non used element of par
            
            if theta is None:
                theta = par[0:len(init['theta'])]
                idx   = idx + len(init['theta'])
            
            if Delta is None:
                Delta = par[idx:(idx + len(init['Delta']))]
                idx   = idx + len(init['Delta'])
            
            if jointThetas and k_theta_g is None:
                k_theta_g = par[idx]
                idx = idx + 1
            
            if not jointThetas and theta_g is None:
                theta_g = par[idx:(idx+ len(init['theta_g']))]
                idx = idx + len(init['theta_g'])
            if g is None:
                g = par[idx]
                idx = idx + 1
            if idx != (len(par)):
                pX = par[idx:len(par)].reshape(-1,X0.shape[1])
                if pX.size==0: pX = None
            
            gr_out = -1.0 * self.dlogLikHet(X0 = X0, Z0 = Z0, Z = Z, mult = mult, Delta = Delta, theta = theta, g = g, k_theta_g = k_theta_g, theta_g = theta_g,
                            logN = logN, SiNK = SiNK, beta0 = beta0, pX = pX, components = components, covtype = covtype, eps = eps, SiNK_eps = SiNK_eps,
                            penalty = penalty, hom_ll = hom_ll)
            return gr_out
        ## Pre-processing for heterogeneous fit
        parinit = lowerOpt = upperOpt = None
        
        if(trace > 2):
            print("Initial value of the parameters:\n")
        
        if known.get("theta") is None:
            parinit = init['theta']
            lowerOpt = lower
            upperOpt = upper
            if(trace > 2): print("Theta: ", init['theta'], "\n")
        
        if known.get('Delta') is None:
            if noiseControl.get('lowerDelta') is None or len(noiseControl['lowerDelta']) != n:
                if logN:
                    noiseControl['lowerDelta'] = np.log(eps)
                else:
                    noiseControl['lowerDelta'] = eps
                
            if isinstance(noiseControl['lowerDelta'],np.floating):
                noiseControl['lowerDelta'] = np.repeat(noiseControl['lowerDelta'], n)
            
            if noiseControl.get('g_max') is None: noiseControl['g_max'] = 1e2 

            if noiseControl.get('upperDelta') is None or len(noiseControl['upperDelta']) != n:
                if logN:
                    noiseControl['upperDelta'] = np.log(noiseControl['g_max'])
                else:
                    noiseControl['upperDelta'] = noiseControl['g_max']
            
            if isinstance(noiseControl['upperDelta'],np.floating):
                noiseControl['upperDelta'] = np.repeat(noiseControl['upperDelta'], n)

            ## For now, only the values at pX are kept
            ## It could be possible to fit a pseudo-input GP to all of Delta
            if "pX" in components:
                init['Delta'] = init['Delta'][idcs_pX]
                noiseControl['lowerDelta'] = noiseControl['lowerDelta'][idcs_pX]
                noiseControl['upperDelta'] = noiseControl['upperDelta'][idcs_pX]
            lowerOpt = np.append(lowerOpt, noiseControl['lowerDelta'])
            upperOpt = np.append(upperOpt, noiseControl['upperDelta'])
            parinit = np.append(parinit, init['Delta'])
            
            if(trace > 2): print("Delta: ", init['Delta'], "\n")
        
        if jointThetas and known.get('k_theta_g')is None:
            parinit = np.append(parinit, init['k_theta_g'])
            lowerOpt = np.append(lowerOpt, noiseControl['k_theta_g_bounds'][0])
            upperOpt = np.append(upperOpt, noiseControl['k_theta_g_bounds'][1])
            if(trace > 2): print("k_theta_g: ", init['k_theta_g'], "\n")
        
        if not jointThetas and known.get('theta_g') is None:
            parinit = np.append(parinit, init['theta_g'])
            lowerOpt = np.append(lowerOpt, noiseControl['lowerTheta_g'])
            upperOpt = np.append(upperOpt, noiseControl['upperTheta_g'])
            if(trace > 2): print("theta_g: ", init['theta_g'], "\n")
        
        if known.get('g') is None:
            parinit = np.append(parinit, init['g'])
            if(trace > 2): print("g: ", init['g'], "\n")
            
            lowerOpt = np.append(lowerOpt, noiseControl['g_bounds'][0])
            upperOpt = np.append(upperOpt, noiseControl['g_bounds'][1])
        
        if "pX" in components:
            parinit = np.append(parinit, init['pX'].squeeze())
            lowerOpt = np.append(lowerOpt, np.repeat(noiseControl['lowerpX'], init['pX'].shape[0]).squeeze())
            upperOpt = np.append(upperOpt, np.repeat(noiseControl['upperpX'], init['pX'].shape[0]).squeeze())
            if(trace > 2): print("pX: ", init['pX'], "\n")
        
        
        ## Case when some parameters need to be estimated
        mle_par = known.copy() # Store infered and known parameters
        
        if len(components) != 0:
            if modHom is not None:
                hom_ll = modHom['ll']
            else:
                ## Compute reference homoskedastic likelihood, with fixed theta for speed
                hom = homGP()
                with (np.errstate(divide='ignore',invalid='ignore') 
                  if settings.get('ignore_MLE_divide_invalid',True) 
                  else contextlib.nullcontext()):
                    modHom_tmp = hom.mleHomGP(X = dict(X0 = X0, Z0 = Z0, mult = mult), Z = Z, lower = lower, upper = upper,
                                        known = dict(theta = known.get("theta"), g = known.get('g_H'), beta0 = known.get('beta0')), covtype = covtype, init = init,
                                        noiseControl = dict(g_bounds = (noiseControl['g_min'], noiseControl['g_max'])), eps = eps,
                                        settings = dict(return_Ki = False))
            
                hom_ll = modHom_tmp['ll']
            parinit  = parinit[parinit!=None].astype(float)
            lowerOpt = lowerOpt[lowerOpt!=None].astype(float)
            upperOpt = upperOpt[upperOpt!=None].astype(float)
            bounds = [(l,u) for l,u in zip(lowerOpt,upperOpt)]
            
            self.arg_max = parinit.copy()
            with (np.errstate(divide='ignore',invalid='ignore') 
                  if settings.get('ignore_MLE_divide_invalid',True) 
                  else contextlib.nullcontext()):
                out = optimize.minimize(
                    fun = fn,
                    args = (X0, Z0, Z, mult, known.get('Delta'), 
                            known.get('theta'), known.get('g'), 
                            known.get('k_theta_g'),
                            known.get('theta_g'), logN, SiNK,
                            known.get('beta0'),known.get('pX'), hom_ll),
                    x0 = parinit,
                    jac = gr,
                    method="L-BFGS-B",
                    bounds = bounds,
                    # tol=1e-8,
                    options=dict(maxiter=maxit,iprint = settings.get('iprint',-1), #,
                                ftol = settings.get('factr',10) * np.finfo(float).eps,#,
                                gtol = settings.get('pgtol',0) # should map to pgtol
                                )
                )
            python_kws_2_R_kws = {
                'x':'par',
                'fun': 'value',
                'nfev': 'counts'
            }
            for key, val in python_kws_2_R_kws.items():
                out[val] = out[key]
            out['counts'] = dict(nfev=out['nfev'],njev=out['njev'])
            if out.success == False:
                out = dict(par = self.arg_max, value = -1.0 * self.max_loglik, counts = out['nfev'],
                iterates = self.iterates,
                message = out['message'])
            
            ## Temporary
            if trace > 0:
                print(out['message'])
                print("Number of variables at boundary values:" , len(np.nonzero(out['par'] == upperOpt)[0]) + len(np.nonzero(out['par'] == lowerOpt)[0]), "\n")
            if trace > 1:
                print("Name | Value | Lower bound | Upper bound \n")

            ## Post-processing
            idx = 0
            if known.get("theta") is None:
                mle_par['theta'] = out['par'][0:len(init['theta'])]
                idx = idx + len(init['theta'])
                if(trace > 1): print("Theta |", mle_par['theta'], " | ", lower, " | ", upper, "\n")
            
            if known.get('Delta') is None:
                mle_par['Delta'] = out['par'][idx:(idx + len(init['Delta']))]
                idx = idx + len(init['Delta'])
                # reconcile Deltas with R implementation
                if trace > 1:
                    for ii in range(np.ceil(len(mle_par['Delta'])/5)):
                        i_tmp = np.arange(1 + 5*(ii-1),min(5*ii, len(mle_par['Delta'])))
                        if logN: print("Delta |", mle_par['Delta'][i_tmp], " | ", np.max(np.log(eps * mult[i_tmp]), init['Delta'][i_tmp] - np.log(1000)), " | ", init['Delta'][i_tmp] + np.log(100), "\n")
                        if not logN: print("Delta |", mle_par['Delta'][i_tmp], " | ", np.max(mult[i_tmp] * eps, init['Delta'][i_tmp] / 1000), " | ", init['Delta'][i_tmp] * 100, "\n")
            if jointThetas:
                if known.get('k_theta_g') is None:
                    mle_par['k_theta_g'] = out['par'][idx]
                    idx = idx + 1
                 
                mle_par['theta_g'] = mle_par['k_theta_g'] * mle_par['theta']
                
                if trace > 1: print("k_theta_g |", mle_par['k_theta_g'], " | ", noiseControl['k_theta_g_bounds'][0], " | ", noiseControl['k_theta_g_bounds'][1], "\n")
            
            if not jointThetas and known.get('theta_g') is None:
                mle_par['theta_g'] = out['par'][idx:(idx + len(init['theta_g']))]
                idx = idx + len(init['theta_g'])
                if trace > 1: print("theta_g |", mle_par['theta_g'], " | ", noiseControl['lowerTheta_g'], " | ", noiseControl['upperTheta_g'], "\n")
            
            if known.get('g') is None:
                mle_par['g'] = out['par'][idx]
                idx = idx + 1
                if trace > 1: print("g |", mle_par['g'], " | ", noiseControl['g_bounds'][0], " | ", noiseControl['g_bounds'][1], "\n")
            
            if idx != len(out['par']):
                mle_par['pX'] = out['par'][idx:len(out['par'])].reshape(-1,X0.shape[1])
                if mle_par['pX'].size==0: mle_par['pX'] is None
                if trace > 1: print("pX |", mle_par['pX'], " | ", np.repeat(noiseControl['lowerpX'], init['pX'].shape[0]), " | ", np.repeat(noiseControl['upperpX'], init['pX'].shape[0]), "\n")
            
        else:
            
            out = dict(message = "All hyperparameters given, no optimization \n", count = 0, value = None)
        
        ## Computation of nu2
  
        if penalty:
            ll_non_pen = self.logLikHet(X0 = X0, Z0 = Z0, Z = Z, mult = mult, Delta = mle_par.get('Delta'), theta = mle_par.get('theta'), g = mle_par.get('g'), k_theta_g = mle_par.get('k_theta_g'), theta_g = mle_par.get('theta_g'),
                                    logN = logN, SiNK = SiNK, beta0 = mle_par.get('beta0'), pX = mle_par.get('pX'), covtype = covtype, eps = eps, SiNK_eps = SiNK_eps, penalty = False, hom_ll = None, trace = trace)
        else:
            ll_non_pen = out['value']

        if modHom is not None:
            if modHom['ll'] >= ll_non_pen:
                if trace >= 0: print("Homoskedastic model has higher log-likelihood: \n", modHom['ll'], " compared to ", ll_non_pen, "\n")
                if settings['checkHom']:
                    if trace >= 0: print("Return homoskedastic model \n")
                    self.__class__ = homGP
                    for key in modHom.__dict__:
                        self.__setitem__(key,modHom.__dict__[key])
                    return self
        
        if mle_par.get('pX') is None:
            Cg = cov_gen(X1 = X0, theta = mle_par['theta_g'], type = covtype)
            Kgi = np.linalg.cholesky(Cg + np.diag(eps + mle_par['g']/mult)).T
            Kgi = dtrtri(Kgi)[0]
            Kgi = Kgi @ Kgi.T
            
            nmean = np.squeeze(Kgi.sum(axis=0) @ mle_par['Delta'] / np.sum(Kgi)) ## ordinary kriging mean
            
            nu_hat_var = max(eps, np.squeeze((mle_par['Delta'] - nmean).T @ Kgi @ (mle_par['Delta'] - nmean))/len(mle_par['Delta']))
            
            if SiNK:
                rhox = 1 / rho_AN(xx = X0, X0 = mle_par['pX'], theta_g = mle_par['theta_g'], g = mle_par['g'], type = covtype, eps = eps, SiNK_eps = SiNK_eps, mult = mult)
                M =  rhox * Cg @ (Kgi @ (mle_par['Delta'] - nmean))
            else:
                M = Cg @ (Kgi @ (mle_par['Delta'] - nmean))
            
        else:
            Cg = cov_gen(X1 = mle_par['pX'], theta = mle_par['theta_g'], type = covtype) 
            Kgi = np.linalg.cholesky(Cg + np.diag(eps + mle_par['g']/mult)).T
            Kgi = dtrtri(Kgi)[0]
            Kgi = Kgi @ Kgi.T
            
            kg = cov_gen(X1 = X0, X2 = mle_par['pX'], theta = mle_par['theta_g'], type = covtype)
            
            nmean = np.squeeze(Kgi.sum(axis=0) @ mle_par['Delta'] / np.sum(Kgi)) ## ordinary kriging mean
            
            nu_hat_var = max(eps, np.squeeze((mle_par['Delta'] - nmean).T @ Kgi @ (mle_par['Delta'] - nmean))/len(mle_par['Delta']))
            
            
            if SiNK:
                rhox = 1 / rho_AN(xx = X0, X0 = mle_par['pX'], theta_g = mle_par['theta_g'], g = mle_par['g'], type = covtype, eps = eps, SiNK_eps = SiNK_eps, mult = mult)
                M =  rhox * kg @ (Kgi @ (mle_par['Delta'] - nmean))
            else:
                M = kg @ (Kgi @(mle_par['Delta'] - nmean))
        
        Lambda = np.squeeze(nmean + M)
  
        if logN:
            Lambda = np.exp(Lambda)
        
        else:
            Lambda[Lambda <= 0] = eps
        
        
        LambdaN = np.repeat(Lambda, mult)

        C = cov_gen(X1 = X0, theta = mle_par['theta'], type = covtype)
        Ki = np.linalg.cholesky(C + np.diag(Lambda/mult + eps)).T
        Ki = dtrtri(Ki)[0]
        Ki = Ki @ Ki.T
        
        if known.get('beta0') is None:
            mle_par['beta0'] = np.squeeze(Ki.sum(axis=1) @ Z0 / np.sum(Ki))
        
        psi_0 = np.squeeze((Z0 - mle_par['beta0']).T @ Ki @ (Z0 - mle_par['beta0']))
        
        #nu = (1.0 / N) * ((((Z-beta0).T @ (Z-beta0) - ((Z0-beta0)*mult).T @ (Z0-beta0)) / g_out) + psi_0)
        #nu2 <- 1/length(Z) * (crossprod((Z - mle_par$beta0)/LambdaN, Z - mle_par$beta0) - crossprod((Z0 - mle_par$beta0) * mult/Lambda, Z0 - mle_par$beta0) + psi_0)
        nu2 = (1 / len(Z)) * (((Z-mle_par['beta0'])/LambdaN).T @ (Z-mle_par['beta0']) - ((Z0 - mle_par['beta0']) * mult/Lambda).T @ (Z0 - mle_par['beta0']) + psi_0)
        
        
        # output
        self.theta = mle_par.get('theta')
        self.Delta = mle_par.get('Delta')
        self.nu_hat = nu2
        self.beta0 = mle_par.get('beta0')
        self.k_theta_g = mle_par.get('k_theta_g')
        self.theta_g = mle_par.get('theta_g')
        self.g = mle_par.get('g')
        self.nmean = nmean 
        self.Lambda = Lambda
        self.ll = -1.0 * out['value'] 
        self.ll_non_pen = ll_non_pen 
        self.nit_opt = out['counts'] 
        self.logN = logN 
        self.SiNK = SiNK 
        self.covtype = covtype
        self.pX = mle_par.get('pX') 
        self.msg = out['message']
        self.X0 = X0
        self.Z0 = Z0
        self.Z = Z 
        self.mult = mult
        self.trendtype = trendtype 
        self.eps = eps
        self.nu_hat_var = nu_hat_var
        self.used_args = dict(noiseControl = noiseControl, settings = settings, lower = lower, upper = upper, known = known)
        self.iterates = self.iterates
        self.time = time() - tic
              
        if SiNK:
            self.SiNK_eps = SiNK_eps
        if settings['return_hom']:
            self.modHom = modHom
            self.modNugs = modNugs
        
        return self
    
    def predict(self,x: ArrayLike, noise_var: bool = False, xprime: ArrayLike | None = None, nugs_only: bool = False, interval: str | None = None, interval_lower: float | None = None, interval_upper: float | None = None, **kwargs) -> dict:
        '''
        Gaussian process predictions using a heterogeneous noise GP object (of ``hetGP``) 

        Parameters
        ----------
        x : ndarray_like
            matrix of designs locations to predict at (one point per row)
        noise_var: bool (default False)
            should the variance of the latent variance process be returned?
        xprime : ndarray_like
            optional second matrix of predictive locations to obtain the predictive covariance matrix between ``x`` and ``xprime``
        nugs_only : bool (default False)  
            if ``True``, only return noise variance prediction
        kwargs : dict
            optional additional elements (not used)

        Returns
        -------

        dict with elements:
            - ``mean``: kriging mean;
            - ``sd2``: kriging variance (filtered, e.g. without the nugget values)
            - ``nugs``: noise variance prediction
            - ``sd2_var``: (returned if ``noise_var = True``) kriging variance of the noise process (i.e., on log-variances if ``logN = TRUE``)
            - ``cov``: (returned if ``xprime`` is given) predictive covariance matrix between ``x`` and ``xprime``
        
        Notes
        -------
        The full predictive variance corresponds to the sum of ``sd2`` and ``nugs``.
        See :func: `~hetgpy.hetGP.mleHetGP` for examples.
        '''
        if len(x.shape)==1:
            x = x.reshape(1,-1)
            if x.shape[1] != self['X0'].shape[1]: raise ValueError("x is not a matrix")
        
        
        if xprime is not None and len(xprime.shape)==1:
            xprime = xprime.reshape(1,-1)
            if xprime.shape[1] != self['X0'].shape[1]: raise ValueError("xprime is not a matrix")

        interval_types = [None,'confidence','predictive']
        return_interval = False
        if interval is not None:
            list_interval = [interval] if type(interval)==str else interval
            if 'confidence' not in list_interval and 'predictive' not in list_interval:
                raise ValueError(f"interval must be one of 'confidence' or 'predictive' not {interval}")
            return_interval = True
        
        if self.get('Kgi') is None:
            if self.get('pX') is None:
                Cg = cov_gen(X1 = self['X0'], theta = self['theta_g'], type = self['covtype'])
            else:
                Cg = cov_gen(X1 = self['pX'], theta = self['theta_g'], type = self['covtype'])
            
            Kgi = np.linalg.cholesky(Cg + np.diag(self['eps'] + self['g']/self['mult'])).T
            Kgi = dtrtri(Kgi)[0]
            Kgi = Kgi @ Kgi.T
            self['Kgi'] = Kgi
        
        if self.get('pX') is None:
            kg = cov_gen(X1 = x, X2 = self['X0'], theta = self['theta_g'], type = self['covtype'])
        else:
            kg = cov_gen(X1 = x, X2 = self['pX'], theta = self['theta_g'], type = self['covtype'])
        
        if self.get('Ki') is None:
            C = cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype'])
            Ki = np.linalg.cholesky(C + np.diag(self['Lambda']/self['mult'] + self['eps'])).T
            Ki = dtrtri(Ki)[0]
            Ki = Ki @ Ki.T
            self['Ki'] = Ki
        if self['SiNK']:
            M =  1/rho_AN(xx = x, X0 = self['pX'], theta_g = self['theta_g'], g = self['g'],
                        type = self['covtype'], SiNK_eps = self['SiNK_eps'], eps = self['eps'], mult = self['mult']) * kg @ (self['Kgi'] @ (self['Delta'] - self['nmean']))
        else:
            M = kg @ (self['Kgi'] @ (self['Delta'] - self['nmean']))
        
        if self['logN']:
            nugs = self['nu_hat'] * np.exp(np.squeeze(self['nmean'] + M))
        else:
            nugs = self['nu_hat'] * np.max(0, np.squeeze(self['nmean'] + M))
        if nugs.shape==(): nugs = np.array([nugs])
        
        if nugs_only:
            return dict(nugs = nugs)
        
        if noise_var:
            if self.get('nu_hat_var') is None:
                self['nu_hat_var'] = max(self['eps'], np.squeeze(((self['Delta'] - self['nmean']).T @ self['Kgi'])) @ (self['Delta'] - self['nmean'])/len(self['Delta'])) ## To avoid 0 variance
                sd2var = self['nu_hat'] * self['nu_hat_var']* np.squeeze(1 - np.diag(kg @ (self['Kgi']@ kg.T)) + (1 - ((self['Kgi'].sum(axis=0))@ kg.T))**2/sum(self['Kgi']))
        else:
            sd2var = None
        
        kx = cov_gen(X1 = x, X2 = self['X0'], theta = self['theta'], type = self['covtype'])

        if self['trendtype'] == 'SK':
            sd2 = self['nu_hat'] - np.diag(kx @ (self['Ki'] @ kx.T))
        else:
            sd2 = self['nu_hat'] * (1 - np.diag(kx @ ((self['Ki'] @ kx.T)))  + (1- (self['Ki'].sum(axis=1))@ kx.T)**2/self['Ki'].sum())
            foo=1
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
        
        preds = dict(mean = np.squeeze(self['beta0'] + kx @ (self['Ki'] @ (self['Z0'] - self['beta0']))),
              sd2 = sd2,
              nugs = nugs,
              sd2var = sd2var,
              cov = cov)
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
    
    def update(self,Xnew: ArrayLike, Znew: ArrayLike, ginit: float = 1e-2, lower: ArrayLike | None = None, upper: ArrayLike | None = None, noiseControl: dict | None = None, settings: dict | None = None,
                         known: dict = {}, maxit: int = 100, method: str = 'quick') -> None:
        r'''
        Update model object with new observations

        Parameters
        ----------
        Xnew: ndarray_like
            new inputs (one observation per row)
        Znew: ndarray_like
            new observations
        lower: ndarray_like
            lower bound for lengthscales. If not provided extracted from `self`
        upper: ndarray_like
            upper bound for lengthscales. If not provided extracted from `self`
        noiseControl: dict
            noise bounds. If not provided extracted from `self`
        settings: dict
            options for optimization. If not provided extracted from `self`
        known: dict
            known hyperparameters to fix at values
        maxit: int
            maximum number of iterations for hyperparameter optimization
        
        Returns
        -------
        self, potentially with updated hyperparameters
        

        Notes
        -----
        If hyperparameters do not need to be updated, maxit can be set to 0. In this case it is possible to pass NAs in Znew, then the model can still be used to provide updated variance predictions
        
        Example
        -------
        >>> import numpy as np, from hetgpy import hetGP
        >>> X = np.array([1, 2, 3, 4, 12]).reshape(-1,1)
        >>> Y = np.array([0, -1.75, -2, -0.5, 5])
        >>> model = hetGP().mle(X,Y,known={'theta':np.array([10.0]),'g':1e-8})
        >>> Xnew  = np.array([2.5]).reshape(-1,1)
        >>> Znew  = model.predict(Xnew)['mean']
        >>> model.update(Xnew,Znew)
        '''
        
        # first reduce Xnew/Znew in case of potential replicates
        newdata = find_reps(Xnew, Znew, normalize = False, rescale = False)
  
        # 'mixed' requires new data
        if np.isnan(Znew).any(): method = 'quick'
  
        
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
                        self.Kgi = update_Kgi_rep(id_X0, self, nrep = newdata['mult'][i])
                    
                    
                    self.mult[id_X0] = self.mult[id_X0] + newdata['mult'][i]
                    
                    ### Update Delta value depending on the selected scheme
                    
                    # if method == 'quick': nothing to be done for replicates
                    
                    # use object to use previous predictions of Lambda/mean
                    if(method == 'mixed'):
                        # model predictions
                        delta_loo = self.LOO_preds_nugs(id_X0) ## LOO mean and variance at X0[id_X0,] (only variance is used)
                        
                        # empirical estimates
                        sd2h = np.mean((self.Z[idZ[id_X0]:(idZ[id_X0] + self.mult[id_X0] - 1)] - self.predict(newdata['X0'][i:i:1,:])['mean'])**2)/self.nu_hat
                        sd2sd2h = 2*sd2h**2/self.mult[id_X0] #variance of the estimator of the variance
                        
                        if self.logN:
                            ## Add correction terms for the log transform
                            sd2h = np.log(sd2h) - digamma((self.mult[id_X0] - 1)/2) - np.log(2) + np.log(self.mult[id_X0] - 1)
                            sd2sd2h = polygamma(1,(self.mult[id_X0] - 1)/2)
                        
                    
                        delta_loo['mean'] = self.Delta[id_X0]
                        newdelta_sd2 = 1/(1/delta_loo['sd2'] + 1/sd2sd2h)
                        new_delta_mean = (delta_loo['mean']/delta_loo['sd2'] + sd2h/sd2sd2h) * newdelta_sd2 
                        newdata['Delta'][id_X0] = new_delta_mean
                    
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
        ## Now deal with new data
        if newdata['X0'].shape[0] > 0:
            if method == 'quick':
                delta_new_init = self.predict(x = newdata['X0'], nugs_only = True)['nugs']/self.nu_hat
                if self.logN: delta_new_init = np.log(delta_new_init)
            if method == 'mixed':
                delta_new_init = np.repeat(np.nan, newdata['X0'].shape[0])
                pred_deltas = self.predict(x = newdata['X0'], noise_var = True)
                
                for i in range(newdata['X0'].shape[0]):
                    sd2h = np.mean((newdata['Zlist'][[i]] - self.predict(newdata['X0'][i:i+1,:])['mean'])**2)/self.nu_hat
                    sd2sd2h = 2*sd2h**2/newdata['mult'][i] #variance of the estimator of the variance
                    
                    pdelta = pred_deltas['nugs'][i]/self.nu_hat
                    if self.logN:
                        pdelta = np.log(pdelta)
                    
                        ## Correction terms for the log transform
                        sd2h = np.log(sd2h) - digamma(newdata['mult'][i]/2) - np.log(2) + np.log(newdata['mult'][i])
                        sd2sd2h = polygamma(1,newdata['mult'][i]/2)

                    newdelta_sd2 = 1/(self.nu_hat/pred_deltas['sd2var'][i] + 1/sd2sd2h)
                    delta_new_init[i] = (self.nu_hat*pdelta/pred_deltas['sd2var'][i] + sd2h/sd2sd2h) * newdelta_sd2
            if maxit == 0:
                for i in range(newdata['X0'].shape[0]):
                    if self.logN: 
                        self.Ki = update_Ki(newdata['X0'][i:i+1,:], self, nrep = newdata['mult'][i], new_lambda = np.exp(delta_new_init[i]))
                    else:
                        self.Ki = update_Ki(newdata['X0'][i:i+1,:], self, new_lambda = delta_new_init[i], nrep = newdata['mult'][i])
                    
                    self.Kgi = update_Kgi(newdata['X0'][i:i+1,:], self, nrep = newdata['mult'][i])
                    self.X0 = np.vstack([self.X0, newdata['X0'][i:i+1,:]])
                    foo=1
                if self.logN: 
                    self.Lambda = np.hstack([self.Lambda, np.exp(delta_new_init)])
                else:
                    self.Lambda = np.hstack([self.Lambda, delta_new_init])
                
            else:
                self.X0 = np.vstack([self.X0, newdata['X0']])        
                foo=1
            self.Z0    = np.hstack([self.Z0, newdata['Z0']])
            self.mult  = np.hstack([self.mult, newdata['mult']])
            if type(newdata['Zlist'])==dict:
                self.Z     = np.hstack([self.Z, np.hstack(list(newdata['Zlist'].values()))])
            else:
                self.Z     = np.hstack([self.Z, newdata['Zlist']])
            self.Delta = np.hstack([self.Delta, delta_new_init])    
        
        if maxit == 0:
            self.nit_opt = 0
            self.msg = "Not optimized \n"
            
        else:
            
            if upper is None: upper = self.used_args['upper']
            if lower is None: lower = self.used_args['lower']
            if noiseControl is None:
                noiseControl = self.used_args['noiseControl'].copy()
                noiseControl['lowerDelta'] = None
                noiseControl['upperDelta'] = None ## must be given to noiseControl in update
             
            if settings is None: settings = self.used_args['settings'].copy()
            if known == {}: known = self.used_args['known'].copy()
            
            self.mleHetGP(X = dict(X0 = self.X0, Z0 = self.Z0, mult = self.mult), Z = self.Z,
                            noiseControl = noiseControl, lower = lower, upper = upper, covtype = self.covtype, settings = settings,
                            init = dict(theta = self.theta, theta_g = self.theta_g, k_theta_g = self.k_theta_g,
                                         Delta = self.Delta, g = np.max([self.g, ginit])),
                            known = known, eps = self.eps, maxit = maxit)
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
    def strip(self):
        r'''
        Removes larger model objects

        Can be rebuilt with `rebuild`
        
        '''
        keys  = ('Ki','Kgi','modHom','modNugs')
        for key in keys:
            if key in self.__dict__.keys():
                self.__dict__.pop(key,None)
        return self

    def rebuild(self, robust = False):
        r'''
        Rebuilds inverse covariance matrix in homGP object (usually after saving). Works in tandem with `strip`

        Parameters
        ----------

        robust: bool
            use `np.linalg.pinv` for covariance matrix inversion. Otherwise use cholesky
        
        Returns
        -------
        self with rebuilt inverse covariance matrix
        '''
        Cg = cov_gen(X1 = self['X0'],theta = self['theta_g'], type=self['covtype'])
        if robust:
            self['Ki'] = np.linalg.pinv(
                cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype']) + np.diag(self['eps'] + self['g'] / self['mult'])
            ).T
            Kgi = np.linalg.pinv(
                Cg + np.diag(self['eps'] + self['g'] / self['mult'])
            )
            self['Kgi'] = Kgi
        else:
            ki = np.linalg.cholesky(
            cov_gen(X1 = self['X0'], theta = self['theta'], type = self['covtype']) + np.diag(self['eps'] + self['Lambda'] / self['mult'])
            ).T
            ki = dtrtri(ki)[0]
            self['Ki'] = ki @ ki.T

            Kgi = np.linalg.cholesky(
                Cg + np.diag(self['eps'] + self['g'] / self['mult'])
            )
            Kgi = dtrtri(Kgi)[0]
            Kgi = Kgi @ Kgi.T
            self['Kgi'] = Kgi



        return self

    def LOO_preds_nugs(self, i):
        '''
        Leave-out-out predictions for the nugget
        
        Parameters
        ----------
        self : hetGP object
        i    : int
            index of the point to remove
        
        Returns
        -------
        LOO preds
        '''
        self.Kgi = self.Kgi - (self.Kgi.sum(axis=0).reshape(-1,1) @ self.Kgi.sum(axis=0).reshape(1,-1)) / self.Kgi.sum()
        yih = self.Delta[i] - (self.Kgi @ (self.Delta - self.nmean))[i]/self.Kgi[i,i]
        sih = 1/self.Kgi[i,i] - self.g  
    
        return(dict(mean = yih, sd2 = sih))

    
    
    
    def plot(self,type='diagnostics',**kwargs):
        r'''
        Plot model behavior. See hetgpy.plot for more details on types of plots

        Parameters
        ----------
        type: str
            type of plot: currently supported are "diagnostics" and "iterates"


        Returns
        -------
        fig, ax: figure and axis object
        '''
        ptypes = ('diagnostics','iterates')
        if type not in ptypes:
            raise ValueError(f"{type} not found, select one of {ptypes}")
        if type=='diagnostics':
            return plot_diagnostics(model=self)
        if type=='iterates':
            return plot_optimization_iterates(model=self)
    def summary(self):
        r'''
        Print summary of model object and hyperparameter optimization
        '''
        print("N = ", len(self.Z), " n = ", len(self.Z0), " d = ", self.X0.shape[1], "\n")
        print(self.covtype, " covariance lengthscale values of the main process: ", self.theta, "\n")
        print("Variance/scale hyperparameter: ", self.nu_hat, "\n")
        
        print("Summary of Lambda values: \n")
        keys = ['Min','1st Qu.', 'Median', '3rd Qu.', 'Max']
        vals = np.quantile(self.Lambda,(0,0.25,0.5,0.75,1))
        vals = ['{:.2e}'.format(v) for v in vals]
        print(dict(zip(keys,vals)))
        if self.trendtype == "SK":
            print("Given constant trend value: ", self.beta0, "\n")
        else:
            print("Estimated constant trend value: ", self.beta0, "\n")
        
        if self.logN:
            print(self.covtype, " covariance lengthscale values of the log-noise process: ", self.theta_g, "\n")
            print("Nugget of the log-noise process: ", self.g, "\n")
            print("Estimated constant trend value of the log-noise process: ", self.nmean, "\n")
        else:
            print(self.covtype, " covariance lengthscale values of the log-noise process: ", self.theta_g, "\n")
            print("Nugget of the noise process: ", self.g, "\n")
            print("Estimated constant trend value of the noise process: ", self.nmean, "\n")
        
        print("MLE optimization: \n", "Log-likelihood = ", self.ll, "; Nb of evaluations (obj, gradient) by L-BFGS-B: ", self.nit_opt, "; message: ", self.msg, "\n")
  


class hetTP():
    def __init__():
        raise NotImplementedError(f"hetTP not available yet.")
        
        
  
            
        
        

                
                
                
                
            

    
        
                
    
            
        