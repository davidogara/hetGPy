import numpy as np
import warnings
from time import time
from scipy.linalg.lapack import dtrtri
from scipy import optimize
from scipy.io import loadmat
from hetgpy.covariance_functions import cov_gen, partial_cov_gen, euclidean_dist
from hetgpy.utils import fast_tUY2, rho_AN, crossprod
from hetgpy.find_reps import find_reps
from hetgpy.auto_bounds import auto_bounds
from hetgpy.homGP import homGP
from hetgpy.plot import plot_optimization_iterates
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)


class hetGP:
    def __init__(self):
        self.iterates = [] # for saving iterates during MLE
        return
    
    # Part II: hetGP functions

    def logLikHet(self, X0, Z0, Z, mult, Delta, theta, g, k_theta_g = None, theta_g = None, logN = None, SiNK = False,
                      beta0 = None, pX = None, eps = MACHINE_DOUBLE_EPS, covtype = "Gaussian", SiNK_eps = 1e-4,
                      penalty = True, hom_ll = None, env = None, trace = 0):
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
        
        LambdaN = np.repeat(Lambda, repeats= mult.astype(int))

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
        loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi) + 1/2 * ldetKi - 1/2 * np.sum((mult - 1) * np.log(Lambda) + np.log(mult)) - N/2
        
        
        if penalty:
            nu_hat_var = np.squeeze((Delta - nmean).T @ Kgi @ (Delta - nmean))/ len(Delta)
    
            ## To avoid 0 variance, e.g., when Delta = nmean
            if nu_hat_var < eps:
                return loglik
    
            pen = - n/2 * np.log(nu_hat_var) - np.sum(np.log(np.diag(Kg_c))) - n/2*np.log(2*np.pi) - n/2
            with open('Py_loglik_iterates.csv',"a") as file:
                file.write(f"{loglik},{pen},{theta[0]},{g},{theta_g[0]},\n")
            if(loglik < hom_ll and pen > 0):
                if trace > 0: warnings.warn("Penalty is deactivated when unpenalized likelihood is lower than its homGP equivalent")
                return loglik
            return loglik + pen
        return loglik
            
        
    ## ' derivative of log-likelihood for logLikHet_Wood with respect to theta and Lambda with all observations
    ## ' Model: K = nu2 * (C + Lambda) = nu using all observations using the replicates information
    ## ' nu2 is replaced by its plugin estimator in the likelihood
    ## ' @param X0 unique designs
    ## ' @param Z0 averaged observations
    ## ' @param Z replicated observations (sorted with respect to X0)
    ## ' @param mult number of replicates at each Xi
    ## ' @param Delta vector of nuggets corresponding to each X0i or pXi, that are smoothed to give Lambda
    ## ' @param logN should exponentiated variance be used
    ## ' @param SiNK should the smoothing come from the SiNK predictor instead of the kriging one
    ## ' @param theta scale parameter for the mean process, either one value (isotropic) or a vector (anistropic)
    ## ' @param k_theta_g constant used for linking nuggets lengthscale to mean process lengthscale, i.e., theta_g[k] = k_theta_g * theta[k], alternatively theta_g can be used
    ## ' @param theta_g either one value (isotropic) or a vector (anistropic), alternative to using k_theta_g
    ## ' @param g nugget of the nugget process
    ## ' @param pX matrix of pseudo inputs locations of the noise process for Delta (could be replaced by a vector to avoid double loop)
    ## ' @param components to determine which variable are to be taken in the derivation:
    ## ' NULL for all, otherwise list with elements from 'theta', 'Delta', 'theta_g', 'k_theta_g', 'pX' and 'g'.
    ## ' @param beta0 mean, if not provided, the MLE estimator is used
    ## ' @param eps minimal value of elements of Lambda
    ## ' @param covtype covariance kernel type
    ## ' @param penalty should a penalty term on Delta be used?
    ## ' @param hom_ll: reference homoskedastic likelihood
    ## ' @export
    def dlogLikHet(self,X0, Z0, Z, mult, Delta, theta, g, k_theta_g = None, theta_g = None, beta0 = None, pX = None,
                    logN = True, SiNK = False, components = None, eps = MACHINE_DOUBLE_EPS, covtype = "Gaussian", SiNK_eps = 1e-4,
                    penalty = True, hom_ll = None, env = None):
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
        
        LambdaN = np.repeat(Lambda, mult.astype(int))

        if not (self.C is None and self.Ki is None and self.ldetKi is None):
            C = self.C
            Ki = self.Ki
            ldetKi = self.ldetKi
        else:
            C = cov_gen(X1 = X0, theta = theta, type = covtype)
            Ki = np.linalg.cholesky(C + np.diag(Lambda/mult + eps)).T
            ldetKi = - 2 * sum(np.log(np.diag(Ki))) # log determinant from Cholesky
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
                loglik = -N/2 * np.log(2*np.pi) - N/2 * np.log(psi/N)  + 1/2 * ldetKi - 1/2 * sum((mult - 1) * np.log(Lambda) + np.log(mult)) - N/2
                pen = - n/2 * np.log(nu_hat_var) - sum(np.log(np.diag(Kg_c))) - n/2*np.log(2*np.pi) - n/2
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

                    t1 = (((Z - beta0)/LambdaN) * np.repeat(dLdtk,mult.astype(int))).T @ ((Z - beta0)/LambdaN)
                    t2 = ((Z0 - beta0)/Lambda * mult * dLdtk).T @ ((Z0 - beta0)/Lambda)
                    t3 = (KiZ0.T @ dK_dthetak) @ KiZ0
                    t4 = 1/2 * np.trace(Ki @ dK_dthetak)
                    dLogL_dtheta[i] = N / 2 * (t1 - t2 + t3)/psi - t4
                    dLogL_dtheta[i] = dLogL_dtheta[i] - 1/2 * sum((mult - 1) * dLdtk/Lambda) # derivative of the sum(a_i - 1)log(lambda_i)
                    
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
            dLogL_dDelta = (M.T @ dLogLdLambda) + rSKgi/sKgi*sum(dLogLdLambda) -  rSKgi / sKgi * sum((M.T @ dLogLdLambda)) #chain rule
        
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
                dLogL_dg = ((d_irho_dg * M - rhox * M @ A0 @ Kgi) @ (Delta - nmean) - (1 - rsM) * ((Kgi @ A0 @ Kgi).sum(axis=0) @ Delta * sKgi - rSKgi @ Delta * sum(rSKgi**2/mult))/sKgi**2).T @ dLogLdLambda #chain rule
            else:
                dLogL_dg = (-M @ (KgiD/mult) - (1 - rsM) * np.squeeze(Delta @ (Kgi @ (rSKgi/mult)) * sKgi - rSKgi @ Delta * sum(rSKgi**2/mult))/sKgi**2).T @ dLogLdLambda #chain rule
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
                dLogL_dg = dLogL_dg + 1/2 * ((KgiD/mult).T @ KgiD) / nu_hat_var - sum(np.diag(Kgi)/mult)/2

        out = np.hstack([dLogL_dtheta,
           dLogL_dDelta,
           dLogL_dkthetag,
           dLogL_dthetag,
           dLogL_dg,
           dLogL_dpX]).squeeze()
        return out[~(out==None)].astype(float)
            
        
        



    def mleHetGP(self,X, Z, lower = None, upper = None, known = dict(),
                     noiseControl = dict(g_bounds = (1e-06, 1)),
                     init = {},
                     covtype = ("Gaussian", "Matern5_2", "Matern3_2"),
                     maxit = 100, eps = MACHINE_DOUBLE_EPS, settings = dict(returnKi = True, factr = 1e9)):
        
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
        
        if len(lower) != len(upper): raise ValueError("upper and lower should have the same size")

        tic = time()

        ## Initial checks
  
        n = X0.shape[0]
  
        if len(X.shape)==1:
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
            noiseControl['g_bounds'] <- (1e-6, 1)
        
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
                mean_var_replicates = np.mean(
                    (
                        (fast_tUY2(mult.T,(Z.squeeze() - np.repeat(Z0,mult.astype(int)))**2)/mult)[np.where(mult > 5)]
                    ))
                if g_init is None: g_init = mean_var_replicates/np.var(Z0,ddof=1)
                
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
            hom = homGP()
            modHom = hom.mleHomGP(
                            X = dict(X0 = X0, Z0 = Z0, mult = mult), Z = Z, lower = lower,
                            known = dict(theta = known.get("theta"), g = known.get('g_H'), beta0 = known.get('beta0')),
                            upper = upper, init = dict(theta = init.get('theta'), g = g_init), 
                            covtype = covtype, maxit = maxit,
                            noiseControl = dict(g_bounds = (noiseControl['g_min'], noiseControl['g_max'])), eps = eps,
                            settings = dict(return_Ki = rKI))
            
            if known.get("theta") is None:
                init['theta'] = modHom['theta']
            
            if init.get('Delta') is None:
                predHom  = hom.predict_hom_GP(x = X0, object = modHom)['mean']
                nugs_est = (Z.squeeze() - np.repeat(predHom, mult.astype(int)))**2 #squared deviation from the homoskedastic prediction mean to the actual observations
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
                    mean_var_replicates_nugs = np.mean((fast_tUY2(mult, (nugs_est - np.repeat(nugs_est0, mult.astype(int)))**2)/mult))
                    init['g'] = mean_var_replicates_nugs / np.var(nugs_est0,ddof=1)
                
                hom = homGP()
                settings_tmp = settings.copy()
                settings_tmp['return_Ki'] = False
                modNugs = hom.mleHomGP(X = dict(X0 = X0, Z0 = nugs_est0, mult = mult), Z = nugs_est,
                                    lower = noiseControl.get('lowerTheta_g'), upper = noiseControl.get('upperTheta_g'),
                                    init = dict(theta = init.get('theta_g'), g =  init.get('g')), 
                                    covtype = covtype, 
                                    noiseControl = noiseControl,
                                    maxit = maxit, eps = eps, settings = settings_tmp)
                prednugs = hom.predict_hom_GP(x = X0, object = modNugs)
            
            else:
                if "nugs_est0" not in locals(): nugs_est0 = init['Delta']
                
                if init.get('g') is None:
                    init['g'] = 0.05
                
                hom = homGP()
                settings_tmp = settings.copy()
                settings_tmp['return_Ki'] = False
                modNugs = hom.mleHomGP(X = dict(X0 = X0, Z0 = nugs_est0, mult = np.repeat(1, X0.shape[0])), 
                                         Z = nugs_est0,
                                    lower = noiseControl.get('lowerTheta_g'), upper = noiseControl.get('upperTheta_g'),
                                    init = dict(theta = init.get('theta_g'), g =  init.get('g')), 
                                    covtype = covtype, noiseControl = noiseControl,
                                    maxit = maxit, eps = eps, settings = settings_tmp)
                prednugs = hom.predict_hom_GP(x = X0, object = modNugs)
                
            
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
        self.arg_max = None
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
        mle_par = known # Store infered and known parameters

        if len(components) != 0:
            if modHom is not None:
                hom_ll = modHom['ll']
            else:
                ## Compute reference homoskedastic likelihood, with fixed theta for speed
                hom = homGP()
                modHom_tmp = hom.mleHomGP(X = dict(X0 = X0, Z0 = Z0, mult = mult), Z = Z, lower = lower, upper = upper,
                                    known = dict(theta = known["theta"], g = known['g_H'], beta0 = known['beta0']), covtype = covtype, init = init,
                                    noiseControl = dict(g_bounds = (noiseControl['g_min'], noiseControl['g_max'])), eps = eps,
                                    settings = dict(return_Ki = False))
            
                hom_ll = modHom_tmp['ll']
            parinit  = parinit[parinit!=None].astype(float)
            lowerOpt = lowerOpt[lowerOpt!=None].astype(float)
            upperOpt = upperOpt[upperOpt!=None].astype(float)
            bounds = [(l,u) for l,u in zip(lowerOpt,upperOpt)]
            
            
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
                    return modHom
        
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
        
        
        LambdaN = np.repeat(Lambda, mult.astype(int))

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
        
        res = dict(theta = mle_par['theta'], Delta = mle_par['Delta'], nu_hat = nu2, beta0 = mle_par['beta0'],
              k_theta_g = mle_par['k_theta_g'], theta_g = mle_par['theta_g'], g = mle_par['g'], nmean = nmean, Lambda = Lambda,
              ll = -1.0 * out['value'], ll_non_pen = ll_non_pen, nit_opt = out['counts'], logN = logN, SiNK = SiNK, covtype = covtype, pX = mle_par.get('pX'), msg = out['message'],
              X0 = X0, Z0 = Z0, Z = Z, mult = mult, trendtype = trendtype, eps = eps,
              nu_hat_var = nu_hat_var,
              used_args = dict(noiseControl = noiseControl, settings = settings, lower = lower, upper = upper, known = known),
              iterates = self.iterates,
              time = time() - tic)
        if SiNK:
            res['SiNK_eps'] = SiNK_eps
        if settings['return_hom']:
            res['modHom'] = modHom
            res['modNugs'] = modNugs
        
        return res
    
    #'Gaussian process predictions using a heterogeneous noise GP object (of class \code{hetGP}) 
    #' @param x matrix of designs locations to predict at (one point per row)
    #' @param object an object of class \code{hetGP}; e.g., as returned by \code{\link[hetGP]{mleHetGP}}
    #' @param noise.var should the variance of the latent variance process be returned?
    #' @param xprime optional second matrix of predictive locations to obtain the predictive covariance matrix between \code{x} and \code{xprime}
    #' @param nugs.only if \code{TRUE}, only return noise variance prediction
    #' @param ... no other argument for this method.
    #' @return list with elements
    #' \itemize{
    #' \item \code{mean}: kriging mean;
    #' \item \code{sd2}: kriging variance (filtered, e.g. without the nugget values)
    #' \item \code{nugs}: noise variance prediction
    #' \item \code{sd2_var}: (returned if \code{noise.var = TRUE}) kriging variance of the noise process (i.e., on log-variances if \code{logN = TRUE})
    #' \item \code{cov}: (returned if \code{xprime} is given) predictive covariance matrix between \code{x} and \code{xprime}
    #' }
    #' @details The full predictive variance corresponds to the sum of \code{sd2} and \code{nugs}.
    #' See \code{\link[hetGP]{mleHetGP}} for examples.
    #' @method predict hetGP 
    #' @export
    def predict_hetGP(self,object, x, noise_var = False, xprime = None, nugs_only = False, **kwargs):
        if len(x.shape)==1:
            x = x.reshape(1,-1)
            if x.shape[1] != object['X0'].shape[1]: raise ValueError("x is not a matrix")
        
        
        if xprime is not None and len(xprime.shape)==1:
            xprime = xprime.reshape(1,-1)
            if xprime.shape[1] != object['X0'].shape[1]: raise ValueError("xprime is not a matrix")

        if object.get('Kgi') is None:
            if object.get('pX') is None:
                Cg = cov_gen(X1 = object['X0'], theta = object['theta_g'], type = object['covtype'])
            else:
                Cg = cov_gen(X1 = object['pX'], theta = object['theta_g'], type = object['covtype'])
            
            Kgi = np.linalg.cholesky(Cg + np.diag(object['eps'] + object['g']/object['mult'])).T
            Kgi = dtrtri(Kgi)[0]
            Kgi = Kgi @ Kgi.T
            object['Kgi'] = Kgi
        
        if object.get('pX') is None:
            kg = cov_gen(X1 = x, X2 = object['X0'], theta = object['theta_g'], type = object['covtype'])
        else:
            kg = cov_gen(X1 = x, X2 = object['pX'], theta = object['theta_g'], type = object['covtype'])
        
        if object.get('Ki') is None:
            C = cov_gen(X1 = object['X0'], theta = object['theta'], type = object['covtype'])
            Ki = np.linalg.cholesky(C + np.diag(object['Lambda']/object['mult'] + object['eps'])).T
            Ki = dtrtri(Ki)[0]
            Ki = Ki @ Ki.T
            object['Ki'] = Ki
        if object['SiNK']:
            M =  1/rho_AN(xx = x, X0 = object['pX'], theta_g = object['theta_g'], g = object['g'],
                        type = object['covtype'], SiNK_eps = object['SiNK_eps'], eps = object['eps'], mult = object['mult']) * kg @ (object['Kgi'] @ (object['Delta'] - object['nmean']))
        else:
            M = kg @ (object['Kgi'] @ (object['Delta'] - object['nmean']))
        
        if object['logN']:
            nugs = object['nu_hat'] * np.exp(np.squeeze(object['nmean'] + M))
        else:
            nugs = object['nu_hat'] * np.max(0, np.squeeze(object['nmean'] + M))
        
        if nugs_only:
            return dict(nugs = nugs)
        
        if noise_var:
            if object.get('nu_hat_var') is None:
                object['nu_hat_var'] = max(object['eps'], np.squeeze(((object['Delta'] - object['nmean']).T @ object['Kgi'])) @ (object['Delta'] - object['nmean'])/len(object['Delta'])) ## To avoid 0 variance
                sd2var = object['nu_hat'] * object['nu_hat_var']* np.squeeze(1 - np.diag(kg @ (object['Kgi']@ kg.T)) + (1 - ((object['Kgi'].sum(axis=0))@ kg.T))**2/sum(object['Kgi']))
        else:
            sd2var = None
        
        kx = cov_gen(X1 = x, X2 = object['X0'], theta = object['theta'], type = object['covtype'])

        if object['trendtype'] == 'SK':
            sd2 = object['nu_hat'] - np.diag(kx @ (object['Ki'] @ kx.T))
        else:
            sd2 = object['nu_hat'] * (1 - np.diag(kx @ ((object['Ki'] @ kx.T)))  + (1- (object['Ki'].sum(axis=1))@ kx.T)**2/object['Ki'].sum())
            foo=1
        if (sd2<0).any():
            sd2[sd2<0] = 0
            warnings.warn("Numerical errors caused some negative predictive variances to be thresholded to zero. Consider using ginv via rebuild.homGP")

        if xprime is not None:
            kxprime = object['nu_hat'] * cov_gen(X1 = object['X0'], X2 = xprime, theta = object['theta'], type = object['covtype'])
            if object['trendtype'] == 'SK':
                if x.shape[0] < xprime.shape[0]:
                    cov = object['nu_hat'] *  cov_gen(X1 = object['X0'], X2 = xprime, theta = object['theta'], type = object['covtype']) - kx @ object['Ki'] @ kxprime
                else:
                    cov = object['nu_hat'] *  cov_gen(X1 = object['X0'], X2 = xprime, theta = object['theta'], type = object['covtype']) - kx @ (object['Ki'] @ kxprime)
            else:
                if x.shape[0] < xprime.shape[0]:
                    cov = object['nu_hat'] *  cov_gen(X1 = object['X0'], X2 = xprime, theta = object['theta'], type = object['covtype']) - kx @ object['Ki'] @ kxprime + ((1-(object['Ki'].sum(axis=0)).T @ kx).T @ (1-object['Ki'].sum(axis=0) @ kxprime))/object['Ki'].sum() #crossprod(1 - tcrossprod(rowSums(object$Ki), kx), 1 - rowSums(object$Ki) %*% kxprime)/sum(object$Ki)
        else:
            cov = None
        

        return dict(mean = np.squeeze(object['beta0'] + kx @ (object['Ki'] @ (object['Z0'] - object['beta0']))),
              sd2 = sd2,
              nugs = nugs,
              sd2var = sd2var,
              cov = cov)
    def plot(self,object,type='iterates',**kwargs):
        if type=='iterates':
            return plot_optimization_iterates(object,**kwargs)
        
        
        
  
            
        
        

                
                
                
                
            

    
        
                
    
            
        