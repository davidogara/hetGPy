import numpy as np
from hetgpy import hetGP, homGP

def LOO_preds(model, ids = None):
    '''
    #' Provide leave one out predictions, e.g., for model testing and diagnostics. 
    #' This is used in the method plot available on GP and TP models.
    #' @title Leave one out predictions
    #' @param model \code{homGP} or \code{hetGP} model, TP version is not considered at this point
    #' @param ids vector of indices of the unique design point considered (default to all)
    #' @return list with mean and variance predictions at x_i assuming this point has not been evaluated
    #' @export
    #' @note For TP models, \code{psi} is considered fixed.
    #' @references
    #' O. Dubrule (1983), Cross validation of Kriging in a unique neighborhood, Mathematical Geology 15, 687--699. \cr \cr
    #' 
    #' F. Bachoc (2013), Cross Validation and Maximum Likelihood estimations of hyper-parameters of Gaussian processes 
    #' with model misspecification, Computational Statistics & Data Analysis, 55--69.
    #' 
    #' @examples
    #' set.seed(32)
    #' ## motorcycle data
    #' library(MASS)
    #' X <- matrix(mcycle$times, ncol = 1)
    #' Z <- mcycle$accel
    #' nvar <- 1
    ## ' plot(X, Z, ylim = c(-160, 90), ylab = 'acceleration', xlab = "time")
    #'
    #' ## Model fitting
    #' model <- mleHomGP(X = X, Z = Z, lower = rep(0.1, nvar), upper = rep(10, nvar),
    #'                   covtype = "Matern5_2", known = list(beta0 = 0))
    #' LOO_p <- LOO_preds(model)
    #'  
    #' # model minus observation(s) at x_i
    #' d_mot <- find_reps(X, Z)
    #' 
    #' LOO_ref <- matrix(NA, nrow(d_mot$X0), 2)
    #' for(i in 1:nrow(d_mot$X0)){
    #'  model_i <- mleHomGP(X = list(X0 = d_mot$X0[-i,, drop = FALSE], Z0 = d_mot$Z0[-i],
    #'                      mult = d_mot$mult[-i]), Z = unlist(d_mot$Zlist[-i]),
    #'                      lower = rep(0.1, nvar), upper = rep(50, nvar), covtype = "Matern5_2",
    #'                      known = list(theta = model$theta, k_theta_g = model$k_theta_g, g = model$g,
    #'                                   beta0 = 0))
    #'  model_i$nu_hat <- model$nu_hat
    ## ' # For hetGP, need to use the same Lambdas to get the same results  
    ## '  model_i$Lambda <- model$Lambda[-i] 
    ## '  model_i <- strip(model_i)
    ## '  model_i <- rebuild(model_i)
    #'  p_i <- predict(model_i, d_mot$X0[i,,drop = FALSE])
    #'  LOO_ref[i,] <- c(p_i$mean, p_i$sd2)
    #' }
    #' 
    #' # Compare results
    #' 
    #' range(LOO_ref[,1] - LOO_p$mean)
    #' range(LOO_ref[,2] - LOO_p$sd2)
    #' 
    #' # Use of LOO for diagnostics
    #' plot(model)
    '''
    if ids is None: ids = np.arange(model.X0.shape[0])
    
    if model.trendtype is not None and model.trendtype == "OK":
        model.Ki = model.Ki - model.Ki.sum(axis=0).reshape(-1,1) @ model.Ki.sum(axis=0).reshape(1,-1) / model.Ki.sum()
    
    if model.__class__.__name__ == 'homGP':
        sds = model.nu_hat * (1/np.diag(model.Ki)[ids] - model.g/model.mult[ids])
    
    if model.__class__.__name__ == 'hetGP':
        sds = model.nu_hat * (1/np.diag(model.Ki)[ids] - model.Lambda[ids]/model.mult[ids])
    
    
    if model.__class__.__name__ == 'homTP':
        sds = (1/np.diag(model.Ki)[ids] - model.g/model.mult[ids])
        # TP correction
        sds = (model.nu + model.psi - 2) / (model.nu + len(model.Z) - model.mult[ids] - 2) * sds


    if model.__class__.__name__ == 'hetTP':
        sds = (1/np.diag(model.Ki)[ids] - model.Lambda[ids]/model.mult[ids])
        # TP correction
        sds = (model.nu + model.psi - 2) / (model.nu + len(model.Z) - model.mult[ids] - 2) * sds
    
    
    ys = model.Z0[ids] - (model.Ki @ (model.Z0 - model.beta0))[ids]/np.diag(model.Ki)[ids]
    return(dict(mean = ys, sd2 = sds))
