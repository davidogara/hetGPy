
# function to handle the updating of the covariance matrices
import hetgpy
from hetgpy.covariance_functions import cov_gen
import numpy as np

def update_Ki(x, model, new_lambda = None, nrep = 1):
    r'''
    ## ' Compute the inverse covariance matrix when adding a new design
    ## ' @param x matrix for the new design
    ## ' @param model \code{homGP} or \code{hetGP} model
    ## ' @param new_delta optional vector. In case of a \code{hetGP} model, value of Delta at \code{x}. 
    ## ' If not provided, it is taken as the prediction of the latent GP. 
    ## ' For \code{homGP} models, it corresponds to \code{g}. 
    ## ' @details see e.g., Bobby's lecture on design, for the partition inverse equations
    ## ' @export
    ## ' @examples 
    ## ' \dontrun{
    ## ' ## Validation
    ## ' ## 1) run example("mleHetGP", ask = FALSE)
    ## ' 
    ## ' nvar <- 2
    ## ' ntest <- 10
    ## ' design <- matrix(runif(ntest * nvar), ntest, nvar)
    ## ' response <- sum(sin(2*pi*design)) + rnorm(ntest)
    ## ' 
    ## ' model <- mleHetGP(X = design, Z = response, upper = rep(1, nvar), lower = rep(0.01,nvar))
    ## ' 
    ## ' nrep <- 4
    ## ' xnew <- matrix(runif(nvar), 1)
    ## ' Kni <- update_Ki(xnew, model, nrep = nrep)
    ## ' 
    ## ' Lambdan <- c(model.Lambda, predict(model, x = xnew).nugs/model.nu_hat)
    ## ' multn <- c(model.mult, nrep)
    ## ' Kn <- cov_gen(rbind(model.X0, xnew), theta = model.theta, type = model.covtype) + diag(model.eps + Lambdan/multn)
    ## ' Kni_ref <- chol2inv(chol(Kn))
    ## ' print(max(abs(Kni %*% Kn - diag(nrow(model.X0) + 1))))                                 
    ## ' }
    '''

    if type(model)==hetgpy.homTP or type(model)==hetgpy.hetTP:
        kn1 = model.sigma2 * cov_gen(x, model.X0, theta = model.theta, type = model.covtype)
        if new_lambda is None:
            new_lambda = model.predict(object = model, x = x, nugs_only = True)['nugs']
        vn = (model.sigma2 - kn1 @ (model.Ki @ kn1.T)).squeeze() + new_lambda/nrep + model.eps
    else:
        kn1 = cov_gen(x, model.X0, theta = model.theta, type = model.covtype)
        if new_lambda is None:
            new_lambda = model.predict(x = x, nugs_only = True)['nugs']/model.nu_hat
        vn = (1 - kn1 @ (model.Ki @ kn1.T)).squeeze() + new_lambda/nrep + model.eps
        if vn.shape==(): vn = np.array(vn)

    gn = - (model.Ki @ kn1.T) / vn
    Ki = model.Ki + (gn @ gn.T) * vn

    return np.vstack([np.hstack([Ki, gn]), np.vstack((gn, 1/vn)).T])

## Same as update_Ki but for Kgi
def update_Kgi(x, model, nrep = 1):
    kn1 = cov_gen(x, model.X0, theta = model.theta_g, type = model.covtype)
    nugsn = model.g/nrep
    vn = (1 - kn1 @ (model.Kgi @ kn1.T)).squeeze() + nugsn/nrep + model.eps
    if vn.shape==(): vn = np.array(vn)
    gn = - (model.Kgi @ kn1.T) / vn
    Kgi = model.Kgi + (gn @ gn.T) * vn

    return np.vstack([np.hstack([Kgi, gn]), np.vstack((gn, 1/vn)).T])

## ## Verification
## ## 1) run example("mleHetGP", ask = FALSE)
## 
## Kni <- hetGP:::update_Ki_rep(16, model)
## multn <- model.mult
## multn[16] <- multn[16] + 1
## Kn <- cov_gen(model.X0, theta = model.theta, type = model.covtype) + diag(model.eps + model.Lambda/multn)
## Kni_ref <- chol2inv(chol(Kn))
## print(max(abs(Kni %*% Kn - diag(nrow(model.X0))))) 
##
## update of Ki in case model.X0[id,] is replicated nrep times
def update_Ki_rep(id, model, nrep = 1):
  if model.get('Lambda') is None:
    tmp = model.g
  else:
    tmp = model.Lambda[id]
  
  B = (model.Ki[id,:].T @ model.Ki[id,:]) / ((model.mult[id]*(model.mult[id] + nrep)) / (nrep*tmp) - model.Ki[id, id])
  Ki = model.Ki + B
  return Ki


## update of Kgi in case model.X0[id,] is replicated nrep times
def update_Kgi_rep(id, model, nrep = 1):
  tmp = model.g
  B = (model.Kgi[id,:].T @ model.Kgi[id,:]) / ((model.mult[id]*(model.mult[id] + nrep)) / (nrep*tmp) - model.Kgi[id, id])
  Kgi = model.Kgi + B
  return Kgi
