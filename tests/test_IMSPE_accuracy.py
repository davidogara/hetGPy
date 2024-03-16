#test_IMSPE.py
import numpy as np
from rpy2.robjects import r

from hetgpy.IMSE import IMSPE, Wij, crit_IMSPE
from hetgpy.hetGP import hetGP

def test_IMSPE():
    '''
    From the ?hetGP::crit_IMSPE (1D example)

    Tests whether given *the same* inputs, does crit_IMSPE give the same outputs?
    See test_IMSPE_argmin to check whether crit_IMSPE gives the same argmin

    '''
    r('''
    library(hetGP)
    set.seed(42)
    ftest <- function(x, coef = 0.1) return(sin(2*pi*x) + rnorm(1, sd = coef))

    n <- 9
    designs <- matrix(seq(0.1, 0.9, length.out = n), ncol = 1)
    X <- matrix(designs[rep(1:n, sample(1:10, n, replace = TRUE)),])
    Z <- apply(X, 1, ftest)

    prdata <- find_reps(X, Z, inputBounds = matrix(c(0,1), nrow = 2, ncol = 1))
    Z <- prdata$Z
    
    model <- mleHetGP(X = list(X0 = prdata$X0, Z0 = prdata$Z0, mult = prdata$mult),
                  Z = Z, lower = 0.1, upper = 5)

    ngrid <- 501
    xgrid <- matrix(seq(0,1, length.out = ngrid), ncol = 1)

    ## Precalculations
    Wijs <- Wij(mu1 = model$X0, theta = model$theta, type = model$covtype)


    t0 <- Sys.time()

    IMSPE_grid <- apply(xgrid, 1, crit_IMSPE, Wijs = Wijs, model = model)

    t1 <- Sys.time()
    ''')

    X = np.array(r('X'))
    Z = np.array(r('Z'))
    xgrid = np.array(r('xgrid'))
    model = hetGP()
    model.mleHetGP(X = X, Z = Z, lower = np.array([0.1]), upper = np.array([5]))
    
    # since we are concerned with testing the accuracy of IMSPE, we take the model attributes from R

    # mean terms
    model.theta = np.array(r('model$theta'))
    model.beta0 = np.array(r('model$beta0'))
    model.Ki = np.array(r('model$Ki'))

    # noise terms
    model.g = np.array(r('model$g'))
    model.Delta = np.array(r('model$Delta'))
    model.nmean = np.array(r('model$nmean'))
    model.Lambda = np.array(r('model$Lambda'))
    model.Kgi = np.array(r('model$Kgi'))
    model.nu_hat = np.array(r('model$nu_hat'))
    model.theta_g = np.array(r('model$theta_g'))
    model.k_theta_g = np.array(r('model$k_theta_g'))
    Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)

    IMSPE_grid = np.array([crit_IMSPE(x,model,Wijs=Wijs) for x in xgrid]).squeeze()
    arr = np.array(r('IMSPE_grid'))
    assert np.allclose(IMSPE_grid,arr)

    return

if __name__ == "__main__":
    test_IMSPE()