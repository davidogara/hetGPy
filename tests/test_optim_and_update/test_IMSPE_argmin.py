#test_IMSPE.py
import numpy as np
from rpy2.robjects import r

from hetgpy.IMSE import Wij, crit_IMSPE
from hetgpy import hetGP
from time import time

def test_IMSPE():
    '''
    From the ?hetGP::crit_IMSPE (1D example)

    Tests whether given the same inputs, does crit_IMSPE return the same argmin?

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
    
    Wijs = Wij(mu1 = model.X0, theta = model.theta, type = model.covtype)
    t0 = time()
    IMSPE_grid = np.array([crit_IMSPE(x,model,Wijs=Wijs) for x in xgrid]).squeeze()
    t1 = time()
    arr = np.array(r('IMSPE_grid'))

    argmin_py = IMSPE_grid.argmin()
    argmin_R = arr.argmin()
    assert np.allclose(argmin_py,argmin_R)

    return

if __name__ == "__main__":
    test_IMSPE()