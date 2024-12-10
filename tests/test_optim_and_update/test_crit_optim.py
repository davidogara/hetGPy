from rpy2.robjects import r
import numpy as np
from hetgpy import homGP
from hetgpy.optim import crit_optim
def test():
    r('''
    library(hetGP)
    ftest <- function(x, coef = 0.1) return(sin(2*pi*sum(x)) + rnorm(1, sd = coef))
    
    n <- 25 # must be a square
    xgrid0 <- seq(0.1, 0.9, length.out = sqrt(n))
    designs <- as.matrix(expand.grid(xgrid0, xgrid0))
    X <- designs[rep(1:n, sample(1:10, n, replace = TRUE)),]
    Z <- apply(X, 1, ftest)
    nvar = ncol(X)
    model <- mleHomGP(X, Z, lower = rep(0.1, nvar), upper = rep(1, nvar))
    
    ngrid <- 51
    xgrid <- seq(0,1, length.out = ngrid)
    Xgrid <- as.matrix(expand.grid(xgrid, xgrid))
    
    nsteps <- 2 # Increase for more steps
    crit <- "crit_EI" 
    ''')
    rand = np.random.default_rng(seed=42)
    ftest = lambda x: np.sin(2*np.pi*np.sum(x)) + rand.normal(size = 1, scale = 0.1)
    X, Z, nsteps = np.array(r('X')), np.array(r('Z')).squeeze(), np.array(r('nsteps'))[0].astype(int)
    crit = "crit_EI"
    model = homGP()
    model.mleHomGP(X = X, Z = Z, lower = np.array([0.1,0.1]), upper = np.array([1,1]))
    for i in range(nsteps):
        r('''
        res <- crit_optim(model, h = 3, crit = crit, ncores = 1,
                            control = list(multi.start = 100, maxit = 50))
        # If a replicate is selected
        if(!res$path[[1]]$new) print("Add replicate")
        newX <- res$par
        newZ <- ftest(newX)
        model <- update(object = model, Xnew = newX, Znew = newZ)
        ''')
        res = crit_optim(model, h = 3, crit = crit, 
                         control =dict(multi_start = 100, maxit = 50), ncores = 1)
        assert np.allclose(res['value'],np.array(r('res$value')),atol=0.01)
        if not res['path'][0]['new']:
            print("Add replicate")
        newX = res['par']
        newZ = ftest(newX)
        model.update(Xnew=newX,Znew=newZ)
if __name__ == "__main__":
    test()
