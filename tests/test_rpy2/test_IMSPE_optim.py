import numpy as np
from rpy2.robjects import r
from hetgpy.homGP import homGP
from hetgpy.hetGP import hetGP
from hetgpy.IMSE import crit_IMSPE, IMSPE_optim


def test_IMSPE_optim():
    r('''
     library(hetGP)
     nvar <- 2 
     
     set.seed(42)
     ftest <- function(x, coef = 0.1) return(sin(2*pi*sum(x)) + rnorm(1, sd = coef))
     
     n <- 25 # must be a square
     xgrid0 <- seq(0.1, 0.9, length.out = sqrt(n))
     designs <- as.matrix(expand.grid(xgrid0, xgrid0))
     X <- designs[rep(1:n, sample(1:10, n, replace = TRUE)),]
     Z <- apply(X, 1, ftest)
     
     model <- mleHomGP(X, Z, lower = rep(0.1, nvar), upper = rep(1, nvar))
     
     ngrid <- 51
     xgrid <- seq(0,1, length.out = ngrid)
     Xgrid <- as.matrix(expand.grid(xgrid, xgrid))
     
     preds <- predict(x = Xgrid, object =  model)
     
     ## Sequential IMSPE search
     nsteps <- 1 # Increase for better results
     for(i in 1:nsteps){
       res <- IMSPE_optim(model, control = list(multi.start = 30, maxit = 30))
       newX <- res$par
       newZ <- ftest(newX)
       model <- update(object = model, Xnew = newX, Znew = newZ)
     }
    ''')
    X, Z = np.array(r('X')), np.array(r('Z'))
    model = hetGP()
    model.mleHetGP(
        X = X,
        Z = Z,
        covtype="Gaussian",
        lower = np.array([0.1,0.1]),
        upper = np.array([1,1]),
        settings = {'checkHom':False}
    )
    nsteps = np.array(r('nsteps')).astype(int)[0]
    for i in range(nsteps):
        res = IMSPE_optim(model, control = {'mult_start':30,'maxit':30})
        assert np.allclose(res['value'],np.array(r('res$value')),atol=0.01)
if __name__ == "__main__":
    test_IMSPE_optim()