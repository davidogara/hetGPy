import numpy as np
from rpy2.robjects import r
from hetgpy.hetGP import hetGP
from copy import copy

def noise_fun(x,coef=1):
    return coef * (0.05 + np.sqrt(np.abs(x)*20/(2*np.pi))/10)
def test_update():
    r('''
    library(hetGP)
    set.seed(42)
    ## Spatially varying noise function
    noisefun <- function(x, coef = 1){
    return(coef * (0.05 + sqrt(abs(x)*20/(2*pi))/10))
    }

    ## Initial data set
    nvar <- 1
    n <- 20
    X <- matrix(seq(0, 2 * pi, length=n), ncol = 1)
    mult <- sample(1:10, n, replace = TRUE)
    X <- rep(X, mult)
    Z <- sin(X) + rnorm(length(X), sd = noisefun(X))

    ## Initial fit
    testpts <- matrix(seq(0, 2*pi, length = 10*n), ncol = 1)
    model <- model_init <- mleHetGP(X = X, Z = Z, lower = rep(0.1, nvar), 
    upper = rep(5, nvar), maxit = 1000)
    nsteps <- 5
    npersteps <- 10
    ''')
    X = np.array(r('X')).reshape(-1,1)
    Z = np.array(r('Z'))
    n = 20
    testpts = np.array(r('testpts'))
    model = hetGP()
    model.mleHetGP(
        X = X,
        Z = Z,
        lower = 0.1 + 0.0*np.arange(X.shape[1]),
        upper = 5 + 0.0*np.arange(X.shape[1]),
        maxit = 500
    )
    model_init = copy(model)
    for i in range(np.array(r('nsteps')).astype(int)[0]):
        print('Running step',i)
        r('''
        newIds <- sort(sample(1:(10*n), npersteps,replace=F))
        newmult <- sample(1:5,length(newIds), replace = T)
        newIds <- rep(newIds,newmult)
        newX <- testpts[newIds, ,drop = FALSE] 
        newZ <- sin(newX) + rnorm(length(newX), sd = noisefun(newX))
        model <- update(object = model, Xnew = newX, Znew = newZ)
        X <- c(X, newX)
        Z <- c(Z, newZ)
        ''')
        newX = np.array(r('newX')).reshape(-1,1)
        newZ = np.array(r('newZ'))
        model.update(Xnew = newX, Znew = newZ.squeeze(),maxit=500,lower = np.array([0.1]), upper = np.array([5]))
        X = np.vstack([X,newX])        
        Z = np.hstack([Z,newZ.squeeze()])
    final_preds = model.predict(testpts)

if __name__ == "__main__":
    test_update()