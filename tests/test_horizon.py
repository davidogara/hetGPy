from hetgpy.IMSE import horizon
from hetgpy.hetGP import hetGP
from hetgpy.find_reps import find_reps
import numpy as np
from rpy2.robjects import r

def test_horizon():
    # R model
    r('''
    ## motorcycle data
    library(hetGP)
    library(MASS)
    X <- matrix(mcycle$times, ncol = 1)
    Z <- mcycle$accel
    nvar <- 1
    ## Model fitting
    data_m <- find_reps(X, Z, rescale = TRUE)
    
    model <- mleHetGP(X = list(X0 = data_m$X0, Z0 = data_m$Z0, mult = data_m$mult),
                  Z = Z, lower = rep(0.1, nvar), upper = rep(5, nvar),
                  covtype = "Matern5_2")
    ## Compute best allocation                  
    A <- allocate_mult(model, N = 1000)
    ''')
    model = hetGP()
    X, Z = np.array(r('X')), np.array(r('Z'))
    data_m = find_reps(X, Z, rescale=True)
    model.mleHetGP(X = {'X0':data_m['X0'], 'Z0' : data_m['Z0'], 'mult': data_m['mult']},
                   Z = Z,
                   lower = np.array([0.1]),
                   upper = np.array([5.0]),
                   covtype = "Matern5_2",
                   maxit=1000)
    # align models for comparison
    align_models = True
    if align_models:
        model.theta  = np.array(r('model$theta'))
        model.g      = np.array(r('model$g'))
        model.Ki     = np.array(r('model$Ki'))
        model.Delta  = np.array(r('model$Delta'))
        model.Lambda = np.array(r('model$Lambda'))
    h = horizon(model,current_horizon = 1, previous_ratio=0.5,target=0.75)
    rh = np.array(r('horizon(model,current_horizon = 1, previous_ratio=0.5,target=0.75)'))
    assert h == rh
if __name__ == "__main__":
    test_horizon()