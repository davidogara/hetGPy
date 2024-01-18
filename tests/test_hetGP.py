from scipy.io import loadmat
from scipy.stats import norm
import numpy as np
from hetgpy.hetGP import hetGP
import pytest
from rpy2.robjects import r
import gc
from time import time

@pytest.fixture()
def mcycle():
     '''
    1D testing case
    '''
     d = loadmat('tests/data/mcycle.mat')
     X = d['times'].reshape(-1,1)
     Z = d['accel']
     return X, Z

@pytest.fixture()
def SIR():
    d = loadmat('tests/data/SIR.mat')
    X = d['X']
    Z = d['Y']
    return X, Z

def RString_mcycle(covtype = "Gaussian"):
     '''Helper function to format RString'''
     RStr = '''
        library(MASS)
        library(hetGP)
        X = as.matrix(mcycle$times)
        Z = mcycle$accel
        xgrid = as.matrix(seq(0,60,length.out = 301))
        model = mleHetGP(X = X, Z = Z, 
                    covtype = "{}",
                    lower = 1, upper = 100,maxit=2e2,
                    settings = list(factr=10e7))
        preds = predict(model,xgrid)
        # predictive interval
        preds$upper = qnorm(0.95, preds$mean, sqrt(preds$sd2 + preds$nugs)) 
        '''.format(covtype)
     return RStr

def RString_SIR(covtype = "Matern5_2"):
     '''Helper function to format RString'''
     RStr = '''
        library(R.matlab)
        library(hetGP)
        m = readMat("tests/data/SIR.mat")
        X = m[["X"]]
        Z = m[["Y"]]
        xseq  = seq(0,1,length.out = 100)
        xgrid = as.matrix(expand.grid(xseq,xseq))
        model = mleHetGP(
                X = X,
                Z = Z,
                covtype = "{}",
                lower = c(0.05,0.05),
                upper = c(2,2),
                maxit = 1e4
            )
        preds = predict(model,xgrid)
        # predictive interval
        preds$upper = qnorm(0.95, preds$mean, sqrt(preds$sd2 + preds$nugs)) 
        '''.format(covtype)
     return RStr

def compute_mcycle(X,Z,covtype, verbose = False):
    model = hetGP()

    # train
    tic = time()
    model.mleHetGP(
        X = X,
        Z = Z,
        init = {},
        known = {},
        maxit=1e3,
        lower = np.array([1]),
        upper = np.array([100]),
        covtype=covtype,
        settings = {'factr': 10e7}
    )
    toc = time()
    foo=1

    
    # predict
    xgrid =  np.linspace(0,60,301).reshape(-1,1)
    preds = model.predict(
        x = xgrid
    )
    preds['upper'] =  norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

    # Run R experiment
    RStr = RString_mcycle(covtype=covtype)
    r(RStr)
    py_preds = dict(mean=preds['mean'],upper=preds['upper'],theta=model['theta'])
    r_preds  = dict(mean=np.array(r('preds$mean')),upper=np.array(r('preds$upper')),theta=np.array(r('model$theta')))
    
    if verbose:
        print(f"Python ran in: {round(model['time'],3)} seconds")
        print(f"R ran in: {round(np.array(r('model$time'))[0],3)} seconds")
        print(f"Max mean diff {round(np.abs(py_preds['mean']-r_preds['mean']).max(),3)}")
        print(f"Max upper diff {round(np.abs(py_preds['upper']-r_preds['upper']).max(),3)}")
    model = None
    return(py_preds,r_preds)
def test_hetGP_mcycle_Gaussian(mcycle,verbose=False):
    X, Z = mcycle
    py_preds, r_preds = compute_mcycle(X,Z,covtype="Gaussian",verbose=verbose)
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=5)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=10)
    gc.collect()
    

def test_hetGP_mcycle_Matern5_2(mcycle,verbose=False):
    X, Z = mcycle
    py_preds, r_preds = compute_mcycle(X,Z,covtype="Matern5_2",verbose=verbose)
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=5)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=15)
    gc.collect()

def compute_SIR(X,Z,covtype, verbose=False):
    xseq = np.linspace(0,1,100)
    xgrid = np.array([(y,x) for x in xseq for y in xseq])

    model = hetGP()

    model.mleHetGP(X = X, 
                   Z = Z, 
                   covtype=covtype,
                   known= {},
                   init = {},
                   lower = np.array([0.5,0.5]),
                   upper = np.array([2,2]),
                   maxit=5e3,
                   settings={'factr':10e7}
                   )
    preds = model.predict(x=xgrid)
    preds['upper'] =  norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

    RStr = RString_SIR(covtype=covtype)
    r(RStr)
    py_preds = dict(mean=preds['mean'],upper=preds['upper'],theta=model['theta'])
    r_preds  = dict(mean=np.array(r('preds$mean')),upper=np.array(r('preds$upper')),theta=np.array(r('model$theta')))
    
    if verbose:
        print(f"Python ran in: {round(model['time'],3)} seconds")
        print(f"R ran in: {round(np.array(r('model$time'))[0],3)} seconds")
        print(f"Max mean diff {round(np.abs(py_preds['mean']-r_preds['mean']).max(),3)}")
        print(f"Max upper diff {round(np.abs(py_preds['upper']-r_preds['upper']).max(),3)}")
    model = None
    return py_preds, r_preds

def test_hetGP_SIR_Matern5_2(SIR,verbose=False):
    X, Y = SIR
    py_preds, r_preds = compute_SIR(X,Y,covtype="Matern5_2",verbose=True)
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=0.025)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=0.025)
    gc.collect()
    

if __name__ == "__main__" :
    #m = loadmat('tests/data/mcycle.mat')
    #X = m['times'].reshape(-1,1)
    #Z = m['accel']
    #test_hetGP_mcycle_Matern5_2((X,Z),verbose=True)
    m = loadmat('tests/data/SIR.mat')
    X = m['X']
    Z = m['Y']
    test_hetGP_SIR_Matern5_2((X,Z),verbose=True)

