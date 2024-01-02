# test hetGPy on mcycle while holding out other parts of the fit

# test predict hetGP
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy import hetGP
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

def test_hetGP_find_Delta():

    # data
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel']

    # full fit from R
    m = loadmat('tests/data/hetGP_mcycle_fit.mat')
    
    # model
    het = hetGP.hetGP()
    Xgrid = np.linspace(0,60,301).reshape(-1,1) 

    # fix g, theta, theta_g, and k_theta_g
    known = {'g':m['g'], 'theta': m['theta'], 
             'theta_g': m['theta_g'],'k_theta_g': m['k_theta_g']}
    res = het.mleHetGP(
        X = X, 
        Z = Z,
        known = known,
        init = dict(),
        covtype = "Gaussian"
    )
    preds = het.predict_hetGP(object = res, x = Xgrid)
    assert np.allclose(preds['mean'],m['mean'])
    assert np.allclose(preds['sd2'],m['sd2'])
    print("Successfully solved for Delta!")

def test_hetGP_find_theta_and_g():
    # data
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel']

    # full fit from R
    m = loadmat('tests/data/hetGP_mcycle_fit.mat')
    
    # model
    het = hetGP.hetGP()
    Xgrid = np.linspace(0,60,301).reshape(-1,1) 

    # fix Delta, theta_g, and k_theta_g
    known = {'Delta': m['Delta'],
             'beta0': m['beta0'], 
             'theta': m['theta'],
             'theta_g': m['theta_g'],'k_theta_g': m['k_theta_g']}
    res = het.mleHetGP(
        X = X, 
        Z = Z,
        known = known,
        init = dict(),
        covtype = "Gaussian"
    )
    preds = het.predict_hetGP(object = res, x = Xgrid)
    assert np.allclose(preds['mean'],m['mean'],atol=1e-1)
    assert np.allclose(preds['sd2'],m['sd2'])
    print("Successfully solved for g and theta!")
if __name__ == "__main__":
    test_hetGP_find_Delta()
    test_hetGP_find_theta_and_g()