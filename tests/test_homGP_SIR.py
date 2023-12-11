import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy import hetGP
import matplotlib.pyplot as plt
from scipy.stats import norm

def test_homGP_SIR():
    '''
    2D testing case
    '''
    d = loadmat('tests/data/SIR.mat')

    X = d['X']
    Z = d['Y'].reshape(-1,1)

    model = hetGP.hetGP()

    # train
    res = model.mleHomGP(
        X = X,
        Z = Z,
        init = {},
        covtype="Gaussian",
        lower = np.array((0.05, 0.05)), # original paper used Matern5_2
        upper = np.array((10, 10)),
        maxit = 1e4,
        settings = {'factr': 100}
    )

    # get R data
    mat = loadmat('tests/data/homGP_SIR.mat')

    # predict
    xgrid = mat['XX']
    preds = model.predict_hom_GP(
        object = res,
        x = xgrid
    )

    # Compare to R:
    '''
    # Test 
    '''
    preds_mean_R = mat['mean']
    preds_sd_R   = mat['sd2']
    
    
    # model pars
    assert np.allclose(res['theta'],mat['theta'])
    assert np.allclose(res['g'],mat['g'])
    assert np.allclose(res['Ki'], mat['Ki'],atol=1e-3)
    
    # predictions
    assert np.allclose(preds['mean'],preds_mean_R,atol=1e-6)
    assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-6)
    
    return

if __name__ == "__main__" :
    test_homGP_SIR()