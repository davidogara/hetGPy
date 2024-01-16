import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy.homGP import homGP
import matplotlib.pyplot as plt
from scipy.stats import norm
# R objects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
hetGP_R = importr('hetGP')
np_cv_rules = default_converter + numpy2ri.converter

def test_homGP_SIR():
    '''
    2D testing case
    '''
    d = loadmat('tests/data/SIR.mat')

    X = d['X']
    Z = d['Y'].reshape(-1,1)

    model = homGP()

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
    xseq = np.linspace(0,1,100)
    xgrid = np.array([(y,x) for x in xseq for y in xseq])
    preds = model.predict_hom_GP(
        object = res,
        x = xgrid
    )

    # Compare to R:
    '''
    # Test 
    '''
    with np_cv_rules.context():
        objR = hetGP_R.mleHomGP(X = X, 
                        Z = Z, 
                        covtype = "Gaussian",
                        lower   = np.array((0.05, 0.05)), # original paper used Matern5_2
                        upper   = np.array((10, 10)),
                        maxit   = 1e4)
        

    preds_mean_R = mat['mean']
    preds_sd_R   = mat['sd2']
    
    
    # model pars
    assert np.allclose(res['theta'],objR['theta'])
    assert np.allclose(res['g'],objR['g'])
    assert np.allclose(res['Ki'], objR['Ki'],atol=1e-3)
    
    # predictions
    assert np.allclose(preds['mean'],preds_mean_R,atol=1e-6)
    assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-6)
    
    return

if __name__ == "__main__" :
    test_homGP_SIR()