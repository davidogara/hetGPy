# test predict hetGP
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy import hetGP
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

def test_hetGP_predict():
    # read in R structures
    m = loadmat('tests/data/hetGP_mcycle_fit.mat')

    het = hetGP()
    for key in m:
        het.__dict__[key] = m[key]
    het.covtype = "Gaussian"
    
    Xgrid = np.linspace(0,60,301).reshape(-1,1)
    obj = m.copy() 
    # systematically remove objects from model
    obj['mean'] = np.nan
    obj['sd2']  = np.nan
    
    # remove covariances and re-generate
    obj['Cg']  = None
    obj['Ki']  = None
    obj['Kgi'] = None

    obj['nu_hat_var'] = None

    preds = het.predict(x = Xgrid)

    assert np.allclose(preds['mean'],m['mean'])
    assert np.allclose(preds['sd2'],m['sd2'])

if __name__ == "__main__":
    test_hetGP_predict()


