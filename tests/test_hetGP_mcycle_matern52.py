import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy import hetGP
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

def test_hetGP_mcycle():
    '''
    1D testing case
    '''
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel']

    model = hetGP.hetGP()

    # train
    res = model.mleHetGP(
        X = X,
        Z = Z,
        init = {},
        covtype="Matern5_2",
        lower = np.array([0.3902384]),
        upper = np.array([55.2]),
        settings={'factr': 10e7}
    )

    
    # predict
    xgrid =  np.linspace(0,60,301).reshape(-1,1)
    preds = model.predict_hetGP(
        object = res,
        x = xgrid
    )

    # Compare to R:
    mat = loadmat('tests/data/hetGP_mcycle_matern52.mat')
    preds_mean_R = mat['mean']
    preds_sd_R   = mat['sd2']
    
    
    return

     


if __name__ == "__main__" :
    test_hetGP_mcycle()