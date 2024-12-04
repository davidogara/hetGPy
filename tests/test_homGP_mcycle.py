from scipy.io import loadmat
import numpy as np
from hetgpy.homGP import homGP
from tests.utils import read_yaml

def test_homGP_mcycle():
    '''
    1D testing case
    '''
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel'].reshape(-1,1)

    model = homGP()

    # train
    model.mleHomGP(
        X = X,
        Z = Z,
        init = {},
        covtype="Gaussian",
        settings={'factr': 10} # high quality solution
    )

    
    # predict
    xgrid =  np.linspace(0,60,301).reshape(-1,1)
    preds = model.predict(
        x = xgrid
    )

    # Compare to R:
    R_compare    = read_yaml('tests/R/results/test_homGP_mcycle.yaml')
    preds_mean_R = R_compare['mean']
    preds_sd_R   = R_compare['sd2']
    
    
    # model pars
    assert np.allclose(model['theta'],R_compare['theta'])
    assert np.allclose(model['g'],R_compare['g'])
    
    # predictions
    assert np.allclose(preds['mean'],preds_mean_R,atol=1e-6)
    assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-6)
    
    return


if __name__ == "__main__" :
    test_homGP_mcycle()