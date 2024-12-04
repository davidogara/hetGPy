from scipy.io import loadmat
import numpy as np
from hetgpy.homGP import homGP
from tests.utils import read_yaml


def test_homGP_SIR():
    '''
    2D testing case
    '''
    d = loadmat('tests/data/SIR.mat')

    X = d['X']
    Z = d['Y'].reshape(-1,1)

    model = homGP()

    # train
    model.mleHomGP(
        X = X,
        Z = Z,
        init = {},
        covtype="Matern5_2",
        lower = np.array((0.05, 0.05)), # original paper used Matern5_2
        upper = np.array((10, 10)),
        maxit = 1e4,
        settings = {'factr': 100}
    )

    # get R data
    mat = loadmat('tests/data/homGP_SIR.mat')

    # predict
    xseq = np.linspace(0,1,10)
    xgrid = np.array([(y,x) for x in xseq for y in xseq])
    preds = model.predict(
        x = xgrid
    )

    # Compare to R:
    R_compare = read_yaml('tests/R/results/test_homGP_SIR.yaml')
    preds_mean_R = R_compare['mean']
    preds_sd_R   = R_compare['sd2']
    
    
    # model pars
    assert np.allclose(model['theta'],R_compare['theta'],atol=1e-2)
    assert np.allclose(model['g'],R_compare['g'],atol=1e-3)
    
    # predictions
    assert np.allclose(preds['mean'],preds_mean_R,atol=1e-2)
    assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-2)
    
    return

if __name__ == "__main__" :
    test_homGP_SIR()