from hetgpy import hetGP
import numpy as np
import pandas as pd
from scipy.io import loadmat
import gc

def test_homGP_mcycle():
    '''
    1D testing case
    '''
    d = pd.read_csv('tests/data/mcycle.csv')
    X = d['times'].values.reshape(-1,1)
    Z = d['accel'].values.reshape(-1,1)

    model = hetGP.hetGP()

    res = model.mleHomGP(
        X = X,
        Z = Z,
        covtype="Gaussian",
        lower = np.array([0.1]),
        upper = np.array([10]),
        init={}
    )

    # get R values
    theta_R = 10
    g_R = 0.5009635

    cond1 = np.allclose(theta_R,res['theta'])
    cond2 = np.allclose(res['g'],g_R)
    assert cond1 and cond2
    return

def test_mleHomGP_SIR():
    '''
    2D testing case
    '''
    d = loadmat('tests/sirEval/SIR.mat')
    X = d['X']
    Y = d['Y'].reshape(-1,1)

    model = hetGP.hetGP()

    res = model.mleHomGP(
        X = X,
        Z = Y,
        lower = np.array([0.05,0.05]),
        upper = np.array([10,10]),
        maxit=1e4,
        init = {} # need to fix init so it doesn't need to be initialized explicitly: https://stackoverflow.com/questions/52488478/pytest-leaks-attrs-objects-between-tests
    )
    # R results
    theta_R = np.array([4.349992, 1.515677])
    g_R = 0.00742059

    cond1 = np.allclose(theta_R,res['theta'],rtol=1e-3)
    cond2 = np.allclose(g_R,res['g'],rtol=1e-2)
    assert cond1 and cond2
    return

if __name__ == "__main__":
    test_homGP_mcycle()
    test_mleHomGP_SIR()