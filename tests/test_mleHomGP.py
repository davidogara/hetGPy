from hetgpy import hetGP
import numpy as np
import pandas as pd

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
        lower = np.array([0.1]),
        upper = np.array([10])
    )

    # get R values
    theta_R = 10
    g_R = 0.5009635

    cond1 = np.allclose(theta_R,res['theta'])
    cond2 = np.allclose(res['g'],g_R)
    assert cond1 and cond2

if __name__ == "__main__":
    test_homGP_mcycle()