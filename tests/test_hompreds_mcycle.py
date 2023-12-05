import pandas as pd
from scipy.io import loadmat
import numpy as np
from hetgpy import hetGP
import matplotlib.pyplot as plt

def test_hom_preds_mcycle():
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
        init = dict(),
        covtype="Gaussian"
    )
    xgrid        =  np.linspace(0,60,301).reshape(-1,1)
    
    # predict
    preds = model.predict_hom_GP(
        object = res,
        x = xgrid
    )
    mat = loadmat('tests/data/mcycle_Gauss_hom_preds.mat')
    preds_mean_R = mat['mean']
    preds_sd_R   = mat['sd2']

    # plot
    df = pd.DataFrame(
        {'hetGPy': preds['mean'],
         'hetGP': preds_mean_R}
    )
    df['diff'] = df['hetGPy'] - df['hetGP']
    fig, ax = plt.subplots(nrows=1,ncols=3)
    ax[0].plot(df['hetGPy'],color='b',label='hetGPy')
    ax[1].plot(df['hetGP'],color='r',label='hetGP')
    ax[2].plot(df['diff'],color='k',label='diff')
    fig.legend()
    plt.show()

    assert np.allclose(preds['mean'],preds_mean_R,atol=1e-2)
    assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-2)

if __name__ == "__main__" :
    test_hom_preds_mcycle()