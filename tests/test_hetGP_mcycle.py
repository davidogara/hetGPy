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
        covtype='Gaussian',
        settings={'factr': 10e7} # high quality solution
    )

    
    # predict
    xgrid =  np.linspace(0,60,301).reshape(-1,1)
    preds = model.predict_hetGP(
        object = res,
        x = xgrid
    )

    # Compare to R:
    mat = loadmat('tests/data/hetGP_mcycle.mat')
    preds_mean_R = mat['mean']
    preds_sd_R   = mat['sd2']
    
    
    # model pars
    #assert np.allclose(res['theta'],mat['theta'])
    #assert np.allclose(res['g'],mat['g'])
    #assert np.allclose(res['Ki'], mat['Ki'],atol=1e-7)
    
    # predictions
    #assert np.allclose(preds['mean'],preds_mean_R,atol=1e-6)
    #assert np.allclose(preds['sd2'],preds_sd_R,atol=1e-6)
    #make_plot(preds,mat,xgrid,X,Z,save_plot=True)
    #same_plot(preds,mat,xgrid,X,Z, save_plot = True)
    return

     
def same_plot(preds,mat,xgrid,X,Z, save_plot = True):
        '''
        Make a plot comparing the hetGP to hetGPy.
        Note that this function assumes a LaTeX installation (due to usetex = True)
        '''
        if os.path.exists('mplstyle/latex.mplstyle'): plt.style.use('mplstyle/latex.mplstyle')
        preds_mean_R = mat['mean']
        preds_sd_R   = mat['sd2']
        preds_sd_R_05 = mat['CI_lower']
        preds_sd_R_95 = mat['CI_upper']
        df = pd.DataFrame(
            {
            # python results
            'hetGPy': preds['mean'],
            'hetGPy-0.05': norm.ppf(0.05, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze(),
            'hetGPy-0.95': norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze(),
            # R results
            'hetGP': preds_mean_R,
            'hetGP-0.05': preds_sd_R_05,
            'hetGP-0.95': preds_sd_R_95
            },
            index=xgrid.squeeze()
        )
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (11.5,8))
        
        ax.scatter(x=X,y=Z,color="k",label='Data')
        ax.plot(df['hetGPy'],color='b',label='hetGPy')
        ax.plot(df['hetGPy-0.05'],color='b',linestyle='dashed')
        ax.plot(df['hetGPy-0.95'],color='b',linestyle='dashed')
        
        
        ax.plot(df['hetGP'],color='r',label='hetGP')
        ax.plot(df['hetGP-0.05'],color='r',linestyle='dashed')
        ax.plot(df['hetGP-0.95'],color='r',linestyle='dashed')
        fig.suptitle('hetGP Predictive Surface')
        
        ax.legend(edgecolor='black')
        if save_plot: 
            
            fig.tight_layout()
            fig.savefig('tests/figures/hetGP_mcycle_same_plot.pdf'); plt.close()
        else:
            plt.show()

def make_plot(preds,mat,xgrid,X,Z, save_plot = True):
        '''
        Make a plot comparing the hetGP to hetGPy.
        Note that this function assumes a LaTeX installation (due to usetex = True)
        '''
        if os.path.exists('mplstyle/latex.mplstyle'): plt.style.use('mplstyle/latex.mplstyle')
        preds_mean_R = mat['mean']
        preds_sd_R   = mat['sd2']
        preds_sd_R_05 = mat['CI_lower']
        preds_sd_R_95 = mat['CI_upper']
        df = pd.DataFrame(
            {
            # python results
            'hetGPy': preds['mean'],
            'hetGPy-0.05': norm.ppf(0.05, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze(),
            'hetGPy-0.95': norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze(),
            # R results
            'hetGP': preds_mean_R,
            'hetGP-0.05': preds_sd_R_05,
            'hetGP-0.95': preds_sd_R_95
            },
            index=xgrid.squeeze()
        )
        df['diff'] = df['hetGPy'] - df['hetGP']
        df['diff-0.05'] = df['hetGPy-0.05'] - df['hetGP-0.05']
        df['diff-0.95'] = df['hetGPy-0.95'] - df['hetGP-0.95']
        fig, ax = plt.subplots(nrows=1,ncols=3, figsize = (11.5,8))
        
        ax[0].scatter(x=X,y=Z,color="k",label='Data')
        ax[0].plot(df['hetGPy'],color='b',label='hetGPy')
        ax[0].plot(df['hetGPy-0.05'],color='b',linestyle='dashed')
        ax[0].plot(df['hetGPy-0.95'],color='b',linestyle='dashed')
        
        ax[1].scatter(x=X,y=Z,color="k",label='Data')
        ax[1].plot(df['hetGP'],color='r',label='hetGP')
        ax[1].plot(df['hetGP-0.05'],color='r',linestyle='dashed')
        ax[1].plot(df['hetGP-0.95'],color='r',linestyle='dashed')
        
        ax[2].plot(df['diff'],color='k',label='diff (mean)')
        ax[2].plot(df['diff-0.05'],color='k',label = 'diff (lower)',linestyle='dashed')
        ax[2].plot(df['diff-0.95'],color='k',label = 'diff (upper)',linestyle='dashed')

        ax[1].set_xlabel('times (ms)')
        ax[0].set_ylabel('accel (g)')
        fig.suptitle('hetGP Predictive Surface')
        for a in ax:
            a.legend(edgecolor='black')
        if save_plot: 
            
            fig.tight_layout()
            fig.savefig('tests/figures/hetGP_mcycle.pdf'); plt.close()
        else:
            plt.show()


if __name__ == "__main__" :
    test_hetGP_mcycle()