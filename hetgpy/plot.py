'''
Suite of plotting functions for model checks/diagnostics/etc.
'''

import warnings
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_optimization_iterates(object, keys_and_title = None):
    r'''
    Plot maximum likelihood iterates

    Parameters
    ----------
    object: hetgpy.homGP.homGP or hetgpy.hetGP.hetGP model
        hetGPy object
    keys_and_title: iterable for model component to extract (theta, g, etc.)

    Returns
    -------
    fig, ax: matplotlib figure and axes
    '''
    def extract_variable(key):
        # extract iterates from model object
        out = np.array([d[key] for d in object['iterates']])
        if len(out.shape)==1:
            out = out.reshape(-1,1)
        return out
    
    fig, ax = plt.subplots(nrows=1,ncols = len(keys_and_title),figsize=(11.5,8))
    fig.supxlabel('Iteration')
    xs = np.arange(len(object['iterates']))
    i = 0
    for key, ax_title in keys_and_title.items():
        ys = extract_variable(key)
        for j in range(ys.shape[1]):
            label = key
            if key in ('theta','Delta'):
                label = r'$\{}_{}$'.format(key,j+1) 
            ax[i].plot(xs, ys[:,j],label=label)
        # axis options
        ax[i].set_title(ax_title)
        i+=1
    return fig, ax

def plot_diagnostics(model):
    r'''
    Diagnostics plot which mirrors the plot(model) routine in hetGP
    
    Plots the LOO predctions against the model data

    Parameters
    ----------
    model: hetGPy model

    Returns
    -------
    fig, ax: matplotlib figure and axes
    '''
    preds = model.predict(model.X0)
    preds['upper'] = norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'])).squeeze()
    preds['lower'] = norm.ppf(0.05, loc = preds['mean'], scale = np.sqrt(preds['sd2'])).squeeze()

    fig, ax = plt.subplots()
    idxs = np.repeat(np.arange(len(model.X0)),model.mult)
    ax.hlines(
        y=preds['mean'],
        xmin=preds['lower'],
        xmax=preds['upper'],
        label='Prediction Interval',zorder=-10)
    ax.scatter(model.Z,
        preds['mean'][idxs],
        facecolors='none',
        edgecolors='black',
        label='Observations',zorder=5)
    ax.axline((0, 0), slope=1,color='black',linestyle='dashed')

    ax.scatter(model.Z0[(model.mult>1).nonzero()[0]],
            preds['mean'][(model.mult>1).nonzero()[0]],
            label='Averages (if mult > 1)',color='red',zorder=10)
    ax.legend(loc='upper left',edgecolor='black')
    ax.set_title('Model Diagnostics')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    return fig, ax

    