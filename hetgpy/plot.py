import matplotlib.pyplot as plt
import numpy as np
import os

def plot_optimization_iterates(object, keys_and_title = None , stylesheet = None):
    if stylesheet is not None and os.path.exists(stylesheet): plt.style.use('mplstyle/latex.mplstyle')
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
    