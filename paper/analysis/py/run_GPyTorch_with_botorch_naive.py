'''run gpytorch with replicates'''

# run_gpytorch
import os
if not os.getcwd().endswith('py'):
    os.chdir('paper/py')
import pandas as pd
import numpy as np
from hetgpy.find_reps import find_reps
from hetgpy.homGP import homGP
import gpytorch
import torch
from time import time
from scipy.io import loadmat
import numpy as np

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_scipy

def run_homGPyTorch(name,covtype):
    print(f"Running {name}")
    # initialize likelihood and model
    m = loadmat(f'../R/data/{name}.mat')
    X = m['X']
    Z = m['Y']
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    if covtype == "Gaussian":
        K = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
    elif covtype == "Matern5_2":
        K = gpytorch.kernels.MaternKernel(nu=5/2,ard_num_dims=X.shape[1])
    else:
        raise KeyError(f"Not supported {covtype}")
    GP = SingleTaskGP(
        train_X=torch.from_numpy(X), 
        train_Y=torch.from_numpy(Z).reshape(-1,1), 
        likelihood=likelihood,
        covar_module=K)
    
    # Find optimal model hyperparameters
    GP.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, GP)
    s = time()
    bounds = {'lengthscale': (0.1,5.0)}
    fit = fit_gpytorch_mll_scipy(mll,bounds=bounds)

    e = time() - s
    # compute ll
    X = m['X']
    Z = m['Y']
    prdata = find_reps(X,Z)
    X0 = prdata['X0']
    Z0 = prdata['Z0']
    mult = prdata['mult']
    tmp_model = homGP()
    theta = GP.covar_module.lengthscale.detach().numpy().squeeze()
    ll = tmp_model.logLikHom(
        X0, Z0, Z, mult, theta = theta, 
        g = likelihood.noise.item(), 
        beta0 = None, covtype = covtype)
    out = dict(
        time = e, 
        ll = ll,
        theta = list(np.round(theta,3)),
        nfev = fit.step,
        msg = "",
        name = name,
        covtype = covtype
    )
    return pd.DataFrame([out])


if __name__ == "__main__":

    df = pd.concat([
        run_homGPyTorch('Branin', "Gaussian"),
        run_homGPyTorch("Goldstein-Price","Gaussian"),
        run_homGPyTorch("Hartmann-4D","Gaussian"),
        run_homGPyTorch("Hartmann-6D","Gaussian"),
        run_homGPyTorch("Sphere-6D","Gaussian"),
        run_homGPyTorch('Branin', "Matern5_2"),
        run_homGPyTorch("Goldstein-Price","Matern5_2"),
        run_homGPyTorch("Hartmann-4D","Matern5_2"),
        run_homGPyTorch("Hartmann-6D","Matern5_2"),
        run_homGPyTorch("Sphere-6D","Matern5_2")
    ])
    df.to_csv('./hetgpy-hom-GPyTorch-tests-naive.csv',index=False)