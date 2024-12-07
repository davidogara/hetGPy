'''run gpytorch with replicates'''

# run_gpytorch
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
if not os.getcwd().endswith('py'):
    os.chdir('paper/py')
import pandas as pd
import numpy as np
from hetgpy.find_reps import find_reps
from hetgpy.homGP import homGP
import gpytorch
from gpytorch.distributions import base_distributions
import torch
from time import time
from scipy.io import loadmat
import numpy as np

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_scipy

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

class MyLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self,replicates,*params,**kwargs):
        '''
        Overrides GaussianLikelihood to include:
        param: replicates: 
                Tensor of len(train_x) that contains number of reps
                at each design location
        '''
        super().__init__()
        self.replicates = replicates
    def forward(self, function_samples, *params, **kwargs):
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        noise = noise.squeeze().div(self.replicates).reshape(-1,1)
        return base_distributions.Normal(function_samples, noise.sqrt())

def run_homGPyTorch(name,covtype):
    print(f"Running {name}")
    # initialize likelihood and model
    m = loadmat(f'../R/data/{name}.mat')
    X = m['X']
    Z = m['Y']
    d = find_reps(X,Z)
    X = d['X0']
    Z = d['Z0']
    replicates = torch.tensor(d['mult'].astype(int),dtype=int)
    likelihood = MyLikelihood(replicates=replicates)
    
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
    losses = []
    bounds = {'lengthscale': (0.1,5.0)}
    options = {'ftol':1e7*np.finfo(float).eps,'gtol':0}
    def cb(f,i):
        losses.append(i)
        return
    fit = fit_gpytorch_mll_scipy(mll,bounds=bounds,options=options,callback=cb)

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
    nreps = 5
    dfl = []
    for i in range(nreps):
        dfl.append(pd.concat([
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
    ]).assign(rep=i+1)
    )
    df = pd.concat(dfl)
    df.to_csv('./hetgpy-hom-GPyTorch-tests-SK.csv',index=False)