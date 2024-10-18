import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from hetgpy.homGP import homGP
from scipy.io import loadmat
import pandas as pd
import numpy as np
def run_homGP(name,covtype):
    print(f"Running {name}")
    m = loadmat(f"../R/data/{name}.mat")
    X = m['X']
    Y = m['Y']
    nvar = X.shape[1]
    model = homGP()
    model.mleHomGP(
        X = X,
        Z = Y,
        covtype = covtype,
        lower = np.repeat(0.1,nvar),
        upper = np.repeat(5.0,nvar),
        maxit = 1000
    )
    out = dict(
        time = model.time, 
        ll = model.ll,
        theta = list(np.round(model.theta,3)),
        nfev = model.nit_opt,
        msg = model.msg,
        name = name,
        covtype = covtype
    )
    return pd.DataFrame([out])
if __name__ == "__main__":

    df = pd.concat([
        run_homGP('Branin', "Gaussian"),
        run_homGP("Goldstein-Price","Gaussian"),
        run_homGP("Hartmann-4D","Gaussian"),
        run_homGP("Hartmann-6D","Gaussian"),
        run_homGP("Sphere-6D","Gaussian"),
        run_homGP('Branin', "Matern5_2"),
        run_homGP("Goldstein-Price","Matern5_2"),
        run_homGP("Hartmann-4D","Matern5_2"),
        run_homGP("Hartmann-6D","Matern5_2"),
        run_homGP("Sphere-6D","Matern5_2")
    ])
    df.to_csv('./hetgpy-hom-GP-tests.csv',index=False)


    
