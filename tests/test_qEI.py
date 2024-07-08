from hetgpy.test_functions import f1d
from hetgpy.homGP import homGP
from hetgpy.optim import crit_qEI
import numpy as np
def test():
    ftest = f1d
    n_init = 5 # number of unique designs
    X = np.linspace(0, 1, n_init).reshape(-1,1)
    Z = ftest(X).squeeze()
    xgrid = np.linspace(0,1,51).reshape(-1,1)
    model = homGP()
    model.mleHomGP(X = X, 
                Z = Z, 
                lower = np.array([0.01]), 
                upper = np.array([1.0]), 
                known = {'g': 2e-8}, 
                covtype = "Gaussian")
    cst = model.Z0.min()
    xbatch = np.array((0.37, 0.17, 0.7)).reshape(3, 1)
    fqEI = crit_qEI(xbatch, model, cst)
    return
if __name__ == "__main__":
    test()