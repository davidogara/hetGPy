# test EI example from hetGP documentation
from hetgpy import hetGP
from hetgpy.optim import crit_EI
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects import r
from scipy.io import loadmat
def test_EI_f1d():
    # from hetGP examplse
    m = loadmat('tests/data/f1d.mat')
    X = m['X'].reshape(-1,1)
    Z = m['Z']
    f1d_xgrid = m['f1dxgrid']
    
    ## Predictive grid
    ngrid = 51
    xgrid = np.linspace(0,1, ngrid)
    Xgrid = xgrid.reshape(-1,1)

    
    
    model = hetGP() 
    model.mleHetGP(X = X, Z = Z, 
                lower = np.array([0.001]), 
                upper = np.array([1.0]), 
                known = {}, init = {},
                maxit=5e3,settings={'factr':10},
                )

    EIgrid = crit_EI(Xgrid, model)
    preds = model.predict(x = Xgrid)
    preds['upper'] = norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()
    preds['lower'] = norm.ppf(0.05, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

if __name__ == "__main__":
    test_EI_f1d()
