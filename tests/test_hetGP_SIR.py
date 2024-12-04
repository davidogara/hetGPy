from scipy.io import loadmat
from scipy.stats import norm
import numpy as np
from hetgpy import hetGP
import pytest
from tests.utils import read_yaml
R_compare = read_yaml('tests/R/results/test_hetGP_SIR.yaml')



@pytest.fixture()
def SIR():
    d = loadmat('tests/data/SIR.mat')
    X = d['X']
    Z = d['Y']
    return X, Z



def compute_SIR(X,Z,covtype):
    xseq = np.linspace(0,1,100)
    xgrid = np.array([(y,x) for x in xseq for y in xseq])

    model = hetGP()

    model.mleHetGP(X = X, 
                   Z = Z, 
                   covtype=covtype,
                   known= {},
                   init = {},
                   lower = np.array([0.5,0.5]),
                   upper = np.array([2,2]),
                   maxit=5e3,
                   settings={'factr':10e7}
                   )
    preds = model.predict(x=xgrid)
    preds['upper'] =  norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

    
    py_preds = dict(mean=preds['mean'],upper=preds['upper'],theta=model['theta'])
    r_preds  = R_compare[covtype]
    
    return py_preds, r_preds

def test_hetGP_SIR_Matern5_2(SIR):
    X, Y = SIR
    py_preds, r_preds = compute_SIR(X,Y,covtype="Matern5_2")
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=0.025)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=0.025)
    

if __name__ == "__main__" :
    m = loadmat('tests/data/SIR.mat')
    X = m['X']
    Z = m['Y']
    test_hetGP_SIR_Matern5_2((X,Z))

