from scipy.io import loadmat
from scipy.stats import norm
import numpy as np
from hetgpy import hetGP
import pytest
from tests.utils import read_yaml
R_compare = read_yaml('tests/R/results/test_hetGP_mcycle.yaml')

@pytest.fixture()
def mcycle():
     '''
    1D testing case
    '''
     d = loadmat('tests/data/mcycle.mat')
     X = d['times'].reshape(-1,1)
     Z = d['accel']
     return X, Z

def compute_mcycle(X,Z,covtype):
    model = hetGP()

    # train
    model.mleHetGP(
        X = X,
        Z = Z,
        init = {},
        known = {},
        maxit=1e3,
        lower = np.array([1]),
        upper = np.array([100]),
        covtype=covtype,
        settings = {'factr': 10e7}
    )
    # predict
    xgrid =  np.linspace(0,60,301).reshape(-1,1)
    preds = model.predict(
        x = xgrid
    )
    preds['upper'] =  norm.ppf(0.95, loc = preds['mean'], scale = np.sqrt(preds['sd2'] + preds['nugs'])).squeeze()

    
    py_preds = dict(mean=preds['mean'],upper=preds['upper'],theta=model['theta'])
    r_preds  = R_compare[covtype]
    
    
    return(py_preds,r_preds)
def test_hetGP_mcycle_Gaussian(mcycle):
    X, Z = mcycle
    py_preds, r_preds = compute_mcycle(X,Z,covtype="Gaussian")
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=5)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=10)
    
    

def test_hetGP_mcycle_Matern5_2(mcycle):
    X, Z = mcycle
    py_preds, r_preds = compute_mcycle(X,Z,covtype="Matern5_2")
    assert np.allclose(py_preds['mean'],r_preds['mean'],atol=5)
    assert np.allclose(py_preds['upper'],r_preds['upper'],atol=15)