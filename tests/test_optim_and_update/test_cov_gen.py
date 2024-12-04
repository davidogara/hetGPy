import pytest
from hetgpy.covariance_functions import cov_gen
from hetgpy.example_data import mcycle
from hetgpy.find_reps import find_reps
from scipy.io import loadmat
import numpy as np
# R objects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
hetGP_R = importr('hetGP')
np_cv_rules = default_converter + numpy2ri.converter

## 1D Examples: mcycle data with Gauss and Matern5_2

@pytest.fixture()
def mcycleX0():
    m = mcycle()
    X = m['times'].reshape(-1,1)
    Z = m['accel']
    test = find_reps(
        X = X,
        Z = Z,
        rescale=False, 
        return_Zlist=True,
        normalize=False
    )
    return test['X0']

def test_mcycle_gauss(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([1]), type = "Gaussian")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 1,type = "Gaussian")
    assert np.allclose(C,C_R)

def test_mcycle_gauss_theta_2(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([2]), type = "Gaussian")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 2,type = "Gaussian")
    assert np.allclose(C,C_R)

def test_mcycle5_2_matern(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([1]), type = "Matern5_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 1,type = "Matern5_2")
    assert np.allclose(C,C_R)

def test_mcycle5_2_matern(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([2]), type = "Matern5_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 2,type = "Matern5_2")
    assert np.allclose(C,C_R)

def test_mcycle3_2_matern(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([1]), type = "Matern3_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 1,type = "Matern3_2")
    assert np.allclose(C,C_R) 

def test_mcycle3_2_matern_theta2(mcycleX0):
    C = cov_gen(X1 = mcycleX0, theta = np.array([2]), type = "Matern3_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = mcycleX0,theta = 2,type = "Matern3_2")
    assert np.allclose(C,C_R) 

## 2D Example: SIR Model
    
@pytest.fixture()
def SIRX0():
    S = loadmat('tests/data/SIR.mat')
    X = S['X']
    Z = S['Y']
    test = find_reps(
        X = X,
        Z = Z,
        rescale=False, 
        return_Zlist=True,
        normalize=False
    )
    return test['X0']

def test_SIR_gauss(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,1]), type = "Gaussian")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,1]),type = "Gaussian")
    assert np.allclose(C,C_R)

def test_SIR_gauss_aniso(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,2]), type = "Gaussian")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,2]),type = "Gaussian")
    assert np.allclose(C,C_R)

def test_SIR_matern5_2(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,1]), type = "Matern5_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,1]),type = "Matern5_2")
    assert np.allclose(C,C_R)

def test_SIR_matern5_2_aniso(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,2]), type = "Matern5_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,2]),type = "Matern5_2")
    assert np.allclose(C,C_R)

def test_SIR_matern3_2(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,1]), type = "Matern3_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,1]),type = "Matern3_2")
    assert np.allclose(C,C_R)

def test_SIR_matern3_2_aniso(SIRX0):
    C = cov_gen(X1 = SIRX0, theta = np.array([1,2]), type = "Matern3_2")
    with np_cv_rules.context():
        C_R = hetGP_R.cov_gen(X1 = SIRX0,theta = np.array([1,2]),type = "Matern3_2")
    assert np.allclose(C,C_R)