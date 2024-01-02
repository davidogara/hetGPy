# test_matern52.py
from hetgpy.covariance_functions import matern_52
from scipy.io import loadmat
import numpy as np

def test_52_SIR():
    X1   = loadmat('tests/data/SIR.mat')['X']
    Rmat = loadmat('tests/data/SIR_Matern52.mat')

    C = matern_52(X1, X1, theta = Rmat['theta'])

    assert np.allclose(C,Rmat['C'])
if __name__ == "__main__":
    test_52_SIR()