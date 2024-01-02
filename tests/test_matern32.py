# test matern32

from hetgpy.covariance_functions import matern_32
from scipy.io import loadmat
import numpy as np

def test_32_mcycle():
    X1   = loadmat('tests/data/mcycle.mat')['times'].reshape(-1,1)
    Rmat = loadmat('tests/data/mcycle_Matern32.mat')

    C = matern_32(X1, X1, theta = Rmat['theta'])

    assert np.allclose(C,Rmat['C'])

def test_32_SIR():
    X1   = loadmat('tests/data/SIR.mat')['X']
    Rmat = loadmat('tests/data/SIR_Matern32.mat')

    C = matern_32(X1, X1, theta = Rmat['theta'])

    assert np.allclose(C,Rmat['C'])


if __name__ == "__main__":
    #test_32_mcycle()
    test_32_SIR()

