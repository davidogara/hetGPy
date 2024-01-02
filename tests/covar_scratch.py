from hetgpy.covariance_functions import euclidean_dist
from scipy.io import loadmat
import scipy
import numpy as np
from timeit import timeit
from time import time
import numba as nb
from scipy.spatial.distance import cdist

def compare_mcycle():
    '''1D test'''
    X1 = loadmat('tests/data/mcycle.mat')['times'].reshape(-1,1)

    euclid = euclidean_dist(X1,X1)
    numpy_norm = np.linalg.norm(X1[None,:]-X1[:,None],axis=2)**2
    assert np.allclose(euclid,numpy_norm)

def compare_SIR():
    '''2D example'''
    X1 = loadmat('tests/data/SIR.mat')['X']
    euclid = euclidean_dist(X1,X1)
    numpy_norm = np.linalg.norm(X1[None,:]-X1[:,None],axis=2)**2
    assert np.allclose(euclid,numpy_norm)

@nb.njit()
def eu_numba(X1):
    return np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * X1.dot(X1.T) + np.sum(np.square(X1), axis=1)

def eu_no_numba(X1):
    return np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * X1.dot(X1.T) + np.sum(np.square(X1), axis=1)

@nb.njit("float64[:,:](float64[:,::1])")
def eu_with_sig(X1):
    return np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * X1 @ X1.T + np.sum(np.square(X1), axis=1)

def eu_scipy_spatial(X1):
    return cdist(X1,X1)**2

def eu_with_sig_no_dot(X1):
    return np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * np.tensordot(X1, X1.T, axes=(1, 0)) + np.sum(np.square(X1), axis=1)


def eu_einsum(X1):
    return np.einsum('ijk->ij',np.square(X1)[:,np.newaxis,:]) - 2.0 * np.einsum('ij,jk->ik',X1,X1.T) + np.einsum('ij->i',np.square(X1))

def eu_einsum_opt(X1):
    return np.einsum('ijk->ij',np.square(X1)[:,np.newaxis,:],optimize=True) - 2.0 * np.einsum('ij,jk->ik',X1,X1.T,optimize=True) + np.einsum('ij->i',np.square(X1),optimize=True)




@nb.njit()
def numpy_dists(X1):
    return scipy.linalg.norm(X1[None,:]-X1[:,None],axis=2)**2

def time_SIR():
    X1 = loadmat('tests/data/SIR.mat')['X']
    X1 = np.ascontiguousarray(X1)
    # precompile 
    tmp = eu_with_sig(X1)

    raw      = timeit('eu_no_numba(X1)', number=5, globals=globals(), setup="X1 = loadmat('tests/data/SIR.mat')['X']")
    with_sig = timeit('eu_with_sig(X1)', number=5, globals=globals(), setup="X1 = loadmat('tests/data/SIR.mat')['X'];X1 = np.ascontiguousarray(X1)")
    spatial =  timeit('eu_scipy_spatial(X1)', number=5, globals=globals(), setup="X1 = loadmat('tests/data/SIR.mat')['X'];X1 = np.ascontiguousarray(X1)")
    
    print(f"Raw: {round(raw,5)}")
    print(f"Sig: {round(with_sig,5)}")
    print(f"spatial: {round(spatial,5)}")
    

def check_cdist():
    X1 = loadmat('tests/data/SIR.mat')['X']
    euclid = euclidean_dist(X1,X1)
    cd = cdist(X1,X1)**2
    assert np.allclose(euclid,cd)
    

if __name__ == "__main__":
    time_SIR()


