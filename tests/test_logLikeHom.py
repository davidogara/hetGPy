import sys
sys.path.append('../')
from scipy.io import loadmat
import numpy as np
from hetgpy.covariance_functions import *
from hetgpy.homGP import homGP
from hetgpy.find_reps import find_reps

def test_ll_hom():
    '''
    from R:
    X = mcycle$times %>% as.matrix()
    Z = mcycle$accel

    prdata = find_reps(X, Z)

    target_ll = logLikHom(
    X0 = prdata$X0,
    Z0 = prdata$Z0,
    Z = prdata$Z,
    beta0 = NULL,
    covtype = "Gaussian",
    mult = prdata$mult,
    theta = 0.5,
    g = 0.1
    )
    '''
    target_ll = -674.8049
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel']
    model = homGP()
    prdata = find_reps(
        X = X,
        Z = Z
    )
    computed_ll = model.logLikHom(
        X0 = prdata['X0'],
        Z0 = prdata['Z0'],
        Z = prdata['Z'],
        beta0 = None,
        covtype = "Gaussian",
        mult = prdata['mult'],
        theta = np.array([0.5]),
        g = 0.1
    )
    assert np.allclose(target_ll,computed_ll)

def test_dll_Hom():
    '''
    from R:
    X = mcycle$times %>% as.matrix()
    Z = mcycle$accel

    prdata = find_reps(X, Z)

    target_dll = dlogLikHom(
    X0 = prdata$X0,
    Z0 = prdata$Z0,
    Z = prdata$Z,
    beta0 = NULL,
    covtype = "Gaussian",
    mult = prdata$mult,
    theta = 0.5,
    g = 0.1
    )
    '''
    target_dll = (17.60184, 181.40440)
    d = loadmat('tests/data/mcycle.mat')
    X = d['times'].reshape(-1,1)
    Z = d['accel']
    model = homGP()
    prdata = find_reps(
        X = X,
        Z = Z
    )
    # why run this? because this generates Ki, C, and beta0
    computed_ll = model.logLikHom(
        X0 = prdata['X0'],
        Z0 = prdata['Z0'],
        Z = prdata['Z'],
        beta0 = None,
        covtype = "Gaussian",
        mult = prdata['mult'],
        theta = np.array([0.5]),
        g = 0.1
    )

    computed_dll = model.dlogLikHom(
        X0 = prdata['X0'],
        Z0 = prdata['Z0'],
        Z = prdata['Z'],
        beta0 = None,
        covtype = "Gaussian",
        mult = prdata['mult'],
        theta = np.array([0.5]),
        g = 0.1
    )

    print(target_dll)
    print(computed_dll)
    assert np.allclose(target_dll,computed_dll.squeeze())


def test_ll_and_dll_hom_anisotropic():

    target_ll =  -97.12093
    target_dll = np.array([1.6656848, 0.1213165, 0.2708796, -12.7440165])

    d = loadmat('tests/data/mcycle.mat')
    # make some fake inputs
    X = d['times'][0:90].reshape(30,3)
    Z = d['accel'][0:30]

    model = homGP()

    prdata = find_reps(
        X = X,
        Z = Z
    )
    computed_ll = model.logLikHom(
        X0 = prdata['X0'],
        Z0 = prdata['Z0'],
        Z = prdata['Z'],
        beta0 = None,
        covtype = "Gaussian",
        mult = prdata['mult'],
        theta = np.array([1,2,3]),
        g = 0.1
    )
    assert np.allclose(target_ll,computed_ll)

    computed_dll = model.dlogLikHom(
        X0 = prdata['X0'],
        Z0 = prdata['Z0'],
        Z = prdata['Z'],
        beta0 = None,
        covtype = "Gaussian",
        mult = prdata['mult'],
        theta = np.array([1,2,3]),
        g = 0.1
    )
    print(target_dll)
    print(computed_dll)
    assert np.allclose(target_dll,computed_dll.squeeze())
if __name__ == "__main__":
    test_ll_and_dll_hom_anisotropic()

