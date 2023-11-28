import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from hetgpy.utils import *
from hetgpy.hetGP import hetGP

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
    d = pd.read_csv('tests/data/mcycle.csv')
    X = d['times'].values.reshape(-1,1)
    Z = d['accel'].values
    model = hetGP()
    prdata = model.find_reps(
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
    d = pd.read_csv('tests/data/mcycle.csv')
    X = d['times'].values.reshape(-1,1)
    Z = d['accel'].values
    model = hetGP()
    prdata = model.find_reps(
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


    assert np.allclose(target_dll,computed_dll)


def test_ll_and_dll_hom_anisotropic():

    target_ll =  -97.12093
    target_dll = np.array([1.6656848, 0.1213165, 0.2708796, -12.7440165])

    d = pd.read_csv('tests/data/mcycle.csv')
    # make some fake inputs
    X = d['times'].values[0:90].reshape(30,3)
    Z = d['accel'].values[0:30]

    model = hetGP()

    prdata = model.find_reps(
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
    assert np.allclose(target_dll,computed_dll)
if __name__ == "__main__":
    test_ll_and_dll_hom_anisotropic()

