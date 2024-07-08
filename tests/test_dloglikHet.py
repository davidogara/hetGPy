# test_dloglikHet

from hetgpy import hetGP
from hetgpy.find_reps import find_reps
import numpy as np
from scipy.io import loadmat

def test_grad():
    # data
    X = loadmat('tests/data/mcycle.mat')['times'].reshape(-1,1)
    Z = loadmat('tests/data/mcycle.mat')['accel']
    d = find_reps(X,Z)

    R = loadmat('tests/data/checkGrad.mat')

    

    # model
    model = hetGP()
    model.Cg = None; model.Kg_c = None; model.Kgi = None
    model.C = None;  model.Ki = None;  model.ldetKi = None
    grad = model.dlogLikHet(X0 = d['X0'],
                     Z0 = d['Z0'],
                     mult = d['mult'],
                     Z = Z,
                     Delta = R['Delta'],
                     theta = np.array([R['theta']]),
                     g = R['g'],
                     k_theta_g=R['k_theta_g'],
                     hom_ll = float('-inf'),
                     covtype="Gaussian"

    )
    R_GRAD_OUT = R['gr_out']
                          
    assert np.allclose(grad,R_GRAD_OUT)
if __name__ == "__main__":
    test_grad()