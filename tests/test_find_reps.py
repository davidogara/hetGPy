from hetgpy.find_reps import find_reps
from rpy2.robjects import default_converter, numpy2ri
import numpy as np
from time import time
from scipy.io import loadmat
from rpy2.robjects.packages import importr
hetGP_R = importr('hetGP')


def test_find_reps_on_mcycle():

    mcycle = loadmat('tests/data/mcycle.mat')
    X = mcycle['times'].reshape(-1,1)
    Z = mcycle['accel']
    test = find_reps(
        X = X,
        Z = Z,
        rescale=False, 
        return_Zlist=True,
        normalize=False
    )
    
    # run in R
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        test_R = hetGP_R.find_reps(
            X = X,
            Z = Z,
            rescale=False, 
            return_Zlist=True,
            normalize=False
        )
        
    for key in ('X0', 'Z0', 'mult', 'Z', 'Zlist'):
        if key=="Zlist":
            for k in test[key]:
                assert np.allclose(test[key][k], test_R[key][str(k+1)])
        else:
            assert np.allclose(test[key], test_R[key])

def test_find_reps_SIR():
    SIR = loadmat('tests/data/SIR.mat')
    X = SIR['X']
    Z = SIR['Y']
    test = find_reps(
        X = X,
        Z = Z,
        rescale=False, 
        return_Zlist=True,
        normalize=False
    )
    
    # run in R
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        test_R = hetGP_R.find_reps(
            X = X,
            Z = Z,
            rescale=False, 
            return_Zlist=True,
            normalize=False
        )
        
    for key in ('X0', 'Z0', 'mult', 'Z', 'Zlist'):
        if key=="Zlist":
            for k in test[key]:
                assert np.allclose(test[key][k], test_R[key][str(k+1)])
        else:
            assert np.allclose(test[key], test_R[key])


if __name__ == "__main__":
    test_find_reps_on_mcycle()