from hetgpy.find_reps import find_reps
import numpy as np
from scipy.io import loadmat
from tests.utils import read_yaml
R_compare = read_yaml('tests/R/results/test_find_reps.yaml')
R_compare['mcycle']['X0'] = np.array(R_compare['mcycle']['X0']).reshape(-1,1)
R_compare['SIR']['X0'] = np.array(R_compare['SIR']['X0']).reshape(2,-1).T

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
    test_R = R_compare['mcycle']
    for key in ('X0', 'Z0', 'mult', 'Z', 'Zlist'):
        if key=="Zlist":
            for k in test[key]:
                assert np.allclose(test[key][k], np.array(test_R[key][str(k+1)]))
        else:
            assert np.allclose(test[key], test_R[key])

def test_find_reps_SIR():
    SIR = loadmat('tests/data/SIR.mat')
    
    X = SIR['X'][0:200,:]
    Z = SIR['Y'][0:200]
    test = find_reps(
        X = X,
        Z = Z,
        rescale=False, 
        return_Zlist=True,
        normalize=False
    )
    
    # run in R
    
    test_R = R_compare['SIR']
        
    for key in ('X0', 'Z0', 'mult', 'Z', 'Zlist'):
        if key=="Zlist":
            for k in test[key]:
                assert np.allclose(test[key][k], np.array(test_R[key][str(k+1)]))
        else:
            assert np.allclose(test[key], test_R[key])


if __name__ == "__main__":
    test_find_reps_on_mcycle()