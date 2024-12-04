from hetgpy import hetGP
from hetgpy.find_reps import find_reps
from hetgpy.IMSE import allocate_mult
from hetgpy.example_data import mcycle
from tests.utils import read_yaml
import numpy as np



def test_allocate_mult():
    m = mcycle()
    X, Z = m['times'], m['accel']
    data_m = find_reps(X, Z, rescale=True)
    # align models for comparison
    r_model      = read_yaml('tests/R/results/test_allocate_mult.yaml')
    model = hetGP()
    model.mleHetGP(X = {'X0':data_m['X0'], 
                        'Z0' : data_m['Z0'], 
                        'mult': data_m['mult']},
                   Z = Z,
                   known = {'theta':np.array(r_model['theta']).reshape(-1),
                            'g': r_model['g']},
                   lower = np.array([0.1]),
                   upper = np.array([5.0]),
                   covtype = "Matern5_2",
                   maxit=1000)
    model.Delta  = r_model['Delta']
    model.Lambda = r_model['Lambda']

    A = allocate_mult(model,N = 1000).astype(int)
    Ar = r_model['A']
    assert np.allclose(A,Ar,atol=1) # allow for difference of one replicate

if __name__ == "__main__":
    test_allocate_mult()
