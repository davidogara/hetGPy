from hetgpy.IMSE import horizon
from hetgpy.hetGP import hetGP
from hetgpy.find_reps import find_reps
import numpy as np
from tests.utils import read_yaml
from hetgpy.example_data import mcycle
def test_horizon():
    # R model
    model = hetGP()
    m = mcycle()
    X, Z = m['times'], m['accel']
    data_m = find_reps(X, Z, rescale=True)
    r_model      = read_yaml('tests/R/results/test_horizon.yaml')
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
    h = horizon(model,current_horizon = 1, previous_ratio=0.5,target=0.75)
    rh = r_model['h']
    assert h == rh
if __name__ == "__main__":
    test_horizon()