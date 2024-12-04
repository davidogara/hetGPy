'''
Test the auto_bounds functionality
'''

import sys
sys.path.append('../')
from hetgpy.auto_bounds import auto_bounds
import numpy as np
from tests.utils import read_yaml
R_results = read_yaml('tests/R/results/test_auto_bounds.yaml')

X = np.array([[ 1, 34],
       [30,  3],
       [45, 11],
       [13, 10],
       [17,  9]])

def compare_to_R(covtype):
    r_out = R_results[covtype]
    l = r_out['lower']
    u = r_out['upper']
    return dict(lower=l,upper=u)
def test_gauss():
    ctype ="Gaussian"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 
def test_matern3():
    ctype ="Matern3_2"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 
def test_matern5():
    ctype ="Matern5_2"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 

if __name__ == "__main__":
    test_matern5()