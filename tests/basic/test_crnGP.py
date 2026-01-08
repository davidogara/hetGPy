import sys
sys.path.append('./')
import numpy as np
from hetgpy import crnGP
from tests.utils import read_yaml
def test_known():
    compare = read_yaml('tests/R/results/crnGP_known.yaml')

    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    known = dict(theta = np.array([10]), g = np.array([0.1]), rho = np.array([0.6]))
    model = crnGP()
    model.mle(X=X,Z=Z,covtype="Gaussian",known=known)
    
    for key in compare.keys():
        assert np.allclose(model[key],compare[key])
    
    return

if __name__ == "__main__":
    test_known()