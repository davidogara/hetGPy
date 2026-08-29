'''
Test crnGPs shape
'''
from hetgpy import crnGP
import numpy as np
import pytest
def test_crnGP_shape():
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]

    model = crnGP()
    # catch 1d array error
    with pytest.raises(ValueError):
        model.mle(X[:,0],Z)

    # check for forgetting seed col
    with pytest.raises(ValueError):
        model.mle(X[:,0:1],Z)

    # check for seed not resolving
    # i.e. seed seed col to seed + 0.1
    with pytest.raises(ValueError):
        tmp = X.copy()
        tmp[:,-1] += 0.1
        model.mle(tmp,Z)

if __name__ == "__main__":
    test_crnGP_shape()