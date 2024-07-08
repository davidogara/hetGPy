from scipy.io import loadmat
from scipy.stats import norm
import numpy as np
from hetgpy import hetGP
from hetgpy.optim import crit_EI
import pytest
from rpy2.robjects import r
import gc
from time import time

@pytest.fixture()
def mcycle():
     '''
    1D testing case
    '''
     d = loadmat('tests/data/mcycle.mat')
     X = d['times'].reshape(-1,1)
     Z = d['accel']
     return X, Z


def test_EI(mcycle):
    X, Z = mcycle
    model = hetGP()
    model.mleHetGP(X = X, Z = Z)
    xgrid = np.linspace(0,60,301)
    EIs = crit_EI(model = model, x = xgrid)
    proposal = xgrid[EIs.argmax()]

    r('''
    library(MASS)
    library(hetGP)
    X = as.matrix(mcycle$times)
    Z = mcycle$accel
    xgrid = as.matrix(seq(0,60,length.out = 301))
    model = mleHetGP(X = X, Z = Z, 
                covtype = "Gaussian",
                lower = 1, upper = 100,maxit=2e2,
                settings = list(factr=10e7))
    EIs = crit_EI(x = xgrid, model = model)
    proposal = xgrid[which.max(EIs)]
    '''
    )
    assert np.allclose(proposal, np.array(r('proposal'))[0])
if __name__ == "__main__":
    m = loadmat('tests/data/mcycle.mat')
    X = m['times'].reshape(-1,1)
    Z = m['accel']
    test_EI((X,Z))