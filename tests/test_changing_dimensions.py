# test multiple inputs over repeated experiments
import sys
sys.path.append('../')
sys.path.append('./')
import numpy as np
rand = np.random.default_rng(42)
from hetgpy import hetGP, homGP

SETTINGS = {'ignore_MLE_divide_invalid':True} # turn this on to supress runtime divide by 0 or NA warnings
def test_hom():
    n = 50
    p = 3
    X = rand.integers(low=1,high=50,size=(n,p))
    Z = np.sin(np.arange(len(X)))
    model = homGP()
    model.mle(X,Z,covtype='Gaussian',settings=SETTINGS)

    model2 = homGP()
    extra = rand.integers(low=1,high=10,size=n).reshape(-1,1)
    X2 = np.hstack([X, extra])
    model2.mle(X2,Z,covtype='Gaussian',init={},known={},settings=SETTINGS)

def test_het():
    n = 50
    p = 3
    X = rand.integers(low=1,high=50,size=(n,p))
    Z = np.sin(np.arange(len(X)))
    model = hetGP()
    model.mle(X,Z,covtype='Gaussian',settings=SETTINGS)

    model2 = hetGP()
    extra = rand.integers(low=1,high=10,size=n).reshape(-1,1)
    X2 = np.hstack([X, extra])
    model2.mle(X2,
               Z,
               covtype='Gaussian',
               lower= [0.1 for i in range(X2.shape[1])],
               upper= [10 for i in range(X2.shape[1])],
               init={},known={},
               settings=SETTINGS)

if __name__ == "__main__":
    test_hom()
    test_het()