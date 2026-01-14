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

def test_known_with_time():

    compare = read_yaml('tests/R/results/crnGP_known_with_time.yaml')

    XS = np.array(compare['XS']).reshape(-1,2,order='F')
    ids = np.array(compare['ids']) - 1
    t = np.array(compare['t'])
    Y0 = np.array(compare['Y0']).reshape(50,len(t),order='F')

    model = crnGP()
    known = dict(theta = np.array([0.3,0.5]), g = np.array([1e-3]), rho = np.array([0.3]))
    model.mle(XS[ids,:], Z = Y0, T0=t, covtype="Matern5_2",known=known)
    model.beta0 = model.Ki.sum(axis=1) @ model.Z.flatten(order='F') / model.Ki.sum()
    for key in ['ll','theta','g','rho','beta0']:
        assert np.allclose(model[key],np.array(compare[key]))
    foo=1

def test_predict_OK():
    compare = read_yaml('tests/R/results/crnGP_preds_OK.yaml')
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    known = dict(theta = np.array([10]), g = np.array([0.1]), rho = np.array([0.6]))
    model = crnGP()
    model.mle(X=X,Z=Z,covtype="Gaussian",known=known)

    npred = 50

    xp = np.linspace(0,2*np.pi,npred).reshape(-1,1)
    Xp = np.vstack([xp,xp])
    pseeds = ([1] * npred) + ([2] * npred)
    Xp= np.hstack([Xp,np.array(pseeds).reshape(-1,1)])

    preds = model.predict(x=Xp,xprime=Xp)
    keys = ['mean','sd2','nugs']
    for key in keys:
        assert np.allclose(preds[key],compare[key])
    
    assert np.allclose(preds['cov'],np.array(compare['cov']).reshape(100,100),atol=1e-6)
    return

def test_predict_SK():
    compare = read_yaml('tests/R/results/crnGP_preds_SK.yaml')
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    known = dict(theta = np.array([10]), g = np.array([0.1]), rho = np.array([0.6]),beta0 = np.mean(Z))
    model = crnGP()
    model.mle(X=X,Z=Z,covtype="Gaussian",known=known)

    npred = 50

    xp = np.linspace(0,2*np.pi,npred).reshape(-1,1)
    Xp = np.vstack([xp,xp])
    pseeds = ([1] * npred) + ([2] * npred)
    Xp= np.hstack([Xp,np.array(pseeds).reshape(-1,1)])

    preds = model.predict(x=Xp,xprime=Xp)
    keys = ['mean','sd2','nugs']
    for key in keys:
        assert np.allclose(preds[key],compare[key])
    
    assert np.allclose(preds['cov'],np.array(compare['cov']).reshape(100,100),atol=1e-6)
    return



if __name__ == "__main__":
    test_known_with_time()