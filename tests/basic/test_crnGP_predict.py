from hetgpy import crnGP
import numpy as np
import pytest
def test_crn_predict():
    '''
    Check for state mutation
    '''
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    model = crnGP()
    model.mle(X = X , Z = Z, covtype='Matern5_2')
    Xtest = np.repeat(np.array([1.0,2]).reshape(1,-1),repeats = 10,axis=0)
    preds = []
    Kitemp = model.Ki.copy()
    for x in Xtest:
        x = x.reshape(1,-1)
        preds.append(model.predict(x)['mean'])
    preds = np.array(preds)
    # check they are all the same
    assert (preds[0] == preds).all()

def test_forgotten_seed_input():
    '''
    Reproduces issue  on hetgpy == 1.0.6
    '''
    rng = np.random.default_rng(123)
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    Z += (0.2 * rng.normal(size = len(Z)))
    model = crnGP()
    model.mle(X = X , Z = Z, covtype='Matern5_2')
    Xp = (np.array([6.0])).reshape(-1,1)
    # forgetting the seed should throw an error
    with pytest.raises(ValueError):
        pred = model.predict(x=Xp)

    # and including it should pass
    Xps = np.column_stack([Xp,1])
    pred = model.predict(Xps)

def test_crnGP_EI_default_cst():
    '''
    Default cst in EI needs to sub in a seed
    '''
    rng = np.random.default_rng(123)
    pps = 10 # points per seed
    x = np.linspace(0,2*np.pi,pps).reshape(-1,1)
    X = np.vstack([x,x])
    seeds = ([1] * pps) + ([2] * pps)
    X = np.hstack([X,np.array(seeds).reshape(-1,1)])
    Z = np.sin(X[:,0]) + X[:,-1]
    Z += (0.2 * rng.normal(size = len(Z)))
    model = crnGP()
    model.mle(X = X , Z = Z, covtype='Matern5_2')
    # make grid from training data with a little offset
    Xgrid= X.copy()
    Xgrid[:,0] += 0.3
    crit = model.crit_EI(Xgrid,cst=None)
    return
    
if __name__ == "__main__":
    test_crnGP_EI_default_cst()
