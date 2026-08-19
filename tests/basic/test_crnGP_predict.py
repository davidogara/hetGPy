from hetgpy import crnGP
import numpy as np

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
if __name__ == "__main__":
    test_crn_predict()
