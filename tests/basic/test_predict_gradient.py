'''
Tests the gradient of the prediction
'''
from hetgpy import hetGP, homGP, crnGP
from hetgpy.example_data import mcycle
from hetgpy.optim import predict_gr
import numpy as np
import pytest

m = mcycle()
X, Y = m['times'], m['accel']
Xg = np.linspace(X.min(), X.max(), 100)[1:-1].reshape(-1,1)

def finite_differences(model, Xg, eps=1e-3):
    d = Xg.shape[1]
    spatial = d - 1 if isinstance(model, crnGP) else d
    grs = {}
    for key in ['mean', 'sd2']:
        cols = []
        for j in range(spatial):
            # perturb along inputs, but seed slot is 0 (no FD applied there)
            h = np.zeros(d); h[j] = eps
            cols.append((model.predict(Xg + h)[key] - model.predict(Xg - h)[key]) / (2*eps))
        grs[key] = np.column_stack(cols)
    return grs

@pytest.mark.parametrize("GP",[hetGP,homGP])
@pytest.mark.parametrize("ctype",["Matern5_2","Matern3_2","Gaussian"])
def test_predict_gr(GP,ctype):
    model = GP()
    model.mle(X = X, Z = Y, covtype=ctype)
    preds_gr = predict_gr(model,Xg)
    finite_diffs = finite_differences(model,Xg = Xg)
    assert np.allclose(preds_gr['mean'], finite_diffs['mean'],atol = 1e-4)
    assert np.allclose(preds_gr['sd2'], finite_diffs['sd2'],atol = 1e-2)

@pytest.mark.parametrize("ctype",["Matern5_2","Matern3_2","Gaussian"])
def test_predict_gr_crn(ctype):
    from itertools import product
    model = crnGP()
    rng = np.random.default_rng(42)
    S = rng.choice([1,2,3], size = len(X)).reshape(-1,1)
    Xtr = np.hstack([X,S])
    args = Xg,np.array([1,2,3]).reshape(-1,1)
    Xp = np.array(list(map(np.concatenate,list(product(*args)))))

    model.mle(X = Xtr, Z = Y, covtype=ctype)
    preds_gr = predict_gr(model,Xp)
    finite_diffs = finite_differences(model,Xg = Xp)
    assert np.allclose(preds_gr['mean'], finite_diffs['mean'],atol = 1e-4)
    assert np.allclose(preds_gr['sd2'], finite_diffs['sd2'],atol = 1e-4)



if __name__ == "__main__":
    test_predict_gr_crn(ctype='Matern5_2')
    test_predict_gr_crn(ctype='Matern3_2')
    test_predict_gr(GP=hetGP,ctype='Matern5_2')