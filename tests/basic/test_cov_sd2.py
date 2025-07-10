import numpy as np
from hetgpy import hetGP, homGP
from hetgpy.example_data import mcycle
from scipy.io import loadmat
m = mcycle() #1D example


def _compare(X,Z,covtype,mtype='hetGP',beta0=None,verbose=False):
    if mtype=='hetGP':
        model = hetGP()
    else:
        model = homGP()
    known = {}
    if beta0 is not None:
        known['beta0'] = beta0
    model.mle(
        X=X,Z=Z,
        covtype=covtype,maxit=20,
        lower=[0.5 for _ in range(X.shape[1])],
        upper=[5 for _ in range(X.shape[1])],
        known=known
    )
    
    if verbose:
        msg = f"Model {mtype}, {covtype} ran in {round(model.time,2)}"
        print(msg)
    Xp = model.X0 # avoid replicates
    preds = model.predict(x=Xp,xprime=Xp)
    return np.allclose(preds['sd2'],np.diag(preds['cov']))

def test_mcycle(verbose=False):
    X, Z = m['times'], m['accel']
    beta0 = np.mean(Z)
    for beta in [None,beta0]:
        for mtype in ['hetGP','homGP']:
            for ctype in ['Gaussian','Matern3_2','Matern5_2']:
                res = _compare(X,Z,covtype=ctype,mtype=mtype,verbose=verbose,beta0=beta)
                assert res

def test_SIR(verbose=False):
    sir = loadmat('tests/data/SIR.mat') #2D example
    X, Z = sir['X'], sir['Y']
    beta0 = np.mean(Z)
    for beta in [None,beta0]:
        for mtype in ['hetGP','homGP']:
            for ctype in ['Matern3_2','Matern5_2']: # no gaussian for SIR
                res = _compare(X,Z,covtype=ctype,mtype=mtype,verbose=verbose,beta0=beta)
                assert res


if __name__ == "__main__":
    verbose = True
    test_mcycle(verbose)
    test_SIR(verbose)
