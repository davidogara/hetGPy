from hetgpy import homGP
from hetgpy.homGP_jax import homGP_jax
import jax.numpy as np
from hetgpy.example_data import mcycle
from hetgpy.find_reps import find_reps
from scipy.io import loadmat
m = mcycle()

def SIR():
    m = loadmat('tests/data/SIR.mat')
    return m['X'], m['Y']

def compare(data,ctype):
    X,Z = data
    prep = find_reps(X,Z)
    for key in ('X0','Z','mult','Z0'):
        prep[key] = np.array(prep[key])
    l = [1 for _ in range(X.shape[1])]
    u = [10 for _ in range(X.shape[1])]
    kwargs = dict(lower=l,upper=u,covtype=ctype)
    model1 = homGP().mle(X=X,Z=Z, **kwargs)
    model2 = homGP_jax().mle_jax(X=prep,Z=Z,**kwargs)
    print(f'Testing {ctype}')
    assert np.allclose(model1.ll,model2.ll)
    assert np.allclose(model1.theta,model2.theta,atol=1e-3)
    assert np.allclose(model1.g,model2.g,atol=1e-3)


def test_mcycle():
    X,Z = m['times'], m['accel']
    for ctype in ['Gaussian','Matern5_2','Matern3_2']:
        compare(data = (X,Z),ctype=ctype)
    
    return

def test_SIR():
    X,Z = SIR()
    for ctype in ['Matern5_2','Matern3_2']:
        compare(data = (X,Z),ctype=ctype)

if __name__ == "__main__":
    test_mcycle()
