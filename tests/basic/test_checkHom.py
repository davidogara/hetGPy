from hetgpy import hetGP, homGP
import numpy as np
rand = np.random.default_rng(42)
def test_hom():
    x = np.linspace(0,2*np.pi,20).reshape(-1,1)
    y = np.cos(np.pi*x)
    y+= 0.05*rand.normal(size = y.shape)

    model = hetGP()
    model.mle(
        X = x,
        Z = y,
        covtype="Gaussian",
        known = {'g':1e-8,'theta':np.array([5.0])}

    )
    assert type(model)==homGP

if __name__ == "__main__":
    test_hom()