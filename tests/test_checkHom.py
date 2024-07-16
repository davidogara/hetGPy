from hetgpy import hetGP, homGP
import numpy as np
def test_hom():
    x = np.linspace(0,2*np.pi,20).reshape(-1,1)
    y = np.sin(x)

    model = hetGP()
    model.mle(
        X = x,
        Z = y,
        covtype="Gaussian",
        lower = np.array([1e-3]),
        upper = np.array([10.0]),
        known = {'g':1e-8,'theta':np.array([5.0])}

    )
    assert type(model)==homGP

if __name__ == "__main__":
    test_hom()