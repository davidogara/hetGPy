import numpy as np
from scipy.stats import norm
from hetgpy.qEI import pnorm, dnorm

def compute_pnorm(loc,scale):
    x = np.linspace(-10,10,1000)
    scipy_compare = norm.cdf(x, loc=loc, scale = scale)
    out = np.zeros_like(x)
    for i in range(len(out)):
        out[i] = pnorm(x[i], loc, scale, 1,0)
    return np.allclose(out,scipy_compare)
def compute_dnorm(loc,scale):
    x = np.linspace(-10,10,1000)
    scipy_compare = norm.pdf(x, loc=loc, scale = scale)
    out = np.zeros_like(x)
    for i in range(len(out)):
        out[i] = dnorm(x[i], loc, scale, 0)
    return np.allclose(out,scipy_compare)

def test_pnorm():
    means = [-0.2, -0.1, 0.0, 0.1, 0.2]
    sds   = [0.25, 0.5, 0.75, 1.0, 5.0]
    checks = []
    for i in range(len(means)):
        check = compute_pnorm(means[i],sds[i])
        checks.append(check)
    assert all(checks)

def test_dnorm():
    means = [-0.2, -0.1, 0.0, 0.1, 0.2]
    sds   = [0.25, 0.5, 0.75, 1.0, 5.0]
    checks = []
    for i in range(len(means)):
        check = compute_dnorm(means[i],sds[i])
        checks.append(check)
    assert all(checks)
if __name__ == "__main__":
    test_pnorm()
    test_dnorm()
