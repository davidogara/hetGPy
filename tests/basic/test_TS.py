'''
Test Thompson sampling
'''
from hetgpy import hetGP
from hetgpy.example_data import mcycle
import numpy as np
m = mcycle()
X, Y = m['times'], m['accel']
Xg = np.linspace(X.min(), X.max(), 50).reshape(-1,1)

# fit GPs
het = hetGP()
het.mle(X=X, Z = Y, covtype='Matern5_2')


def test_TS():
    return