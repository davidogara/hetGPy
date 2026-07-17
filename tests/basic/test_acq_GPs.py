'''
Tests adding the acquisition functions to the GPs
'''
from hetgpy import hetGP, homGP, crnGP
from hetgpy.example_data import mcycle
import numpy as np
from hetgpy.optim import crit_EI
m = mcycle()
X, Y = m['times'], m['accel']
Xg = np.linspace(X.min(), X.max(), 200)

# fit GPs
het = hetGP()
hom = homGP()
for model in [het,hom]:
    model.mle(X=X, Z = Y, covtype='Matern5_2')

def test_EI_het():
    GP = het
    EI_native = crit_EI(x = Xg, model = GP)
    EI_from_GP = GP.crit_EI(x=Xg)
    assert np.allclose(EI_native, EI_from_GP)

def test_EI_hom():
    GP = hom
    EI_native = crit_EI(x = Xg, model = GP)
    EI_from_GP = GP.crit_EI(x=Xg)
    assert np.allclose(EI_native, EI_from_GP)