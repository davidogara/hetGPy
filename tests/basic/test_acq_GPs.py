'''
Tests adding the acquisition functions to the GPs
'''
from hetgpy import hetGP, homGP, crnGP
from hetgpy.example_data import mcycle
import numpy as np
m = mcycle()
X, Y = m['times'], m['accel']
Xg = np.linspace(X.min(), X.max(), 50).reshape(-1,1)

# fit GPs
het = hetGP()
hom = homGP()
for model in [het,hom]:
    model.mle(X=X, Z = Y, covtype='Matern5_2')
GPs = [het,hom]
## -- from optim.py
def test_EI():
    from hetgpy.optim import crit_EI
    for GP in GPs:
        crit_native = crit_EI(x = Xg, model = GP)
        crit_GP = GP.crit_EI(x=Xg)
        assert np.allclose(crit_native, crit_GP)
def test_qEI():
    from hetgpy.optim import crit_qEI
    for GP in GPs:
        crit_native = crit_qEI(x = Xg, model = GP)
        crit_GP = GP.crit_qEI(x=Xg)
        assert np.allclose(crit_native, crit_GP)

def test_logEI():
    from hetgpy.optim import crit_logEI
    for GP in GPs:
        crit_native = crit_logEI(x = Xg, model = GP)
        crit_GP = GP.crit_logEI(x=Xg)
        assert np.allclose(crit_native, crit_GP)

## -- from IMSE.py
def test_IMSPE():
    from hetgpy.IMSE import crit_IMSPE
    xg = Xg[20].reshape(-1,1)
    for GP in GPs:
        crit_native = crit_IMSPE(x = xg, model = GP)
        crit_GP = GP.crit_IMSPE(x = xg)
        assert np.allclose(crit_native, crit_GP)
def test_MEE():
    from hetgpy.optim import crit_MEE
    for GP in GPs:
        crit_native = crit_MEE(x = Xg, model = GP)
        crit_GP = GP.crit_MEE(x = Xg)
        assert np.allclose(crit_native, crit_GP)
def test_cSUR():
    from hetgpy.optim import crit_cSUR
    for GP in GPs:
        crit_native = crit_cSUR(x = Xg, model = GP)
        crit_GP = GP.crit_cSUR(x = Xg)
        assert np.allclose(crit_native, crit_GP)
def test_ICU():
    from hetgpy.optim import crit_ICU
    for GP in GPs:
        crit_native = crit_ICU(x = Xg, model = GP,Xref=Xg)
        crit_GP = GP.crit_ICU(x = Xg,Xref=Xg)
        assert np.allclose(crit_native, crit_GP)
def test_tMSE():
    from hetgpy.optim import crit_tMSE
    for GP in GPs:
        crit_native = crit_tMSE(x = Xg, model = GP)
        crit_GP = GP.crit_tMSE(x = Xg)
        assert np.allclose(crit_native, crit_GP)
def test_MCU():
    from hetgpy.optim import crit_MCU
    for GP in GPs:
        crit_native = crit_MCU(x = Xg, model = GP)
        crit_GP = GP.crit_MCU(x = Xg)
        assert np.allclose(crit_native, crit_GP)
if __name__ == "__main__":
    test_ICU()






