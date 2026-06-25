'''
Tests that sd2var is properly returned when `noise_var=True` when using hetGP.predict

closes https://github.com/davidogara/hetGPy/issues/39 (thanks to Marie Cloet)
'''
from hetgpy import hetGP
from hetgpy.example_data import mcycle
import numpy as np

m = mcycle()
X,Y = m['times'], m['accel']

# predictions
Xp = np.linspace(X.min(),X.max(),500).reshape(-1,1)
def test_sd2var():
    '''
    Test hetGP.predict with sd2var = True
    '''
    GP = hetGP()
    GP.mle(X=X,Z=Y,covtype='Matern5_2')
    
    preds = GP.predict(x=Xp,noise_var=True)

if __name__ == "__main__":
    with np.errstate(all='ignore'):
        test_sd2var()
