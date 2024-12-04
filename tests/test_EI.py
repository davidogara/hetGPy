import numpy as np
from hetgpy import hetGP
from hetgpy.optim import crit_EI
from hetgpy.example_data import mcycle as mcycle_data
from tests.utils import read_yaml
m = mcycle_data()
R_results = read_yaml('tests/R/results/test_EI.yaml')
def test_EI():
    X, Z = m['times'], m['accel']
    model = hetGP()
    model.mleHetGP(X = X, Z = Z)
    xgrid = np.linspace(0,60,301)
    EIs = crit_EI(model = model, x = xgrid)
    proposal = xgrid[EIs.argmax()]

    
    assert np.allclose(proposal, R_results['proposal'])
if __name__ == "__main__":
    test_EI()