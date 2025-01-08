from hetgpy.utils import duplicated
import numpy as np
from tests.utils import read_yaml
R_data = read_yaml('tests/R/results/test_duplicated.yaml')
for key in R_data:
    R_data[key] = np.array(R_data[key])
def test_duplicates():
    # from help(duplicated) in R
    x = R_data['x']
    test1 = (duplicated(x) == R_data['D1'].astype(bool)).all()
    test2 = (duplicated(x,fromLast = True) == R_data['D1_L'].astype(bool)).all()
    assert test1
    assert test2

    # 2d example
    X = np.array(R_data['X']).reshape(3,3).T
    R_data['D2'] = np.array(R_data['D2'])
    R_data['D2_L'] = np.array(R_data['D2_L'])
    test3 = (duplicated(X) == R_data['D2'].astype(bool)).all()
    test4 = (duplicated(X,fromLast=True) == R_data['D2_L'].astype(bool)).all()
    assert test3
    assert test4