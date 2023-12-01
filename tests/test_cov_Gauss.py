import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from hetgpy.covariance_functions import *

def test_one_input():

    x = pd.read_csv('tests/data/test_mycle_cov_gen.csv').values

    one_input = pd.read_csv('tests/data/mcycle_cov_Gauss_theta_1.csv',header=None).values

    out1 = cov_Gaussian(x,x,theta = np.array([1]))
    assert np.allclose(out1,one_input)

def test_multiple_inputs():
    x = pd.read_csv('tests/data/test_mycle_cov_gen.csv').values
    multi_input = pd.read_csv('tests/data/mcycle_cov_Gauss_theta_1_2_3.csv',header=None).values

    out1 = cov_Gaussian(x,x,theta = np.array([1,2,3]))
    assert np.allclose(out1,multi_input)
    