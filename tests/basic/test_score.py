'''Check score functions for GPs'''
import yaml
from hetgpy.example_data import mcycle
from hetgpy import homGP, hetGP
import numpy as np
m = mcycle()
X = m['times']
Z = m['accel']

def read_yaml(fp):
    with open(fp,'r') as stream:
        out = yaml.safe_load(stream)
    return out

def test_hom():
    GP = homGP()
    pars = read_yaml('tests/R/results/homGP_score.yaml')

    GP.mle(
        X = X, Z = Z, covtype="Matern5_2", 
        known = {
            'theta':np.array(pars['theta']).reshape(-1),
            'beta0':pars['beta0'],
            'g': pars['g']
        }
    )
    score = GP.score(Xtest=X,Ztest = Z, return_rmse = False)
    np.allclose(score['score'],pars['score'])

def test_het():
    GP = hetGP()
    pars = read_yaml('tests/R/results/hetGP_score.yaml')

    GP.mle(
        X = X, Z = Z, covtype="Matern5_2", 

        known = {
            'theta':np.array(pars['theta']).reshape(-1),
            'beta0':pars['beta0'],
            'g':pars['g'],
            'Delta': np.array(pars['Delta']),
            'Lambda': np.array(pars['Lambda'])
        },
        noiseControl=dict(g_min = np.finfo(float).eps,g_max=100)
    )
    score = GP.score(Xtest=X,Ztest = Z, return_rmse = False)
    np.allclose(score['score'],pars['score'])
    