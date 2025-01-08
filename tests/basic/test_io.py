from hetgpy import hetGP, homGP
from hetgpy.io import save, load
from hetgpy.example_data import mcycle
import os

def test_save_and_load_homGP():
    m = mcycle()
    X, Z = m['times'], m['accel']

    model = homGP()
    model.mle(X=X,Z=Z)
    save(model,filename='hom.pkl')

    model2 = load('hom.pkl',rebuild=True)
    for key in ('g','theta'):
        assert model[key] == model2[key]
    assert (model['Ki'] == model2['Ki']).all()
    if os.path.exists('hom.pkl'):
        os.remove('hom.pkl')
    return

def test_save_and_load_hetGP():
    m = mcycle()
    X, Z = m['times'], m['accel']

    model = hetGP()
    model.mle(X=X,Z=Z)
    save(model,filename='het.pkl')

    model2 = load('het.pkl',rebuild=True)
    for key in ('g','theta'):
        assert model[key] == model2[key]
    assert (model['Ki'] == model2['Ki']).all()
    assert (model['Delta']==model2['Delta']).all()
    if os.path.exists('het.pkl'):
        os.remove('het.pkl')
    return

