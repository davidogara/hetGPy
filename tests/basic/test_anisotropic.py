from hetgpy import hetGP, homGP
from scipy.io import loadmat

def load_data():
    m = loadmat('tests/data/SIR.mat')
    return m['X'], m['Y']


def test_homGP_gauss():
    X,Y = load_data()
    model = homGP()
    model.mle(X,Y,lower=[1,1],upper=[10,10],covtype="Gaussian")
    assert len(model.theta)==X.shape[1]
    return

def test_hetGP_gauss():
    X,Y = load_data()
    model = hetGP()
    model.mle(X,Y,lower=[0.01,0.01],upper=[10,10],covtype="Gaussian")
    assert len(model.theta)==X.shape[1]
    return

def test_homGP_32():
    X,Y = load_data()
    model = homGP()
    model.mle(X,Y,lower=[1,1],upper=[10,10],covtype="Matern3_2")
    assert len(model.theta)==X.shape[1]
    return

def test_hetGP_32():
    X,Y = load_data()
    model = hetGP()
    model.mle(X,Y,lower=[1,1],upper=[10,10],covtype="Matern3_2")
    assert len(model.theta)==X.shape[1]
    return

def test_homGP_52():
    X,Y = load_data()
    model = homGP()
    model.mle(X,Y,lower=[1,1],upper=[10,10],covtype="Matern5_2")
    assert len(model.theta)==X.shape[1]
    return

def test_hetGP_52():
    X,Y = load_data()
    model = hetGP()
    model.mle(X,Y,lower=[1,1],upper=[10,10],covtype="Matern5_2")
    assert len(model.theta)==X.shape[1]
    return


if __name__ == "__main__":
    test_hetGP_52()