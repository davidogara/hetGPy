'''
test the trace command
closes issue 32 (thank you to Marie Cloet for raising and the pull request)
'''
from hetgpy import hetGP
from hetgpy.example_data import mcycle

m = mcycle()
X, Y = m['times'], m['accel']


def test_trace_0():

    model = hetGP()
    model.mle(
        X = X, Z = Y, covtype = 'Matern5_2', settings = {'trace': 0}
    )

def test_trace_3():

    model = hetGP()
    model.mle(
        X = X, Z = Y, covtype = 'Matern5_2', settings = {'trace': 3}
    )

if __name__ == "__main__":
    import numpy as np
    with np.errstate(all='ignore'):
        np.set_printoptions(3)
        test_trace_3()