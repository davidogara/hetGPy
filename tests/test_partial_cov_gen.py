from hetgpy.covariance_functions import partial_cov_gen
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import numpy as np
from hetgpy.example_data import mcycle
from hetgpy.find_reps import find_reps

from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

# Create a converter that starts with rpy2's default converter
# to which the numpy conversion rules are added.
np_cv_rules = default_converter + numpy2ri.converter
hetGP_R = importr('hetGP')
m = mcycle()

def test_partial_cov_gen(ctype='Matern3_2'):
    X,Z = m['times'], m['accel']
    prep = find_reps(X,Z)
    args = ('theta_k','k_theta_g')
    args_to_pop = {'theta_k':('k_theta_g','i1','i2'),
                   'k_theta_g':('i1','i2')}
    X0 = prep['X0']
    for arg in args:
        kw_base = dict(
            X1=X0,
            theta = np.array([2.0]),
            k_theta_g = np.array([1.5]),
            arg=arg,
            type=ctype,
            i1 = 2,
            i2 = 1
        )
        kw = kw_base.copy()
        for item in args_to_pop[arg]:
            kw.pop(item)
        out_1 = partial_cov_gen(**kw)
        with np_cv_rules.context():
            Rkw = kw.copy()
            for key in ('theta','k_theta_g'):
                if key in Rkw.keys():
                    Rkw[key] = float(kw[key].squeeze())
            out_2 = hetGP_R.partial_cov_gen(**Rkw)
        assert np.allclose(out_1,out_2)
    return

if __name__ == "__main__":
    test_partial_cov_gen('Matern3_2')

