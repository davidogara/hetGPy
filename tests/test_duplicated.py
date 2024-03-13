from hetgpy.utils import duplicated
import numpy as np
from rpy2.robjects import r

def test_duplicates():
    # from help(duplicated) in R
    r("x <- c(9:20, 1:5, 3:7, 0:8)")
    x = np.array(r("x"))
    test1 = (duplicated(x) == np.array(r("duplicated(x)")).astype(bool)).all()
    test2 = (duplicated(x,fromLast = True) == np.array(r("duplicated(x,fromLast = T)")).astype(bool)).all()
    assert test1
    assert test2

    # 2d example
    r('X <- matrix(c(c(1,2,3),c(1,2,3),c(4,5,6)),nrow=3,byrow=T)')
    X = np.array(r('X'))
    test3 = (duplicated(X) == np.array(r('duplicated(X)')).astype(bool)).all()
    test4 = (duplicated(X,fromLast=True) == np.array(r('duplicated(X,fromLast=T)')).astype(bool)).all()
    assert test3
    assert test4