'''
Test the auto_bounds functionality
'''

import sys
sys.path.append('../')
from hetgpy.auto_bounds import auto_bounds
import numpy as np
from rpy2.robjects import r

X = np.array([[ 1, 34],
       [30,  3],
       [45, 11],
       [13, 10],
       [17,  9]])
def prep_R():
    r('''
    X <- matrix(c(1, 34,
       30,  3,
       45, 11,
       13, 10,
       17,  9),byrow=T,nrow=5)
    library(hetGP)
    auto_bounds <- function(X, min_cor = 0.01, max_cor = 0.5, covtype = "Gaussian", p = 0.05){
    Xsc <- find_reps(X, rep(1, nrow(X)), rescale = T) # rescaled distances
    # sub in distance function
    dists <- plgp::distance(Xsc$X0) # find 2 closest points
    repr_low_dist <- quantile(x = dists[lower.tri(dists)], probs = p) # (quantile on squared Euclidean distances)
    repr_lar_dist <- quantile(x = dists[lower.tri(dists)], probs = 1-p)
  
    if(covtype == "Gaussian"){
    theta_min <- - repr_low_dist/log(min_cor)
    theta_max <- - repr_lar_dist/log(max_cor)
    return(list(lower = theta_min * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])^2,
                upper = theta_max * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])^2))
    }else{
    tmpfun <- function(theta, repr_dist, covtype, value){
      cov_gen(matrix(sqrt(repr_dist/ncol(X)), ncol = ncol(X)), matrix(0, ncol = ncol(X)), type = covtype, theta = theta) - value
    }
    theta_min <- uniroot(tmpfun, interval = c(sqrt(.Machine$double.eps), 100), covtype = covtype, value = min_cor, 
                         repr_dist = repr_low_dist, tol = sqrt(.Machine$double.eps))$root
    theta_max <- uniroot(tmpfun, interval = c(sqrt(.Machine$double.eps), 100), covtype = covtype, value = max_cor,
                         repr_dist = repr_lar_dist, tol = sqrt(.Machine$double.eps))$root
    return(list(lower = theta_min * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,]),
                upper = max(1, theta_max) * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])))
    }
    } 
    ''')


def compare_to_R(covtype):
    prep_R()
    r(f'out <- auto_bounds(X,covtype="{covtype}")')
    l = np.array(r('out$lower'))
    u = np.array(r('out$upper'))
    return dict(lower=l,upper=u)
def test_gauss():
    ctype ="Gaussian"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 
def test_matern3():
    ctype ="Matern3_2"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 
def test_matern5():
    ctype ="Matern5_2"
    out = auto_bounds(X,covtype=ctype)
    compare = compare_to_R(ctype)
    assert np.allclose(out['lower'],compare['lower'])
    assert np.allclose(out['upper'],compare['upper']) 

if __name__ == "__main__":
    prep_R()
    test_matern5()