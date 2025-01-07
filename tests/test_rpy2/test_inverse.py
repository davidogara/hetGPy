import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
import numpy as np
from scipy.linalg.lapack import dtrtri
from time import time
def test_inverse():
    '''
    Tests fast inversion of a symmetric matrix
    '''
    X = np.array(
        [4,12,-16,12,37,-43,-16,-43,98]
    ).reshape(3,3)

    # in R
    RStr = '''
    X = c(4,12,-16,12,37,-43,-16,-43,98) |> matrix(nrow=3,byrow=T)
    out = chol2inv(chol(X))
    '''
    r_out = robjects.r(RStr)

    tmp = dtrtri(np.linalg.cholesky(X).T)[0]
    X_inv = tmp @ tmp.T
    sol = np.array(robjects.r['out'])
    assert np.allclose(X_inv,sol)


def time_fast_inverse_big(msize=500):
    # make a big symmetric matrix
    X = np.random.default_rng(42).integers(low=1,high=50,size=(msize,msize))
    X = X @ X.T # symmetric
    tic = time()
    tmp = dtrtri(np.linalg.cholesky(X).T)[0]
    X_inv = tmp @ tmp.T
    
    py_time = time()-tic
    print(f"Python computed {msize}x{msize} inverse in: {round(py_time,5)} seconds")
    
    # now do R
    RStr = '''
    func = function(X){
        tic = proc.time()[3]
        X_inv = chol2inv(chol(X))
        toc = round(proc.time()[3] - tic,5) 
        toc = toc |> unlist() |> as.vector()
        return(toc)
    }
    '''
    func = SignatureTranslatedAnonymousPackage(RStr,"func")
    X_for_r = robjects.r.matrix(X, byrow=True,nrow=msize,ncol=msize)
    
    r_time = float(func.func(X_for_r)[0])
    assert py_time < r_time
if __name__ == "__main__":
    for test in (100,200,300):
        time_fast_inverse_big(test)