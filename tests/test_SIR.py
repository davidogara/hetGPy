from hetgpy.test_functions import sirEval
import numpy as np
from rpy2.robjects import r
def test_SIR():

    x = np.array([0.1,0.1]).reshape(1,2)
    y = sirEval(x)
    return
def test_SIR_25():
    '''Run the SIR eval model 25 times and compute the mean'''
    r('''
    library(hetGP)
    n = 25
    x = matrix(c(0.1,0.1),ncol=2,nrow=1) 
    x = matrix(rep(x,10),nrow=10)
    yout =  mean(apply(x,1,sirEval))
    ''')
    n = 25
    ys = np.zeros(shape = n)
    x = np.array([0.1,0.1]).reshape(1,2)
    for i in range(n):
        ys[i] = sirEval(x,seed=i)
    np.allclose(ys.mean(),np.array(r('yout')))

if __name__ == "__main__":
    test_SIR_25()