import sys
sys.path.append('./')
from hetgpy import homGP
from hetgpy.optim import crit_EI
import numpy as np
from rpy2.robjects import r
def test_EI():
    '''
    Expected Improvement example testing whether, given the same hyperparameters, homGP (R) and homGP (python)
    lead to the same function acquisitions

    Problem is inspired from: 
        *   Jones, DR, M Schonlau, and WJ Welch. 1998. “Efficient Global Optimization of Expensive Black-Box Functions.” Journal of Global Optimization 13 (4): 455–92.
    
    and adapted from Gramacy (2020), Ch. 7 (Figure 7.6)
    
    '''
    X = np.array([1, 2, 3, 4, 12]).reshape(-1,1)
    Y = np.array([0, -1.75, -2, -0.5, 5])

    model = homGP()
    model.mle(X=X,
        Z=Y,
        covtype='Gaussian',
        known={'theta':np.array([10.0]),'g':1e-8}
    )
    Xgrid = np.linspace(1,12,20).reshape(-1,1)
    
    EIs = crit_EI(x=Xgrid,model=model)
    acq = Xgrid[EIs.argmax(keepdims=True),:]
    r('''
    library(hetGP)
    X <- matrix(c(1, 2, 3, 4, 12))
    Y  <- c(0, -1.75, -2, -0.5, 5)
    Xgrid <- matrix(seq(1,12,length.out=20))
    model <- mleHomGP(X=X,Z=Y,covtype='Gaussian',known=list(theta=10,g=1e-8))
    EIs <- crit_EI(model=model,x=Xgrid)
    acq <- Xgrid[which.max(EIs)]  
    ''')

    assert(acq==np.array(r('acq')))

    # now update
    model.update(Xnew=acq, Znew = model.predict(acq)['mean'])
    r('''
    model <- update(object=model,
      Xnew=acq,
      Znew=predict(object=model,acq)$mean)
    EIs2 <- crit_EI(model=model,x=Xgrid)
    acq2 <- Xgrid[which.max(EIs2)]  
    ''')

    EIs2 = crit_EI(x=Xgrid,model=model)
    acq2 = Xgrid[EIs2.argmax(keepdims=True),:]

    assert np.allclose(acq2,np.array(r('acq2')),atol=1e-3)

if __name__ == "__main__":
    test_EI()
    

