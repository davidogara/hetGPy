import numpy as np
from hetgpy import hetGP
from hetgpy.example_data import mcycle
from rpy2.robjects import r
m = mcycle()

    
def test():
    '''
    Test performance of `trendtype`="SK"
    Version 1.0.3 and before should "fail" (thanks to Ozge Surer, Miami Ohio)
    Version 1.0.4+ should "pass"
    '''

    model = hetGP()
    times, accel = m['times'], m['accel']
    # simple kriging
    Xgrid = np.linspace(times.min(),times.max(),100).reshape(-1,1)
    known = dict(beta0=1,theta=np.array([10]),g=1e-3)
    model.mle(X=times,Z=accel,covtype='Matern5_2',known=known,maxit=5)
    
    # hetGP SK
    RStr = '''
    library(hetGP)
    library(MASS)
    X = matrix(mcycle$times,ncol=1)
    Delta = rep(-5,94)
    Delta[94] = -2
    Xgrid = matrix(seq(from=min(X),max(X),length.out=100),ncol=1)
    Z = mcycle$accel
    model = mleHetGP(X,Z,covtype="Matern5_2",known=list(beta0=1,theta=10,g=1e-3),maxit=5)
    preds = predict(model,Xgrid)
    '''
    r(RStr)
    # now "align" models by assigning core attributes from hetGP class
    model.nu_hat = np.array(r('model$nu_hat')) 
    #model.Delta = np.array(r('model$Delta'))
    #model.Lambda = np.array(r('model$Lamda'))
    model.Ki = np.array(r('model$Ki'))
    model.C = np.array(r('model$C'))
    #model.Cg = np.array(r('model$Cg'))
    preds = model.predict(Xgrid)

    # compare
    predsR = dict(
        sd2 = np.array(r('preds$sd2')),
    )
    assert np.allclose(preds['sd2'],predsR['sd2'])

def test_OK():
    '''
    Same test as above but tests trendtype="OK" (ordinary kriging)
    '''

    model = hetGP()
    times, accel = m['times'], m['accel']
    # simple kriging
    Xgrid = np.linspace(times.min(),times.max(),100).reshape(-1,1)
    known = dict(theta=np.array([10]),g=1e-3)
    model.mle(X=times,Z=accel,covtype='Matern5_2',known=known,maxit=5)
    
    # hetGP SK
    RStr = '''
    library(hetGP)
    library(MASS)
    X = matrix(mcycle$times,ncol=1)
    Delta = rep(-5,94)
    Delta[94] = -2
    Xgrid = matrix(seq(from=min(X),max(X),length.out=100),ncol=1)
    Z = mcycle$accel
    model = mleHetGP(X,Z,covtype="Matern5_2",known=list(theta=10,g=1e-3),maxit=5)
    preds = predict(model,Xgrid)
    '''
    r(RStr)
    # now "align" models by assigning core attributes from hetGP class
    model.nu_hat = np.array(r('model$nu_hat')) 
    #model.Delta = np.array(r('model$Delta'))
    #model.Lambda = np.array(r('model$Lamda'))
    model.Ki = np.array(r('model$Ki'))
    model.C = np.array(r('model$C'))
    #model.Cg = np.array(r('model$Cg'))
    preds = model.predict(Xgrid)

    # compare
    predsR = dict(
        sd2 = np.array(r('preds$sd2')),
    )
    assert np.allclose(preds['sd2'],predsR['sd2'])

if __name__ == "__main__":
    test()
