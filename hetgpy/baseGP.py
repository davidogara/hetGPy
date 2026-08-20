'''
Defines the basics of the homGP, hetGP, crnGP
'''
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
class GP(ABC):
    def __init__(self):
        pass
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self,item,value):
        self.__dict__[item] = value
    def get(self,key,default=None):
        r'''
        General `get` item (retrives key from self.__dict__) with optional default
        '''
        return self.__dict__.get(key,default)
    @abstractmethod
    def mle(self,X: ArrayLike,
             Z: ArrayLike, 
            covtype: str = "Gaussian"):
        '''Maximum Likelihood Estimation'''
        pass
    @abstractmethod
    def predict(self, x: ArrayLike) -> dict:
        '''
        Predict method, must minimally supply `x` to make predictions on new data
        '''
        pass
    # -- acquisition functions
    ## from optim.py
    def crit_EI(self,x,cst = None, preds = None):
        from hetgpy.optim import crit_EI
        return crit_EI(x = x, model = self, cst = cst, preds = preds)
    def deriv_crit_EI(self,x, cst = None, preds = None):
        from hetgpy.optim import deriv_crit_EI
        return deriv_crit_EI(x = x,model = self,cst = cst,preds = preds)
    def predict_gr(self,x):
        from hetgpy.optim import predict_gr
        return predict_gr(model = self, x = x)
    def crit_qEI(self,x, cst = None, preds = None):
        from hetgpy.optim import crit_qEI
        return crit_qEI(x = x, model = self, cst = cst, preds = preds)
    def crit_search(self, crit, replicate = False, Xcand = None, 
                        control = dict(tol_dist = 1e-6, tol_diff = 1e-6,
                                       multi_start = 20,maxit = 100,
                                       maximin = True, Xstart = None), 
                        seed = None,ncores = 1,**kwargs):
        from hetgpy.optim import crit_search
        return crit_search(model = self,crit = crit, replicate = replicate, Xcand = Xcand,
                           control = control, seed = seed, ncores = ncores, **kwargs)
    def crit_optim(self, crit, h = 2, Xcand = None, 
               control = dict(multi_start = 10, maxit = 100),
                 seed = None, ncores = 1, **kwargs):
        from hetgpy.optim import crit_optim
        return crit_optim(model = self, crit = crit, h = h, Xcand = Xcand, control = control,
                          seed = seed, ncores = ncores, **kwargs)
    def crit_logEI(self, x, cst = None, preds = None):
        from hetgpy.optim import crit_logEI
        return crit_logEI(model = self, x = x, cst = cst, preds = preds)
    def deriv_crit_logEI(self, x, cst = None, preds = None):
        from hetgpy.optim import deriv_crit_logEI
        return deriv_crit_logEI(model = self, x = x, cst = cst, preds = preds)
    ## from IMSE.py
    def crit_IMSPE(self,x=None, id = None, Wijs = None):
        from hetgpy.IMSE import crit_IMSPE
        return crit_IMSPE(model = self, x = x, id = id, Wijs = Wijs)
    def deriv_crit_IMSPE(self,x, id = None, Wijs = None):
        from hetgpy.IMSE import deriv_crit_IMSPE
        return deriv_crit_IMSPE(model = self, x = x, id = id, Wijs = Wijs)
    ## from contour.py
    def crit_MEE(self,x, thres = 0, preds = None):
        from hetgpy.contour import crit_MEE
        return crit_MEE(model = self, x = x, thres = thres, preds = preds)
    def crit_cSUR(self,x, thres = 0, preds = None):
        from hetgpy.contour import crit_cSUR
        return crit_cSUR(model = self, x = x, thres = thres, preds = preds)
    def crit_ICU(self,x, Xref, thres = 0, w = None, preds = None, kxprime = None):
        from hetgpy.contour import crit_ICU
        return crit_ICU(model = self, x = x, thres = thres, Xref = Xref, w = w, preds = preds, kxprime = kxprime)
    def crit_tMSE(self, x, thres = 0, preds = None, seps = 0.05):
        from hetgpy.contour import crit_tMSE
        return crit_tMSE(model = self, x = x, thres = thres, preds = preds, seps = seps)
    def crit_MCU(self,x, thres = 0, gamma = 2, preds = None):
        from hetgpy.contour import crit_MCU
        return crit_MCU(model = self, x = x, thres = thres, gamma = gamma, preds = preds)
    def crit_TS(self,x,n_TS = 1, rng = None, check_PSD = True):
        from hetgpy.optim import crit_TS
        return crit_TS(model = self, x = x, n_TS = n_TS, rng = rng, check_PSD = check_PSD)
    def crit_BAPE(self,x,log=True):
        from hetgpy.optim import crit_BAPE
        return crit_BAPE(model = self, x = x, log = log)
    
    

    
    

    
