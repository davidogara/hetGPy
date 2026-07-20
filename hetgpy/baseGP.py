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
    def crit_EI(self,x,cst = None, preds = None):
        from hetgpy.optim import crit_EI
        return crit_EI(x = x, model = self, cst = cst, preds = preds)
