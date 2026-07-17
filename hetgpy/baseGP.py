'''
Defines the basics of the homGP, hetGP, crnGP
'''
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
class GP(ABC):
    def __init__(self):
        pass
    def crit_EI(self,x,cst = None, preds = None):
        from hetgpy.optim import crit_EI
        return crit_EI(x = x, model = self, cst = cst, preds = preds)
    @abstractmethod
    def mle(self,X: ArrayLike,
             Z: ArrayLike, 
            covtype: str = "Gaussian"):
        '''Maximum Likelihood Estimation'''
        pass