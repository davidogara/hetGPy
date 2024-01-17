# prediction module
from hetgpy.homGP import homGP
from hetgpy.hetGP import hetGP

def predict(object, x,**kwargs):
    if object['class'] == 'homGP':
        return homGP().predict_hom_GP(object,x,**kwargs)
    if object['class'] == 'hetGP':
        return hetGP().predict_het_GP(object,x,**kwargs)