'''Score function'''
import numpy as np
def score(model,Xtest,Ztest,return_rmse = False):
    p = model.predict(Xtest)
    ps2 = p['sd2'] + p['nugs']
    se = (Ztest - p['mean'])**2
    sc = -se/ps2 - np.log(ps2)
    out = {'score':sc.mean()}
    if return_rmse:
        rmse = (se.mean())**0.5
        out['rmse'] = rmse
    return out    