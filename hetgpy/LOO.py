import numpy as np
from hetgpy import hetGP, homGP

def LOO_preds(model, ids = None):
    r'''
    Provide leave one out predictions, e.g., for model testing and diagnostics.  This is used in the method plot available on GP and TP models.
    
    Parameters
    ----------
    
    model: hetgpy.homGP.homGP or hetgpy.hetGP.hetGP
        hetGPy model to evaluate. TPs not considered at this point.
    ids: vector of indices of the unique design point considered (defaults to all) 
    
    Returns
    -------
    
    dict with mean and variance predictions at x_i assuming this point has not been evaluated
    
    Notes
    -----
    For TP models, `psi` is considered fixed

    References
    ----------
    O. Dubrule (1983), Cross validation of Kriging in a unique neighborhood, Mathematical Geology 15, 687--699. \cr \cr
    F. Bachoc (2013), Cross Validation and Maximum Likelihood estimations of hyper-parameters of Gaussian processes 
    
    Examples
    --------
    >>> from hetgpy import homGP
    >>> from hetgpy.LOO import LOO_preds
    >>> from hetgpy.find_reps import find_reps
    >>> from hetgpy.example_data import mcycle
    >>> import numpy as np
    >>> m = mcycle()
    >>> X, Z = m['times'], m['accel']
    >>> model = homGP()
    >>> model.mle(X = X, Z = Z, lower = [0.1], upper = [10],
    >>>          covtype = "Matern5_2", known = dict(beta0 = 0))
    >>> LOO_p = LOO_preds(model)
    >>> d = find_reps(X,Z)
    >>> LOO_ref = np.zeros(shape=(d['X0'].shape[0],2))
    >>> rows = np.arange(d['X0'].shape[0])
    >>> for i in rows:
    >>>     idxs = np.delete(rows,i)
    >>>     model_i = homGP()
    >>>     model_i.mle(X = dict(X0 = d['X0'][idxs,:], Z0 = d['Z0'][idxs],
    >>>                 mult = d['mult'][idxs]), 
    >>>                 Z = np.concatenate([d['Zlist'][j] for j in idxs]),
    >>>                        lower = [0.1], upper = [50], covtype = "Matern5_2",
    >>>                        known = dict(theta = model.theta, g = model.g,
    >>>                                    beta0 = 0))
    >>>     model_i.nu_hat = model.nu_hat
    >>>     p_i = model_i.predict(d['X0'][i:i+1,:])
    >>>     LOO_ref[i,:] = np.array([p_i['mean'], p_i['sd2']]).squeeze()
    >>> print(np.ptp(LOO_ref[:,0] - LOO_p['mean']))
    >>> print(np.ptp(LOO_ref[:,1] - LOO_p['sd2']))
    >>> model.plot()
    '''
    if ids is None: ids = np.arange(model.X0.shape[0])
    
    if model.trendtype is not None and model.trendtype == "OK":
        model.Ki = model.Ki - model.Ki.sum(axis=0).reshape(-1,1) @ model.Ki.sum(axis=0).reshape(1,-1) / model.Ki.sum()
    
    if model.__class__.__name__ == 'homGP':
        sds = model.nu_hat * (1/np.diag(model.Ki)[ids] - model.g/model.mult[ids])
    
    if model.__class__.__name__ == 'hetGP':
        sds = model.nu_hat * (1/np.diag(model.Ki)[ids] - model.Lambda[ids]/model.mult[ids])
    
    
    if model.__class__.__name__ == 'homTP':
        sds = (1/np.diag(model.Ki)[ids] - model.g/model.mult[ids])
        # TP correction
        sds = (model.nu + model.psi - 2) / (model.nu + len(model.Z) - model.mult[ids] - 2) * sds


    if model.__class__.__name__ == 'hetTP':
        sds = (1/np.diag(model.Ki)[ids] - model.Lambda[ids]/model.mult[ids])
        # TP correction
        sds = (model.nu + model.psi - 2) / (model.nu + len(model.Z) - model.mult[ids] - 2) * sds
    
    
    ys = model.Z0[ids] - (model.Ki @ (model.Z0 - model.beta0))[ids]/np.diag(model.Ki)[ids]
    return(dict(mean = ys, sd2 = sds))
