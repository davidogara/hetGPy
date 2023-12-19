import numpy as np
from hetgpy.find_reps import find_reps
from hetgpy.covariance_functions import euclidean_dist
def auto_bounds(X, min_cor = 0.01, max_cor = 0.5, covtype = "Gaussian", p = 0.05):

    Xsc = find_reps(X,np.repeat(1,X.shape[0]), rescale=True) # rescaled distances
    dists = euclidean_dist(Xsc['X0'],Xsc['X0'])
    repr_low_dist = np.quantile(dists[np.tril(dists,k=-1)>0], q = p)
    repr_lar_dist = np.quantile(dists[np.tril(dists,k=-1)>0], q = 1-p)
    if covtype == "Gaussian":
        theta_min = - repr_low_dist / np.log(min_cor)
        theta_max = - repr_lar_dist / np.log(max_cor)
        return dict(lower = theta_min * (Xsc['inputBounds'][1,:] - Xsc['inputBounds'][0,:])**2,
                    upper = theta_max * (Xsc['inputBounds'][1,:] - Xsc['inputBounds'][0,:])**2)
        
    
    else:
        raise NotImplementedError(f"{covtype} not implemented yet")