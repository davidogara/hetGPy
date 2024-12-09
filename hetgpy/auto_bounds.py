import numpy as np
from hetgpy.find_reps import find_reps
from hetgpy.covariance_functions import euclidean_dist, cov_gen
from scipy.optimize import root_scalar
MACHINE_DOUBLE_EPS = np.sqrt(np.finfo(float).eps)
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
        def tmpfun(theta, repr_dist, covtype, value):
            x1 = np.sqrt([repr_dist/X.shape[1]]*X.shape[1]).reshape(-1,X.shape[1])
            x2 = np.zeros(shape=(1,X.shape[1]))
            if type(theta)!=np.ndarray:
                theta = np.array([theta])
                if len(theta) != X.shape[1]: theta = np.repeat(theta,X.shape[1])
            return (cov_gen(x1,x2,theta=theta,type=covtype) - value).squeeze()

        args_theta_min = (repr_low_dist,covtype,min_cor) 
        args_theta_max = (repr_lar_dist,covtype,max_cor)       
        theta_min = root_scalar(f=tmpfun, args = args_theta_min, 
                            bracket = [MACHINE_DOUBLE_EPS, 100],
                            xtol = MACHINE_DOUBLE_EPS).root
        theta_max = root_scalar(f=tmpfun, args = args_theta_max, 
                            bracket = [MACHINE_DOUBLE_EPS, 100],
                            xtol = MACHINE_DOUBLE_EPS).root
        return dict(lower = theta_min * (Xsc['inputBounds'][1,:] - Xsc['inputBounds'][0,:]),
                upper = max(1, theta_max) * (Xsc['inputBounds'][1,:] - Xsc['inputBounds'][0,]))
        