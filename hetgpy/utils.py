import numpy as np

def cov_gen(X1, X2 = None, theta = None, type = None):
    if type=='Gaussian':
        dists  = cov_Gaussian(X1,X2,theta)
    else:
        raise ValueError(f"type: {type} not yet implemented")
    return dists

def euclidean_dist(X1,X2):
    dists = np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2 * X1.dot(X2.T) + np.sum(np.square(X2), axis=1)
    return dists

# numba this
def cov_Gaussian(X1, X2 = None, theta = None):
    '''
    https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized
    threeSums = np.sum(np.square(A)[:,np.newaxis,:], axis=2) - 2 * A.dot(B.T) + np.sum(np.square(B), axis=1)
    np.exp(-threeSums/theta)
    '''
    if len(theta)==1:
            X2 = X1 if X2 is None else X2
            return np.exp(-euclidean_dist(X1,X2)/theta)
    if X2 is None:
        A = X1 * np.repeat(1/np.sqrt(theta),np.repeat(X1.shape[0],len(theta)))
        return np.exp(-1.0*euclidean_dist(A,A))
    else:
        A = X1 * np.repeat(1/np.sqrt(theta),np.repeat(X1.shape[0],len(theta)))
        B = X2 * np.repeat(1/np.sqrt(theta),np.repeat(X2.shape[0],len(theta)))
        return np.exp(-1.0*euclidean_dist(A,B))
