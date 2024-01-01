import numpy as np

def cov_gen(X1, X2 = None, theta = None, type = None):
    if type=='Gaussian':
        dists  = cov_Gaussian(X1,X2,theta)
    else:
        raise ValueError(f"type: {type} not yet implemented")
    return dists

def euclidean_dist(X1,X2):
    dists = np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * X1.dot(X2.T) + np.sum(np.square(X2), axis=1)
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
        A = X1 * 1.0 / np.sqrt(theta)
        return np.exp(-1.0*euclidean_dist(A,A))
    else:
        A = X1 * 1.0 / np.sqrt(theta)
        B = X2 * 1.0 / np.sqrt(theta)
        return np.exp(-1.0*euclidean_dist(A,B))

def partial_d_C_Gaussian_dtheta_k(X1,theta):
    ## Partial derivative of the covariance matrix with respect to theta[k] (to be multiplied by the covariance matrix)
    return euclidean_dist(X1,X1)/theta**2

def partial_d_k_Gaussian_dtheta_k(X1,X2,theta):
    ## Partial derivative of the covariance vector with respect to theta[k] (to be multiplied by the covariance vector)
    return euclidean_dist(X1,X2)/theta**2

def partial_d_Cg_Gaussian_d_k_theta_g(X1,theta,k_theta_g):
    ## Partial derivative of the covariance matrix of the noise process with respect to k_theta_g (to be multiplied by the covariance matrix)
    ## 1-dimensional/isotropic case
    if len(theta) == 1:
        return(euclidean_dist(X1,X1)/(theta*k_theta_g**2))
    xx1 = X1 * 1 / np.sqrt(theta)
    return euclidean_dist(xx1, xx1)/k_theta_g**2

def partial_d_kg_Gaussian_d_k_theta_g(X1,X2,theta, k_theta_g):
    ## Partial derivative of the covariance vector of the noise process with respect to k_theta_g (to be multiplied by the covariance vector)
    ## 1-dimensional/isotropic case
    if len(theta) == 1:
        return(euclidean_dist(X1,X2)/(theta*k_theta_g**2))
    xx1 = X1 * 1 / np.sqrt(theta)
    xx2 = X2 * 1 / np.sqrt(theta)
    return euclidean_dist(xx1, xx2)/k_theta_g**2

def partial_d_C_Gaussian_dX_i_j(X1,theta,i1,i2):
    ## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
    ## @param i1 row
    ## @param i2 column
    tmp = partial_d_dist_dX_i1_i2(X1, i1, i2)

    ## 1-dimensional/isotropic case
    if len(theta) == 1:
        return(tmp/theta)
    return(tmp / theta[i2])

def partial_d_k_Gaussian_dX_i_j(X1, X2, theta, i1, i2):
    ## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
    ## @param i1 row
    ## @param i2 column
    tmp = partial_d_dist_dX1_i1_i2_X2(X1, X2, i1, i2)
    ## 1-dimensional/isotropic case
    if len(theta) == 1:
        return(tmp/theta)
    return(tmp / theta[i2])

def partial_d_dist_dX_i1_i2(X1, i1, i2):
    nr = X1.shape[0]
    s = np.zeros((nr, nr))
    for i in range(nr):
        if(i != (i1 - 1)):
            s[i1 - 1, i] = s[i, i1 - 1] = -2 * (X1[i1-1, i2-1] - X1[i, i2-1])
    return s


def partial_d_dist_dX1_i1_i2_X2(X1, X2, i1, i2):
    s = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X2.shape[0]):
        s[i1 - 1, i] = -2 * (X1[i1-1, i2-1] - X2[i, i2-1])
    return s

def partial_cov_gen(X1,X2 = None, theta = None,k_theta_g = None, type = "Gaussian", arg = None, i1 = None, i2 = None):
    if X2 is None:
        if type== "Gaussian":
            if arg == "theta_k":
                return partial_d_C_Gaussian_dtheta_k(X1 = X1, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Gaussian_d_k_theta_g(X1 = X1, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Gaussian_dX_i_j(X1 = X1, theta = theta, i1 = i1, i2 = i2)
    else:
        if type == "Gaussian": 
            if arg == "theta_k":
                return partial_d_k_Gaussian_dtheta_k(X1 = X1, X2 = X2, theta = theta)
            if arg == "k_theta_g":
                return partial_d_kg_Gaussian_d_k_theta_g(X1 = X1, X2 = X2, theta = theta,k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_k_Gaussian_dX_i_j(X1 = X1, X2 = X2, theta = theta,i1 = i1, i2 = i2)