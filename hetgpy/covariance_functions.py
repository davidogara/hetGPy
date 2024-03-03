import numpy as np
from scipy.spatial.distance import cdist
from hetgpy.src import matern ## high performance c++ functions
from hetgpy.src import gauss

TYPE = type # for checking types

def cov_gen(X1, X2 = None, theta = None, type = None):
    if type=='Gaussian':
        dists  = cov_Gaussian(X1,X2,theta)
    elif type=="Matern5_2":
        dists = cov_Matern5_2(X1,X2,theta)
    elif type=="Matern3_2":
        dists = cov_Matern3_2(X1,X2,theta)
    else:
        raise ValueError(f"type: {type} not yet implemented")
    return dists

def euclidean_dist(X1,X2):
    #dists = np.sum(np.square(X1)[:,np.newaxis,:], axis=2) - 2.0 * X1.dot(X2.T) + np.sum(np.square(X2), axis=1)
    
    # a faster implementation from scipy
    dists = cdist(X1,X2, metric='euclidean')**2
    return dists

def abs_dist(X1,X2):
    dists = cdist(X1,X2,metric='minkowski', p =1)
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

def partial_cov_gen(X1, X2 = None, theta = None,k_theta_g = None, type = "Gaussian", arg = None, i1 = None, i2 = None):
    if X2 is None:
        if type== "Gaussian":
            if arg == "theta_k":
                return partial_d_C_Gaussian_dtheta_k(X1 = X1, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Gaussian_d_k_theta_g(X1 = X1, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Gaussian_dX_i_j(X1 = X1, theta = theta, i1 = i1, i2 = i2)
        if type == "Matern5_2":
            if TYPE(theta)==np.ndarray: 
                theta = theta[0]
            if arg == "theta_k":
                return partial_d_C_Matern5_2_dtheta_k(X1 = X1, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Matern5_2_d_k_theta_g(X1 = X1, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Matern5_2_dX_i_j(X1 = X1, theta = theta, i1 = i1, i2 = i2)
        if type == "Matern3_2":
            if TYPE(theta)==np.ndarray: 
                theta = theta[0]
            if arg == "theta_k":
                return partial_d_C_Matern3_2_dtheta_k(X1 = X1, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Matern3_2_d_k_theta_g(X1 = X1, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Matern3_2_dX_i_j(X1 = X1, theta = theta, i1 = i1, i2 = i2)
    else:
        if type == "Gaussian": 
            if arg == "theta_k":
                return partial_d_k_Gaussian_dtheta_k(X1 = X1, X2 = X2, theta = theta)
            if arg == "k_theta_g":
                return partial_d_kg_Gaussian_d_k_theta_g(X1 = X1, X2 = X2, theta = theta,k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_k_Gaussian_dX_i_j(X1 = X1, X2 = X2, theta = theta,i1 = i1, i2 = i2)
        if type == "Matern5_2":
            if arg == "theta_k":
                return partial_d_C_Matern5_2_dtheta_k(X1 = X1, X2 = X2, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Matern5_2_d_k_theta_g(X1 = X1, X2 = X2, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Matern5_2_dX_i_j(X1 = X1, X2 = X2, theta = theta, i1 = i1, i2 = i2)
        if type == "Matern3_2":
            if TYPE(theta)==np.ndarray: 
                theta = theta[0]
            if arg == "theta_k":
                return partial_d_C_Matern3_2_dtheta_k(X1 = X1, X2 = X2, theta = theta)
            if arg == "k_theta_g":
                return partial_d_Cg_Matern3_2_d_k_theta_g(X1 = X1, X2 = X2, theta = theta, k_theta_g=k_theta_g)
            if arg == "X_i_j":
                return partial_d_C_Matern3_2_dX_i_j(X1 = X1,X2 = X2, theta = theta, i1 = i1, i2 = i2)
            

def cov_Matern3_2(X1, X2 = None, theta = None):
    if X2 is None:
        X2 = X1
    #dlist = [abs_dist(X1[:,i],X2[:,i], theta = theta[i]) for i in np.arange(X1.shape[1])]
    dlist = [
        abs_dist(X1[:,i:i+1],X2[:,i:i+1]) 
        for i in range(X1.shape[1])
        ]
    klist = [
        (1 + np.sqrt(3)/theta[i] * dlist[i]) * np.exp(-1.0*np.sqrt(3)*dlist[i]/theta[i])
        for i in range(X1.shape[1])
        ]
    out = np.prod(klist,axis=0)
    return out

def cov_Matern5_2(X1, X2 = None, theta = None):
    if X2 is None:
        X2 = X1
    abs_dist_list = [
        abs_dist(X1[:,i:i+1],X2[:,i:i+1]) 
        for i in range(X1.shape[1])
        ]
    euclid_dist_list = [
        euclidean_dist(X1[:,i:i+1],X2[:,i:i+1]) 
        for i in range(X1.shape[1])
    ]
    klist = [
        (1 + np.sqrt(5)/theta[i] * abs_dist_list[i] + 5 * euclid_dist_list[i]/(3*theta[i]**2)) * np.exp(-1.0*np.sqrt(5)*abs_dist_list[i]/theta[i])
        for i in range(X1.shape[1])
        ]
    
    out = np.prod(klist,axis=0)
    return out

## Partial derivative of the covariance matrix with respect to theta[k] (to be multiplied by the covariance matrix)
def partial_d_C_Matern5_2_dtheta_k(X1, theta):
    if X1.shape[1] == 1:
      tmp = matern.d_matern5_2_1args_theta_k(X1 = X1, theta = theta)
    else:
      tmp = matern.d_matern5_2_1args_theta_k_iso(X1 = X1, theta = theta)
    
    return tmp

## Partial derivative of the covariance vector with respect to theta[k] (to be multiplied by the covariance vector)
def partial_d_k_Matern5_2_dtheta_k(X1, X2, theta):
    tmp = matern.d_matern5_2_2args_theta_k_iso(X1 = X1, X2 = X2, theta = theta)
    return tmp

## Partial derivative of the covariance matrix of the noise process with respect to k_theta_g (to be multiplied by the covariance matrix)
def partial_d_Cg_Matern5_2_d_k_theta_g(X1, theta, k_theta_g):
  if(isinstance(theta,np.floating) or len(theta) == 1): return matern.d_matern5_2_1args_kthetag(X1/theta, k_theta_g)
  return matern.d_matern5_2_1args_kthetag(X1 * (1/theta), k_theta_g)

## Partial derivative of the covariance matrix of the noise process with respect to k_theta_g (to be multiplied by the covariance matrix)
def partial_d_Cg_Matern5_2_d_k_theta_g(X1, theta, k_theta_g):
  if(isinstance(theta,np.floating) or len(theta) == 1): return matern.d_matern5_2_1args_kthetag(X1/theta, k_theta_g)
  return matern.d_matern5_2_1args_kthetag(X1 * 1/theta, k_theta_g)

## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
## @param i1 row
## @param i2 column
def partial_d_C_Matern5_2_dX_i_j(X1, theta, i1, i2):

  ## 1-dimensional/isotropic case
  if len(theta) == 1:
    tmp = matern.partial_d_dist_abs_dX_i1_i2(X1/theta, i1, i2)
    return tmp/theta
  
  tmp = matern.partial_d_dist_abs_dX_i1_i2(X1/theta[i2], i1, i2)
  return tmp / theta[i2]

## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
## @param i1 row
## @param i2 column
## @param theta lengthscales
def partial_d_k_Matern5_2_dX_i_j(X1, X2, theta, i1, i2):

  ## 1-dimensional/isotropic case
  if len(theta) == 1:
    tmp = matern.partial_d_dist_abs_dX1_i1_i2_X2(X1/theta, X2/theta, i1, i2)
    return tmp/theta

  tmp = matern.partial_d_dist_abs_dX1_i1_i2_X2(X1/theta[i2], X2/theta[i2], i1, i2)
  return tmp / theta[i2]


### C) Matern 3/2 covariance



## Partial derivative of the covariance matrix with respect to theta[k] (to be multiplied by the covariance matrix) 
def partial_d_C_Matern3_2_dtheta_k(X1, theta):
  if X1.shape[1] == 1: return matern.d_matern3_2_1args_theta_k(X1 = X1, theta = theta)
  return matern.d_matern3_2_1args_theta_k_iso(X1 = X1, theta = theta)


## Partial derivative of the covariance vector with respect to theta[k] (to be multiplied by the covariance vector)
def partial_d_k_Matern3_2_dtheta_k(X1, X2, theta):
  tmp = matern.d_matern3_2_2args_theta_k_iso(X1 = X1, X2 = X2, theta = theta)
  return tmp


## Partial derivative of the covariance matrix of the noise process with respect to k_theta_g (to be multiplied by the covariance matrix)
def partial_d_Cg_Matern3_2_d_k_theta_g(X1, theta, k_theta_g):
    if(isinstance(theta,np.floating) or len(theta) == 1): return matern.d_matern3_2_1args_kthetag(X1/theta, k_theta_g)
    return matern.d_matern3_2_1args_kthetag(X1 * 1/theta, k_theta_g)


## Partial derivative of the covariance vector of the noise process with respect to k_theta_g (to be multiplied by the covariance vector)
def partial_d_kg_Matern3_2_d_k_theta_g(X1, X2, theta, k_theta_g):
  if(isinstance(theta,np.floating) or len(theta) == 1): return matern.d_matern3_2_2args_kthetag(X1/theta, X2/theta, k_theta_g)
  return matern.d_matern3_2_2args_kthetag(X1/theta, X2/theta, k_theta_g)


## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
## @param i1 row
## @param i2 column
def partial_d_C_Matern3_2_dX_i_j(X1, theta, i1, i2):
  
  ## 1-dimensional/isotropic case
  if len(theta) == 1:
    tmp = matern.partial_d_dist_abs_dX_i1_i2_m32(X1/theta, i1, i2)
    return(tmp/theta)
  
  tmp = matern.partial_d_dist_abs_dX_i1_i2_m32(X1/theta[i2], i1, i2)
  return(tmp / theta[i2])

## Derivative with respect to X[i,j]. Useful for pseudo inputs, to be multiplied by the covariance matrix
## @param i1 row
## @param i2 column
## @param theta lengthscales
def partial_d_k_Matern3_2_dX_i_j(X1, X2, theta, i1, i2):
  
  ## 1-dimensional/isotropic case
  if len(theta) == 1:
    tmp = matern.partial_d_dist_abs_dX1_i1_i2_X2_m32(X1/theta, X2/theta, i1, i2)
    return(tmp/theta)
  
  tmp = matern.partial_d_dist_abs_dX1_i1_i2_X2_m32(X1/theta[i2], X2/theta[i2], i1, i2)
  return(tmp / theta[i2])



