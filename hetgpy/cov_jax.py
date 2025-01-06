# again, this only works on startup!
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp



def euclidean_dist(X,Y):
    return((X - Y[:,None])**2).sum(axis=-1)

def abs_dist(X,Y):
    return(((X - Y[:,None])**2)**0.5).sum(axis=-1)

def cov_Gaussian(X1,X2,theta):
    if len(theta)==1:
            X2 = X1 if X2 is None else X2
            return jnp.exp(-euclidean_dist(X1,X2)/theta)
    if X2 is None:
        A = X1 * 1.0 / jnp.sqrt(theta)
        return jnp.exp(-1.0*euclidean_dist(A,A))
    else:
        A = X1 * 1.0 / jnp.sqrt(theta)
        B = X2 * 1.0 / jnp.sqrt(theta)
        return jnp.exp(-1.0*euclidean_dist(A,B))

def cov_Matern5_2(X1,X2,theta):
    # from sklearn.gaussian_process.kernels
    dists = jnp.abs(X1/theta-X2[:,None]/theta)
    K = dists * jnp.sqrt(5)
    K = ((1.0 + K + K**2 / 3.0) * jnp.exp(-K)).prod(axis=-1)
    return K

def cov_Matern3_2(X1,X2,theta):
    dists = jnp.abs(X1/theta-X2[:,None]/theta)
    K = dists * jnp.sqrt(3)
    K = ((1.0 + K) * jnp.exp(-K)).prod(axis=-1)
    return K

def cov_gen_jax(X1,X2 = None,theta = None,type = "Gaussian"):
     types = ('Gaussian','Matern5_2','Matern3_2')
     if type not in types:
          raise ValueError(f'{type} not in {types}')
     if X2 is None:
          X2 = X1.copy()
     theta = jnp.array(theta)
     if type=='Gaussian':
          return cov_Gaussian(X1,X2,theta)
     if type=='Matern5_2':
          return cov_Matern5_2(X1,X2,theta)
     if type=='Matern3_2':
          return cov_Matern3_2(X1,X2,theta)
