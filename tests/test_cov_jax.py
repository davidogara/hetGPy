from hetgpy.covariance_functions import cov_gen
from hetgpy.cov_jax import cov_gen_jax
import jax.numpy as jnp
from jax import random
key = random.key(0)
nr, nc = 10,3
X1d = random.uniform(key,shape=(nr,1))
X3d = random.uniform(key,shape=(nr,nc))



def compare(X,theta,ctype):
    out1 = cov_gen(X,X,theta,type=ctype)
    out2 = cov_gen_jax(X,X,theta,type=ctype)
    assert jnp.allclose(out1,out2)

def test_gauss():
    # 1 col, iso
    compare(X1d,theta=jnp.array([2]),ctype='Gaussian')
    # 3 col, iso
    compare(X3d,theta=jnp.array([2]),ctype='Gaussian')
    # 3 col, aniso
    compare(X3d,theta=jnp.array([2,3,4]),ctype='Gaussian')
def test_mat52():
    # 1 col, iso
    compare(X1d,theta=jnp.array([2]),ctype='Matern5_2')
    # 3 col, iso
    compare(X3d,theta=jnp.array([2]),ctype='Matern5_2')
    # 3 col, aniso
    compare(X3d,theta=jnp.array([2,3,4]),ctype='Matern5_2')

def test_mat32():
    # 1 col, iso
    compare(X1d,theta=jnp.array([2]),ctype='Matern3_2')
    # 3 col, iso
    compare(X3d,theta=jnp.array([2]),ctype='Matern3_2')
    # 3 col, aniso
    compare(X3d,theta=jnp.array([2,3,4]),ctype='Matern3_2')

if __name__ == "__main__":
    compare(X1d,theta=jnp.array([2]),ctype='Matern3_2')