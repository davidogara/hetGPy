import matern
from scipy.io import loadmat

m = loadmat('d_matern5_2_1args_theta_k.mat')
theta = 5.871215
tmp = matern.d_matern5_2_1args_theta_k(m['X1'],theta)
foo=1